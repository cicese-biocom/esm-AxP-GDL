import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Union, Optional, Any, Tuple
import pandas as pd
import torch
from pandas import DataFrame
from pydantic.v1 import PositiveInt, PositiveFloat
from tensorboardX import SummaryWriter
from torch import nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from torch_geometric.data import Data

from src.architectures.gnn import GNNParameters
from src.config.types import (
    ValidationMode,
    ESM2ModelForContactMap,
    ESM2Representation,
    EdgeBuildFunction,
    ModelingTask,
    GDLArchitecture, ExecutionMode, DistanceFunction
)

from src.applicability_domain.methods import build_model
from src.data_processing.data_partitioner import to_partition
from src.feature_extraction.features import filter_features, FeaturesContext, FeatureCalculationParameters
from src.graph_builder.graph_builder import build_graphs, BuildGraphsParameters
from src.modeling.executor import (
    ModelTrainer,
    ModelValidator,
    ModelTester,
    Output,
    ModeExecutor,
    ModelInference
)
from src.modeling.prediction_maker import Prediction
from src.modeling.model_selector import Model
from src.utils.base_parameters import BaseParameters
from src.workflow.app_context import ApplicationContext
from src.workflow.logging import Logging
from src.workflow.params_setup import ExecutionParameters


class ModelParameters(BaseParameters):
    esm2_representation: ESM2Representation
    edge_build_functions: List[EdgeBuildFunction]
    distance_function: Optional[DistanceFunction]
    distance_threshold: Optional[PositiveFloat]
    esm2_model_for_contact_map: Optional[ESM2ModelForContactMap]
    probability_threshold: Optional[PositiveFloat]
    amino_acid_representation: str
    hidden_layer_dimension: PositiveInt
    add_self_loops: bool
    use_edge_attr: bool
    dropout_rate: PositiveFloat
    validation_mode: Optional[ValidationMode]
    randomness_percentage: Optional[PositiveFloat]
    modeling_task: ModelingTask
    numbers_of_class: PositiveInt
    classes: Optional[List[int]]
    gdl_architecture: GDLArchitecture


class GDLWorkflow(ABC):
    def __init__(self, execution_mode: ExecutionMode):
        self._parameters = ExecutionParameters(execution_mode).get_parameters()
        self._context = ApplicationContext(**self._parameters.dict())
        self._execute_output: Optional[List[Output]] = []
        self._path: Dict = self._parameters.output_dir

    def run(self):
        # Step 0: Init logger
        self.init_logger()

        # Step 1: Build an applicability domain model
        ad_models = self.build_models_for_applicability_domain()

        # Step 2: When `prediction_batch_size` is set, data loading is performed in chunked batches 
        # rather than loading the entire dataset at once.
        data = self.load_dataset()

        calculated_features = []
        outputs: List[Output] = []
        domains: List[pd.DataFrame] = []
        for i, batch in enumerate(data):
            # Step 3: Configure the names of the output files
            self.config_output_file_names(substr=str(i + 1))

            # Step 4: # The dataset processor performs the following validation and cleaning steps:
            # # 1. Detects and reports duplicated sequence identifiers.
            # # 2. Detects and reports duplicated peptide sequences.
            # # 3. Filters out sequences containing non-natural amino acids and saves the cleaned dataset.
            # # 4. Identifies sequences with invalid activity values according to the provided class validator.
            # # 5. Detects sequences assigned to unsupported or inconsistent partition labels.
            batch = self.process_dataset(batch)

            # Skip iteration if the batch is empty
            if batch.empty:
                continue

            # Step 5: Build graphs
            graphs, perplexities = self.build_graphs(data=batch)

            # Step 6: Calculate sequence and graph features for applicability domain and data partitioning
            features = self.calculate_sequence_and_graph_features(data=batch, graphs=graphs, perplexities=perplexities)
            calculated_features.append(features)

            # Step 7: Partition the graphs into training and validation sets.
            # This step is performed exclusively in training mode.
            partitioned_graphs = self.get_graphs_per_partition(data=batch, graphs=graphs, features=features)

            # Step 8: Initialize GNN model
            model = self.init_gnn_model(graphs=partitioned_graphs)

            # Step 9: Build the executor objects required for the current execution mode (e.g., trainer, validator)
            executors = self.build_executors(model)

            # Step 10: Execute training, testing or inference mode
            output = self.execute(executor=executors, graphs=partitioned_graphs)
            outputs.append(output)

            # Step 11: Calculate applicability domain
            domain = self.calculate_applicability_domain(ad_models, features)
            if domain is not None and not domain.empty:
                domains.append(domain)

        # Step 12: Save features
        self.save_sequence_and_graph_features(calculated_features)

        # Step 12: Prepare and save prediction outputs.
        # This step is executed exclusively during testing and inference.
        self.prepare_and_save_predictions(outputs, domains)
        self.prepare_and_save_predictions(outputs, domains)

        # Step 13: Compute and save executeuation metrics.
        # Metrics are calculated only in testing mode.
        self.calculate_and_save_metrics(outputs)

    def save_sequence_and_graph_features(self, calculated_features):
        file_path = self._path['features_file']
        pd.concat(calculated_features).to_csv(file_path, index=False)
        logging.getLogger('workflow_logger').info(f"Features saved to: {file_path}")

    def init_logger(self):
        Logging.init(
            config_file=Path(os.getenv("LOG_CONFIG_FILE")).resolve(),
            output_dir=self._path.get('log_file')
        )

    @abstractmethod
    def execute(self, executor: Union[Tuple[Any, Any], Any], graphs: List) -> Output:
        pass

    def build_models_for_applicability_domain(self):
        pass

    def load_dataset(self):
        return self._context.dataset_loader.read_file(self._parameters.dataset)

    def config_output_file_names(self, substr: str):
        pass

    def process_dataset(self, data: pd.DataFrame) -> pd.DataFrame:
        return self._context.dataset_processor.process(
            dataset=data,
            output_dir=self._path,
            target_feature_validator=self._context.target_feature_validator,
            classes=self._parameters.classes
        )

    def calculate_sequence_and_graph_features(self, data: pd.DataFrame, graphs: List[Data], perplexities: pd.DataFrame) -> pd.DataFrame:
        return FeaturesContext().compute(
            **FeatureCalculationParameters(
                features_to_calculate=self._parameters.feature_types_for_ad,
                data=data,
                graphs=graphs,
                perplexities=perplexities,
                device=self._parameters.device
            ).dict()
        )

    def get_graphs_per_partition(self, data, graphs, features):
        return graphs

    @abstractmethod
    def init_gnn_model(self, graphs):
        pass

    def calculate_applicability_domain(self, ad_models, features) -> pd.DataFrame:
        pass

    def build_graphs(self, data):
        return build_graphs(
            BuildGraphsParameters(
                **self._parameters.dict(),
                non_pdb_bound_sequences_file=self._path.get('non_pdb_bound_sequences_file'),
                data=data
            )
        )

    def compute_metrics(self):
        pass

    @abstractmethod
    def build_executors(self, model: nn.Module):
        pass

    def calculate_output(self, outputs: List[Output], domain: List[pd.DataFrame]) -> pd.DataFrame:
        pass

    def _calculate_predictions(self, outputs: List[Output]):
        pass

    def _add_applicability_domain_to_predictions(self, domains, predictions):
        pass

    def prepare_and_save_predictions(self, outputs: List[Output], domains: List[pd.DataFrame]) -> None:
        pass

    def calculate_and_save_metrics(self, outputs: List[Output]) -> None:
        pass

class TrainingWorkflow(GDLWorkflow):
    def process_dataset(self, data: pd.DataFrame) -> pd.DataFrame:
        data = super().process_dataset(data)

        # keeping only the instances belonging to the training and validation sets
        data = data[data['partition'].isin([1, 2])].reset_index(drop=True)

        if data.query('partition == 1').empty:
            logging.getLogger('workflow_logger').critical(
                "The dataset does not contain training set."
            )
            quit()

        if data.query('partition == 2').empty:
            if self._parameters.split_method and self._parameters.split_training_fraction:
                data.drop(['partition'], axis=1, inplace=True)
            elif not self._parameters.split_method:
                logging.getLogger('workflow_logger').critical(
                    "The dataset does not contain validation set; the parameter 'split_method' must be specified."
                )
                quit()
            elif not self._parameters.split_training_fraction:
                logging.getLogger('workflow_logger').critical(
                    "The dataset does not contain validation set; the parameter 'split_training_fraction' must be specified."
                )
                quit()
        # if not (data.query('partition == 1').empty and data.query('partition == 2').empty)
        elif self._parameters.split_method or self._parameters.split_training_fraction:
            if self._parameters.split_method:
                logging.getLogger('workflow_logger').critical(
                    "The dataset contains training and validation sets; the parameter split_method must not be specified."
                )
                quit()
            elif self._parameters.split_training_fraction:
                logging.getLogger('workflow_logger').critical(
                    "The dataset contains training and validation sets; the parameter split_training_fraction must not be specified."
                )
                quit()

        return data

    def _partition_data_training_and_validation(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        # partitioning data in training and validation
        data = to_partition(
            split_method=self._parameters.split_method,
            data=data,
            features=features,
            split_training_fraction=self._parameters.split_training_fraction
        )

        filtered_data = data[['id', 'sequence', 'activity', 'partition']]

        # Save training and validation data to csv
        _save_partitioned_data_to_csv(filtered_data, self._path['data_csv'])

        # Save training data to fasta
        training_data = filtered_data[filtered_data['partition'] == 1]
        _save_partitioned_data_to_fasta(training_data, self._path['training_data_fasta'])

        # Save validation data to fasta
        validation_data = filtered_data[filtered_data['partition'] == 2]
        _save_partitioned_data_to_fasta(validation_data, self._path['validation_data_fasta'])

        return data

    def get_graphs_per_partition(self, data: pd.DataFrame, graphs: List, features: pd.DataFrame) -> tuple[list[Any], list[Any]]:
        if 'partition' not in data.columns:
            data = self._partition_data_training_and_validation(data, features)

        partitions = data['partition']

        train_graphs = [g for g, p in zip(graphs, partitions) if p == 1]
        val_graphs = [g for g, p in zip(graphs, partitions) if p == 2]
        
        return train_graphs, val_graphs

    def init_gnn_model(self, graphs):
        train_graphs, val_graphs = graphs

        return self._context.gnn_factory.create(
            GNNParameters(
                **self._parameters.dict(),
                node_feature_dim=train_graphs[0].x.shape[1]
            )
        ).to(device=self._parameters.device)

    def build_executors(self, model: nn.Module) -> Union[Tuple[ModeExecutor, ModeExecutor], ModeExecutor]:
        model_trainer = self._build_model_trainer(model)
        model_validator = self._build_model_validator(model_trainer)

        return model_trainer, model_validator

    def _build_model_validator(self, model_trainer):
        model_validator = ModelValidator(
            model=model_trainer.model,
            device=self._parameters.device,
            loss_fn=self._context.loss_fn,
            prediction=self._context.prediction_maker
        )
        return model_validator

    def _build_model_trainer(self, model):
        model_trainer = ModelTrainer(
            model=model,
            device=self._parameters.device,
            loss_fn=self._context.loss_fn,
            learning_rate=self._parameters.learning_rate,
            weight_decay=self._parameters.weight_decay,
            step_size=self._parameters.step_size,
            gamma=self._parameters.gamma
        )
        return model_trainer

    def execute(self, executor: Union[Tuple[Any, Any], Any], graphs: List) -> None:       
        # Step: get executors
        model_trainer, model_validator = executor

        # Step: create training y validation dataloader
        train_graphs, val_graphs = graphs
        train_data = DataLoader(dataset=train_graphs, batch_size=self._parameters.batch_size)
        val_data = DataLoader(dataset=val_graphs, batch_size=self._parameters.batch_size)

        # Step: init variables
        checkpoints_path = self._path['checkpoints_path']
        best_model = Model()
        current_model = Model()
        metrics = []

        # Step: training and validation
        for epoch in tqdm(range(1, self._parameters.number_of_epochs + 1), desc="Training model"):
            # Step: training model
            train_output = model_trainer.execute(train_data)

            # Step: execute model
            val_output = model_validator.execute(val_data)

            # Step computes validation metrics
            val_metrics = self._context.metrics_calculator.calculate(prediction=val_output.prediction, y_true=val_output.y_true)

            # Step: save the current model (save in disk if save_model_per_epoch=True)
            current_model = Model(
                epoch=epoch,
                model=train_output.model,
                model_state_dict=train_output.model.state_dict(),
                parameters=ModelParameters(**self._parameters.dict()).dict(),
                metrics=val_metrics,
                train_loss=train_output.loss,
                val_loss=val_output.loss,
                optimizer_state_dict=train_output.optimizer_state_dict,
                scheduler_state_dict=train_output.scheduler_state_dict
            )

            if self._parameters.save_ckpt_per_epoch:
                _save_checkpoint(current_model, checkpoints_path)

            # Step: update the best model
            best_model = self._context.model_selector.select(best_model, current_model)

            # Step: format and accumulate in-memory validation metrics for the current epoch
            metrics.append(
                {
                    "Epoch": epoch,
                    "Loss/Train": train_output.loss,
                    "Loss/Val": val_output.loss,
                    **_format_metric(metrics=val_metrics, suffix="/Val")
                }
            )

        # Step: save models
        self._save_checkpoint(best_model, checkpoints_path, current_model)

        # Step: save metrics
        self._save_metrics(metrics)

    def _save_metrics(self, metrics):
        # to csv
        _save_metrics_to_csv(metrics, self._path.get('metrics_file'))
        
        # to tensorboard
        _save_metrics_to_tb(metrics, self._path.get('metrics_path'))

    def _save_checkpoint(self, best_model, checkpoints_path, current_model):
        logger = logging.getLogger('workflow_logger')
        
        if not self._parameters.save_ckpt_per_epoch:
            # Save the model of the last epoch
            filename = _save_checkpoint(current_model, checkpoints_path)
            logger.info(f"Saved model of the last epoch at {filename}")

            # save the best model
            filename = _save_best_model(best_model, checkpoints_path)
            logger.info(f"Saved best model at {filename}")
        else:
            logger.info(f"Saved models per epoch in directory {checkpoints_path}")

            # Mark the best model
            filename = _mark_best_model(best_model, checkpoints_path)
            logger.info(f"Saved model with the best MCC at {filename}")


class PredictionWorkflow(GDLWorkflow, ABC):
    def build_models_for_applicability_domain(self):
        if self._parameters.get_ad is None:
            return None
        
        features_to_build_domain = pd.read_csv(self._parameters.feature_file_for_ad)

        ad_models = []
        with tqdm(total=len(self._parameters.methods_for_ad),
                  desc="Building applicability domain models",
                  disable=False) as progress:
            for method_for_ad in self._parameters.methods_for_ad:
                features_selected = filter_features(features_to_build_domain, method_for_ad['features'])
                model_for_ad = build_model(method_for_ad['method_name'], features_selected)

                ad_models_dict = {
                    'method': method_for_ad,
                    'model': model_for_ad,
                }
                ad_models.append(ad_models_dict)

                progress.update()
        return ad_models

    def calculate_sequence_and_graph_features(self, data: pd.DataFrame, graphs: List[Data], perplexities: pd.DataFrame) -> Optional[DataFrame]:
        if self._parameters.get_ad is None:
            return None

        return super().calculate_sequence_and_graph_features(data, graphs, perplexities)

    def init_gnn_model(self, graphs: List):
        checkpoint = torch.load(self._parameters.gdl_model_path)
        model = checkpoint['model']
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self._parameters.device)
        return model

    def execute(self, executor: Union[Tuple[Any, Any], Any], graphs: List) -> Output:
        data = DataLoader(dataset=graphs, batch_size=self._parameters.batch_size)
        return executor.execute(data)

    def calculate_applicability_domain(self, ad_models: List[Dict], features: pd.DataFrame) -> pd.DataFrame:
        domain = pd.DataFrame()
        if self._parameters.get_ad:
            instance_id = features.iloc[:, 0]
            features_to_execute = features.iloc[:, 1:]

            with tqdm(total=len(ad_models),
                      desc="Getting applicability domain", disable=False) as progress:
                for ad_model in ad_models:
                    method_for_ad = ad_model['method']
                    model_for_ad = ad_model['model']

                    features_to_execute_selected = filter_features(features_to_execute, method_for_ad['features'])
                    outlier, outlier_score = model_for_ad.execute_model(features_to_execute_selected)

                    temp_domain = pd.DataFrame({
                        f"{method_for_ad['method_id']}": ['out' if x == -1 else 'in' for x in outlier],
                        f"{method_for_ad['method_id']}_score": outlier_score
                    })

                    domain = pd.concat([domain, temp_domain], axis=1)
                    progress.update(1)
            domain['sequence'] = instance_id

        return domain
    
    def _add_applicability_domain_to_predictions(self, domains, predictions):
        # merge prediction to applicability domain
        if self._parameters.get_ad:
            predictions = (pd.merge(
                predictions,
                pd.concat(domains, axis=1),
                on='sequence',
                how='inner'
            ))
        return predictions

    def prepare_and_save_predictions(self, outputs: List[Output], domains: List[pd.DataFrame]) -> None:       
        output_predictions = self._calculate_predictions(outputs)
        output_predictions = self._add_applicability_domain_to_predictions(domains, output_predictions)

        file_path = self._path.get('prediction_file')
        output_predictions.to_csv(file_path, index=False)
        logging.getLogger('workflow_logger').info(f"Predictions saved to: {file_path}")

    @abstractmethod
    def _calculate_predictions(self, outputs: List[Output]):
        pass

class TestWorkflow(PredictionWorkflow):
    def process_dataset(self, data: pd.DataFrame) -> pd.DataFrame:
        data = super().process_dataset(data)
        return data[data['partition'].isin([3])].reset_index(drop=True)

    def build_executors(self, model: nn.Module):
        return ModelTester(
            model=model,
            device=self._parameters.device,
            prediction=self._context.prediction_maker,
        )

    def execute(self, executor: Union[Tuple[Any, Any], Any], graphs: List) -> Output:
        return super().execute(executor, graphs)

    def calculate_and_save_metrics(self, outputs: List[Output]) -> None:
        y_true = get_y_true(outputs)
        predictions = get_predictions(outputs)

        metrics = self._context.metrics_calculator.calculate(prediction=predictions, y_true=y_true)
        metrics = _format_metric(metrics, suffix="/Test")
        
        file_path = self._path.get('metrics_file')
        _save_metrics_to_csv([metrics], file_path)        
        logging.getLogger('workflow_logger').info(f"Metrics saved to: {file_path}")

    def _calculate_predictions(self, outputs: List[Output]):
        return pd.DataFrame(
            {
                **get_sequence_info(outputs),
                'y_true': get_y_true(outputs),
                **get_predictions(outputs).dict()
            }
        )

class InferenceWorkflow(PredictionWorkflow):

    def config_output_file_names(self, substr: str):
        path = Path(self._path['prediction_path'])
        self._path['prediction_file'] = path.joinpath(f"Prediction-batch_{substr}.csv")

        path = Path(self._path['features_path'])
        self._path['features_file'] = path.joinpath(f"Features-batch_{substr}.csv")

    def build_executors(self, model: nn.Module):
        return ModelInference(
                model=model,
                device=self._parameters.device,
                prediction=self._context.prediction_maker
            )

    def execute(self, executor: Union[Tuple[Any, Any], Any], graphs: List) -> Output:
        return super().execute(executor, graphs)
        
    def _calculate_predictions(self, outputs: List[Output]):
        return pd.DataFrame(
            {
                **get_sequence_info(outputs),
                **get_predictions(outputs).dict()
            }
        )

# auxiliar methods
def _save_partitioned_data_to_csv(data: pd.DataFrame, csv_file: Path):
    data.to_csv(csv_file, index=False)
    logging.getLogger('workflow_logger'). \
        info(f"Partitioned dataset saved to CSV file. See: {csv_file}")
    
    
def _save_partitioned_data_to_fasta(df: pd.DataFrame, fasta_file: Path):
    fasta_records = []
    for i, row in df.iterrows():
        sequence_id = row['id']
        sequence = row['sequence']
        activity = int(row['activity'])
        record_id = f"{sequence_id}_class_{activity}"
        record = SeqRecord(Seq(sequence), id=record_id, description="")
        fasta_records.append(record)

    with open(fasta_file, 'w') as output_handle:
        SeqIO.write(fasta_records, output_handle, 'fasta')

    logging.getLogger('workflow_logger'). \
        info(f"Training sequences saved to FASTA file. See: {fasta_file}")


def _save_metrics_to_csv(metrics: List[dict], csv_file: Path):
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(csv_file, index=False)


def _save_metrics_to_tb(metrics: List[dict], metrics_path: Path):
    tb_writer = SummaryWriter(log_dir=str(metrics_path), filename_suffix="_metrics")

    # write tensorboard metrics, excluding 'epoch'
    for entry in metrics:
        epoch = entry.get("epoch")
        for key, value in entry.items():
            if key != "epoch":
                tb_writer.add_scalar(key, value, global_step=epoch)


def _save_checkpoint(model: Model, checkpoints_path: Path):
    filename = _get_checkpoint_name(model, checkpoints_path)
    torch.save(model.dict(), filename)
    return filename


def _save_best_model(model: Model, checkpoints_path: Path):
    filename = _get_checkpoint_name(model, checkpoints_path)
    filename = filename.with_name(filename.stem + "_(best).pt")
    torch.save(model.dict(), filename)
    return filename


def _get_checkpoint_name(model: Model, checkpoints_path: Path):
    return checkpoints_path.joinpath(
        f"epoch={model.epoch}_train-loss={model.train_loss:.2f}_val-loss={model.val_loss:.2f}.pt"
    )


def _mark_best_model(best_model: Model, checkpoints_path: Path) -> Path:
    original_name = f"epoch={best_model.epoch}_train-loss={best_model.train_loss:.2f}_val-loss={best_model.val_loss:.2f}.pt"
    original_path = checkpoints_path.joinpath(original_name)

    new_path = original_path.with_name(original_path.stem + "_(best).pt")

    if original_path.exists():
        original_path.replace(new_path)
        return new_path
    else:
        raise FileNotFoundError(f"Checkpoint not found: {original_path}")


def _format_metric(metrics: dict, suffix: str) -> dict:
    return {f"{key}{suffix}": value for key, value in metrics.items()}


def get_y_true(outputs: List[Output]):
    y_true = []
    for execute_output in outputs:
        y_true.extend(execute_output.y_true)

    return y_true


def get_sequence_info(outputs: List[Output]):
    sequence_info = {}
    for execute_output in outputs:
        for key, value in execute_output.sequence_info.items():
            if key not in sequence_info:
                sequence_info[key] = []
            sequence_info[key].extend(value)

    return sequence_info


def get_predictions(outputs: List[Output]):
    predictions = Prediction()
    for execute_output in outputs:
        predictions.add_new_predictions(execute_output.prediction)
    return predictions