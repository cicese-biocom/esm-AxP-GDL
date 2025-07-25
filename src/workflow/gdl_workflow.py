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

from src.architecture.gnn import GNNParametersDTO
from src.config.types import (
    ValidationMode,
    ESM2ModelForContactMap,
    ESM2Representation,
    EdgeConstructionFunctions,
    DistanceFunction,
    ModelingTask,
    GDLArchitecture
)

from src.applicability_domain.methods import build_model
from src.data_processing.data_partitioner import to_partition
from src.feature_extraction.features import filter_features, FeaturesContext, FeatureDTO
from src.graph_construction.graphs import construct_graphs, ConstructGraphDTO
from src.modeling.evaluator import TrainingModeEvaluator, ValidationModeEvaluator, TestModeEvaluator, \
    EvaluationOutputDTO, Evaluator, InferenceModeEvaluator
from src.modeling.prediction import PredictionDTO
from src.modeling.selector import ModelDTO
from src.utils.dto import DTO
from src.workflow.app_context import ApplicationContext
from src.workflow.logging import Logging
from src.workflow.params_setup import TrainingArguments, PredictionArguments, InferenceArguments


class ModelParametersDTO(DTO):
    esm2_representation: ESM2Representation
    edge_construction_functions: List[EdgeConstructionFunctions]
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


def format_metric_keys(metrics: dict, suffix: str) -> dict:
    return {f"{key}{suffix}": value for key, value in metrics.items()}


class GDLWorkflow(ABC):
    def __init__(self, parameters):
        self._parameters: Union[TrainingArguments, PredictionArguments, InferenceArguments] = parameters
        self._context = ApplicationContext(**parameters.dict())

    def run(self):
        # Step 0: Init logging
        self.init_logging()

        # Step 1: Build applicability domain model
        ad_models = self.build_applicability_domain_model()

        # Step 2: load data
        data = self.load_data()

        for i, batch in enumerate(data):
            # Step 3: generate files names
            self.set_output_filename(substr=str(i + 1))

            # Step 4: validate dataset
            batch = self.validate_dataset(batch)

            # Step 5: construct graphs
            graphs, perplexities = self.construct_graphs(data=batch)

            # Step 6: compute feature
            features = self.computing_features(data=batch, graphs=graphs, perplexities=perplexities)

            # Step 7: Get graphs by partition
            graphs = self.split_data(graphs=graphs, data=batch, features=features)

            # Step 8: init GNN model
            model = self.init_gnn_model(graphs=graphs)

            # Step 9: init evaluator
            evaluator = self.init_model_evaluator(model)

            # Step 9: execute modeling task
            eval_output = self.modeling_task(model_evaluator=evaluator, graphs=graphs)

            # Step 10: save prediction
            self.save_prediction(eval_output=eval_output)

            # Step 10: modeling_task modeling task
            self.get_applicability_domain(ad_models, features)

        self.compute_metrics()

    def save_prediction(self, eval_output: EvaluationOutputDTO):
        pass

    def init_logging(self):
        Logging.init(
            config_file=Path(os.getenv("LOG_CONFIG_FILE")).resolve(),
            output_dir=self._parameters.output_dir.get('log_file')
        )

    @abstractmethod
    def modeling_task(self, model_evaluator: Union[Tuple[Evaluator, Evaluator], Evaluator], graphs: List) -> EvaluationOutputDTO:
        pass

    @abstractmethod
    def build_applicability_domain_model(self):
        pass

    def load_data(self):
        return self._context.data_loader.read_file(self._parameters.dataset)

    @abstractmethod
    def set_output_filename(self, substr: str):
        pass

    def validate_dataset(self, data: pd.DataFrame) -> pd.DataFrame:
        return self._context.dataset_validator.processing_dataset(
            dataset=data,
            output_dir=self._parameters.output_dir,
            class_validator=self._context.class_validator,
            classes=self._parameters.classes
        )

    def computing_features(self, data: pd.DataFrame, graphs: List[Data], perplexities: pd.DataFrame) -> pd.DataFrame:
        features = FeaturesContext().compute_features(
            **FeatureDTO(
                features_to_calculate=self._parameters.feature_types_for_ad,
                data=data,
                graphs=graphs,
                perplexities=perplexities,
                device=self._parameters.device
            ).dict()
        )

        features.to_csv(self._parameters.output_dir['features_file'], index=False)
        return features

    def split_data(self, graphs, data, features):
        return graphs

    @abstractmethod
    def init_gnn_model(self, graphs):
        pass

    @abstractmethod
    def get_applicability_domain(self, ad_models, features):
        pass

    def construct_graphs(self, data):
        return construct_graphs(
            ConstructGraphDTO(
                **self._parameters.dict(),
                non_pdb_bound_sequences_file=self._parameters.output_dir.get('non_pdb_bound_sequences_file'),
                data=data
            )
        )

    def compute_metrics(self):
        pass

    @abstractmethod
    def init_model_evaluator(self, model: nn.Module):
        pass


class TrainingWorkflow(GDLWorkflow):
    def compute_metrics(self):
        pass

    def get_applicability_domain(self, ad_models, features):
        pass

    def build_applicability_domain_model(self):
        pass

    def set_output_filename(self, substr: str):
        pass

    def validate_dataset(self, data: pd.DataFrame) -> pd.DataFrame:
        data = super().validate_dataset(data)

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

    def split_data(self, graphs: List, data: pd.DataFrame, features: pd.DataFrame) -> tuple[list[Any], list[Any]]:
        if 'partition' not in data.columns:
            # partitioning data in training and validation
            data = to_partition(
                split_method=self._parameters.split_method,
                data=data,
                features=features,
                split_training_fraction=self._parameters.split_training_fraction
            )

            # Save to csv
            csv_file = self._parameters.output_dir['data_csv']
            filtered_data = data[['id', 'sequence', 'activity', 'partition']]
            filtered_data.to_csv(csv_file, index=False)
            logging.getLogger('workflow_logger'). \
                info(f"Partitioned dataset saved to CSV file. See: {csv_file}")

            # Save to fasta
            # training data
            train_data = filtered_data[filtered_data['partition'] == 1]
            fasta_file = self._parameters.output_dir['training_data_fasta']
            save_to_fasta(train_data, fasta_file)
            logging.getLogger('workflow_logger'). \
                info(f"Training sequences saved to FASTA file. See: {fasta_file}")

            # validation data
            val_data = filtered_data[filtered_data['partition'] == 2]
            fasta_file = self._parameters.output_dir['validation_data_fasta']
            save_to_fasta(val_data, fasta_file)
            logging.getLogger('workflow_logger'). \
                info(f"Validation sequences saved to FASTA file. See: {fasta_file}")

        partitions = data['partition']
        train_graphs = []
        val_graphs = []

        for graph, partition in zip(graphs, partitions):
            if partition == 1:
                train_graphs.append(graph)
            elif partition == 2:
                val_graphs.append(graph)

        return train_graphs, val_graphs
    
    def init_gnn_model(self, graphs):
        train_graphs, val_graphs = graphs

        return self._context.gnn_factory.create(
            GNNParametersDTO(
                **self._parameters.dict(),
                node_feature_dim=train_graphs[0].x.shape[1]
            )
        ).to(device=self._parameters.device)

    def init_model_evaluator(self, model: nn.Module) -> Union[Tuple[Evaluator, Evaluator], Evaluator]:
        training_evaluator = TrainingModeEvaluator(
            model=model,
            device=self._parameters.device,
            loss_fn=self._context.loss_fn,
            learning_rate=self._parameters.learning_rate,
            weight_decay=self._parameters.weight_decay,
            step_size=self._parameters.step_size,
            gamma=self._parameters.gamma
        )

        validation_evaluator = ValidationModeEvaluator(
            model=training_evaluator.model,
            device=self._parameters.device,
            loss_fn=self._context.loss_fn,
            prediction=self._context.prediction_processor
        )

        return training_evaluator, validation_evaluator
    
    def modeling_task(self, model_evaluator: Union[Tuple[Evaluator, Evaluator], Evaluator], graphs: List) -> EvaluationOutputDTO:
        # Step: get evaluators
        training_evaluator, validation_evaluator = model_evaluator

        # Step: create training y validation dataloader
        train_graphs, val_graphs = graphs
        train_data = DataLoader(dataset=train_graphs, batch_size=self._parameters.batch_size)
        val_data = DataLoader(dataset=val_graphs, batch_size=self._parameters.batch_size)

        # Step: init variables
        checkpoints_path = self._parameters.output_dir['checkpoints_path']
        best_model = ModelDTO()
        current_model = ModelDTO()
        metrics = []

        # Step: training and validation
        for epoch in tqdm(range(1, self._parameters.number_of_epochs + 1), desc="Training model"):
            # Step: training model
            train_output = training_evaluator.eval(train_data)

            # Step: eval model
            val_output = validation_evaluator.eval(val_data)

            # Step compute validation metrics
            val_metrics = self._context.metrics.calculate(
                    prediction=val_output.prediction,
                    y_true=val_output.y_true
                )

            # Step: save current model (save in disk if save_model_per_epoch=True)
            current_model = ModelDTO(
                epoch=epoch,
                model=train_output.model,
                model_state_dict=train_output.model.state_dict(),
                parameters=ModelParametersDTO(**self._parameters.dict()).dict(),
                metrics=val_metrics,
                train_loss=train_output.loss,
                val_loss=val_output.loss,
                optimizer_state_dict=train_output.optimizer_state_dict,
                scheduler_state_dict=train_output.scheduler_state_dict
            )

            if self._parameters.save_ckpt_per_epoch:
                save_checkpoints(current_model, checkpoints_path)

            # Step: update best model
            best_model = self._context.best_model_selector.select(best_model, current_model)

            # Step: save validation metrics
            metrics.append(
                {
                    "Epoch": epoch,
                    "Loss/Train": train_output.loss,
                    "Loss/Val": val_output.loss,
                    **format_metric_keys(metrics=val_metrics, suffix="/Val")
                }
            )

        # Step: save metrics
        save_metrics_to_csv(metrics, self._parameters.output_dir.get('metrics_file'))
        save_metrics_to_tb(metrics, self._parameters.output_dir.get('metrics_path'))

        # Step: save models
        if not self._parameters.save_ckpt_per_epoch:
            # Save the model of the last epoch
            filename = save_checkpoints(current_model, checkpoints_path)
            logging.getLogger('workflow_logger').info(f"Saved model of the last epoch at {filename}")

            # save best model
            filename = save_best_model(best_model, checkpoints_path)
            logging.getLogger('workflow_logger').info(f"Saved best model at {filename}")
        else:
            logging.getLogger('workflow_logger').info(f"Saved models per epoch in directory {checkpoints_path}")

            # Mark best model
            filename = mark_best_model(best_model, checkpoints_path)
            logging.getLogger('workflow_logger').info(f"Saved model with the best MCC at {filename}")

    def getting_applicability_domain(self, ad_models: List[Dict], features: pd.DataFrame):
        pass

class PredictionWorkflow(GDLWorkflow, ABC):
    def set_output_filename(self, substr: str):
        path = Path(self._parameters.output_dir['prediction_path'])
        self._parameters.output_dir['prediction_file'] = path.joinpath(f"Prediction-batch_{substr}.csv")

        path = Path(self._parameters.output_dir['features_path'])
        self._parameters.output_dir['features_file'] = path.joinpath(f"Features-batch_{substr}.csv")

    def build_applicability_domain_model(self):
        if self._parameters.get_ad:
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

    def computing_features(self, data: pd.DataFrame, graphs: List[Data], perplexities: pd.DataFrame) -> Optional[DataFrame]:
        if self._parameters.get_ad:
            return super().computing_features(data, graphs, perplexities)

    def init_gnn_model(self, graphs: List):
        checkpoint = torch.load(self._parameters.gdl_model_path)
        model = checkpoint['model']
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self._parameters.device)
        return model

    def modeling_task(self, model_evaluator: Union[Tuple[Evaluator, Evaluator], Evaluator], graphs: List) -> EvaluationOutputDTO:
        data = DataLoader(dataset=graphs, batch_size=self._parameters.batch_size)
        return model_evaluator.eval(data)

    def get_applicability_domain(self, ad_models: List[Dict], features: pd.DataFrame):
        if self._parameters.get_ad:
            instance_id = features.iloc[:, 0]
            features_to_eval = features.iloc[:, 1:]

            domain = pd.DataFrame()
            with tqdm(total=len(ad_models),
                      desc="Getting applicability domain", disable=False) as progress:
                for ad_model in ad_models:
                    method_for_ad = ad_model['method']
                    model_for_ad = ad_model['model']

                    features_to_eval_selected = filter_features(features_to_eval, method_for_ad['features'])
                    outlier, outlier_score = model_for_ad.eval_model(features_to_eval_selected)

                    temp_domain = pd.DataFrame({
                        f"{method_for_ad['method_id']}": ['out' if x == -1 else 'in' for x in outlier],
                        f"{method_for_ad['method_id']}_score": outlier_score
                    })

                    domain = pd.concat([domain, temp_domain], axis=1)
                    progress.update(1)
            domain['sequence'] = instance_id

            csv_path = self._parameters.output_dir['prediction_file']
            prediction = pd.read_csv(csv_path)
            merged_df = pd.merge(prediction, domain, on='sequence', how='inner')
            merged_df.to_csv(csv_path, index=False)


class TestWorkflow(PredictionWorkflow):
    def init_model_evaluator(self, model: nn.Module):
        return TestModeEvaluator(
            model=model,
            device=self._parameters.device,
            prediction=self._context.prediction_processor,
        )

    def validate_dataset(self, data: pd.DataFrame) -> pd.DataFrame:
        data = super().validate_dataset(data)
        return data[data['partition'].isin([3])].reset_index(drop=True)

    def save_prediction(self, eval_output: EvaluationOutputDTO):
        df = pd.DataFrame(
            {
                **eval_output.sequence_info,
                'y_true': eval_output.y_true,
                **eval_output.prediction.dict()
            }
        )
        df.to_csv(self._parameters.output_dir['prediction_file'], index=False)

    def compute_metrics(self):
        # Get the parent directory of the prediction file path
        parent_dir = self._parameters.output_dir['prediction_file'].parent

        # Find all prediction files that match the "Prediction*.csv" pattern
        prediction_files = parent_dir.glob("Prediction*.csv")

        prediction = PredictionDTO()
        y_true = []

        for file in prediction_files:
            pred = pd.read_csv(file).to_dict(orient="list")

            prediction.extend(PredictionDTO(**pred))
            y_true.extend(pred.get("y_true"))

        # Compute test metrics
        test_metrics = format_metric_keys(
            metrics=self._context.metrics.calculate(
                prediction=prediction,
                y_true=y_true
            ),
            suffix="/Test"
        )

        # Step: save metrics
        save_metrics_to_csv([test_metrics], self._parameters.output_dir.get('metrics_file'))


class InferenceWorkflow(PredictionWorkflow):
    def init_model_evaluator(self, model: nn.Module):
        return InferenceModeEvaluator(
            model=model,
            device=self._parameters.device,
            prediction=self._context.prediction_processor,
        )

    def save_prediction(self, eval_output: EvaluationOutputDTO):
        df = pd.DataFrame(
            {
                **eval_output.sequence_info,
                **eval_output.prediction.dict()
            }
        )
        df.to_csv(self._parameters.output_dir['prediction_file'], index=False)


# auxiliar methods
def save_to_fasta(df: pd.DataFrame, fasta_file):
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


def save_metrics_to_csv(metrics: List[dict], csv_file: Path):
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(csv_file, index=False)


def save_metrics_to_tb(metrics: List[dict], metrics_path: Path):
    tb_writer = SummaryWriter(log_dir=metrics_path, filename_suffix="_metrics")

    # write tensorboard metrics, excluding 'epoch'
    for entry in metrics:
        epoch = entry.get("epoch")
        for key, value in entry.items():
            if key != "epoch":
                tb_writer.add_scalar(key, value, global_step=epoch)


def save_checkpoints(model: ModelDTO, checkpoints_path: Path):
    filename = get_checkpoint_name(model, checkpoints_path)
    torch.save(model.dict(), filename)
    return filename


def save_best_model(model: ModelDTO, checkpoints_path: Path):
    filename = get_checkpoint_name(model, checkpoints_path)
    filename = filename.with_name(filename.stem + "_(best).pt")
    torch.save(model.dict(), filename)
    return filename


def get_checkpoint_name(model: ModelDTO, checkpoints_path: Path):
    return checkpoints_path.joinpath(
        f"epoch={model.epoch}_train-loss={model.train_loss:.2f}_val-loss={model.val_loss:.2f}.pt"
    )


def mark_best_model(best_model: ModelDTO, checkpoints_path: Path) -> Path:
    original_name = f"epoch={best_model.epoch}_train-loss={best_model.train_loss:.2f}_val-loss={best_model.val_loss:.2f}.pt"
    original_path = checkpoints_path.joinpath(original_name)

    new_path = original_path.with_name(original_path.stem + "_(best).pt")

    if original_path.exists():
        original_path.replace(new_path)
        return new_path
    else:
        raise FileNotFoundError(f"Checkpoint not found: {original_path}")
