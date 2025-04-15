import logging
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from workflow.ad_methods import build_model
from workflow.features import filter_features, FeaturesContext
from graph.construct_graphs import construct_graphs
from models.GAT.GAT import GATModel
from utils import json_parser as json_parser
from .classification_metrics import ClassificationMetricsContext
from .data_partitioner import to_partition
from .dataset_validator import DatasetValidatorContext
from .parameters_setter import ParameterSetter
from .application_context import ApplicationContext
from .dto_workflow import ModelParameters, TrainingOutputParameter, EvalOutputParameter
from .logging_handler import LoggingHandler
from .path_creator import PathCreatorContext
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO


class GDLWorkflow(ABC):
    def run_workflow(self, context: ApplicationContext, parameters: Dict) -> None:
        # Initialization of workflow parameters
        output_setting = self.create_path(path_creator_context=context.path_creator, parameters=parameters)

        LoggingHandler.initialize_logger(logger_settings_path=Path('settings').joinpath('logger_setting.json'),
                                         log_output_path=output_setting['log_file'])

        workflow_settings = self.parameters_setter(output_setting=output_setting, parameters=parameters)

        # Execution Workflow Args
        self.save_parameters(workflow_settings=workflow_settings)

        ad_models = self.building_applicability_domain_model(workflow_settings)

        # Workflow execution
        data_loaded = context.data_loader.read_file(workflow_settings=workflow_settings)

        for i, data_chunck in enumerate(data_loaded):
            self.generate_filenames(workflow_settings=workflow_settings, substr=str(i + 1))

            data_chunck = self.validate_dataset(workflow_settings, data_chunck, context.dataset_validator)

            graphs, perplexities = construct_graphs(workflow_settings, data_chunck)

            features = self.computing_features(workflow_settings=workflow_settings, data=data_chunck, graphs=graphs,
                                               perplexities=perplexities)

            graphs = self.getting_graphs_by_partition(workflow_settings=workflow_settings, graphs=graphs,
                                                      data=data_chunck, features=features)

            model = self.initialize_model(workflow_settings=workflow_settings, graphs=graphs)

            self.execute(workflow_settings=workflow_settings, graphs=graphs, model=model,
                         classification_metrics=context.classification_metrics, data=data_chunck)

            self.getting_applicability_domain(workflow_settings, ad_models, features)

    @abstractmethod
    def generate_filenames(self, workflow_settings: ParameterSetter, substr: str):
        pass

    def create_path(self, path_creator_context: PathCreatorContext, parameters: Dict):
        return path_creator_context.create_path(parameters['output_path'])

    def parameters_setter(self, output_setting: Dict, parameters: Dict):
        input_workflow_args_file = output_setting['workflow_input_args_file']
        json_parser.save_json(input_workflow_args_file, parameters)

    def validate_dataset(self, workflow_settings: ParameterSetter, data: pd.DataFrame,
                         dataset_validator: DatasetValidatorContext) -> pd.DataFrame:
        data = dataset_validator.processing_dataset(dataset=data,
                                                    output_setting=workflow_settings.output_setting)
        return data

    # computing features
    def computing_features(self, workflow_settings: ParameterSetter,
                           data: pd.DataFrame, graphs: List[Data], perplexities: pd.DataFrame) -> pd.DataFrame:
        csv_path = workflow_settings.output_setting['features_file']
        kwargs = {
            'features_to_calculate': workflow_settings.feature_types_for_ad,
            'data': data,
            'graphs': graphs,
            'perplexities': perplexities}

        feature_context = FeaturesContext()
        features = feature_context.compute_features(**kwargs)

        features.to_csv(csv_path, index=False)
        return features

    def getting_graphs_by_partition(self, workflow_settings: ParameterSetter, graphs: List, data: pd.DataFrame, features: pd.DataFrame) -> List:
        return graphs

    def initialize_model(self, workflow_settings: ParameterSetter, graphs: List):
        return None

    @abstractmethod
    def execute(self, workflow_settings: ParameterSetter, graphs: List, model: GATModel,
                classification_metrics: ClassificationMetricsContext, data: pd.DataFrame) -> Dict:
        pass

    @abstractmethod
    def getting_applicability_domain(self, workflow_settings: ParameterSetter, ad_models: List[Dict],
                                     features: pd.DataFrame):
        pass

    @abstractmethod
    def building_applicability_domain_model(self, workflow_settings: ParameterSetter):
        pass

    def save_parameters(self, workflow_settings: ParameterSetter):
        json_file = workflow_settings.output_setting['workflow_execution_args_file']
        parameters = EvalOutputParameter(**workflow_settings.model_dump())
        json_data = parameters.model_dump()
        json_parser.save_json(json_file, json_data)


class TrainingWorkflow(GDLWorkflow):
    def create_path(self, path_creator_context: PathCreatorContext, parameters: Dict):
        return path_creator_context.create_path(parameters['gdl_model_path'])

    def parameters_setter(self, output_setting: Dict, parameters: Dict):
        super().parameters_setter(output_setting, parameters)
        return ParameterSetter(mode='training', output_setting=output_setting, **parameters)

    def validate_dataset(self, workflow_settings: ParameterSetter, data: pd.DataFrame,
                         dataset_validator: DatasetValidatorContext) -> pd.DataFrame:
        data = super().validate_dataset(workflow_settings, data, dataset_validator)

        # keeping only the instances belonging to the training and validation sets
        data = data[data['partition'].isin([1, 2])].reset_index(drop=True)

        if data.query('partition == 1').empty:
            logging.getLogger('workflow_logger').critical(
                "The dataset does not contain training set."
            )
            quit()

        if data.query('partition == 2').empty:
            if workflow_settings.split_method and workflow_settings.split_training_fraction:
                data.drop(['partition'], axis=1, inplace=True)
            elif not workflow_settings.split_method:
                logging.getLogger('workflow_logger').critical(
                    "The dataset does not contain validation set; the parameter 'split_method' must be specified."
                )
                quit()
            elif not workflow_settings.split_training_fraction:
                logging.getLogger('workflow_logger').critical(
                    "The dataset does not contain validation set; the parameter 'split_training_fraction' must be specified."
                )
                quit()
        # if not (data.query('partition == 1').empty and data.query('partition == 2').empty)
        elif workflow_settings.split_method or workflow_settings.split_training_fraction:
            if workflow_settings.split_method:
                logging.getLogger('workflow_logger').critical(
                    "The dataset contains training and validation sets; the parameter split_method must not be specified."
                )
                quit()
            elif workflow_settings.split_training_fraction:
                logging.getLogger('workflow_logger').critical(
                    "The dataset contains training and validation sets; the parameter split_training_fraction must not be specified."
                )
                quit()

        return data

    def getting_graphs_by_partition(self, workflow_settings: ParameterSetter, graphs: List, data: pd.DataFrame, features: pd.DataFrame) -> List:
        if 'partition' not in data.columns:
            # partitioning data in training and validation
            data = to_partition(
                split_method=workflow_settings.split_method,
                data=data,
                features=features,
                split_training_fraction=workflow_settings.split_training_fraction
            )

            # Save to csv
            csv_file = workflow_settings.output_setting['data_csv']
            filtered_data = data[['id', 'sequence', 'activity', 'partition']]
            filtered_data.to_csv(csv_file, index=False)
            logging.getLogger('workflow_logger'). \
                info(f" Partitioned dataset saved to CSV file. See: {csv_file}")

            # Save to fasta
            # training data
            train_data = filtered_data[filtered_data['partition'] == 1]
            fasta_file = workflow_settings.output_setting['training_data_fasta']
            save_to_fasta(train_data, fasta_file)
            logging.getLogger('workflow_logger'). \
                info(f"Training sequences saved to FASTA file. See: {fasta_file}")

            # validation data
            val_data = filtered_data[filtered_data['partition'] == 2]
            fasta_file = workflow_settings.output_setting['validation_data_fasta']
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

    def save_parameters(self, workflow_settings: ParameterSetter):
        json_file = workflow_settings.output_setting['workflow_execution_args_file']
        parameters = TrainingOutputParameter(**workflow_settings.model_dump())
        json_data = parameters.model_dump()
        json_parser.save_json(json_file, json_data)

    def initialize_model(self, workflow_settings: ParameterSetter, graphs: List):
        super().initialize_model(workflow_settings, graphs)

        train_graphs, val_graphs = graphs
        node_feature_dimension = train_graphs[0].x.shape[1]

        model = GATModel(node_feature_dimension, workflow_settings.hidden_layer_dimension,
                         workflow_settings.numbers_of_class, workflow_settings.dropout_rate,
                         workflow_settings.number_of_heads, workflow_settings.pooling_ratio,
                         workflow_settings.add_self_loops).to(workflow_settings.device)
        return model

    def execute(self, workflow_settings: ParameterSetter, graphs: List, model: GATModel,
                classification_metrics: ClassificationMetricsContext, data: pd.DataFrame) -> Dict:
        train_graphs, val_graphs = graphs

        writer = SummaryWriter(log_dir=workflow_settings.output_setting['metrics_path'], filename_suffix="_metrics")

        optimizer = torch.optim.Adam(model.parameters(), workflow_settings.learning_rate, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
        criterion = torch.nn.CrossEntropyLoss()

        train_dataloader = DataLoader(dataset=train_graphs, batch_size=workflow_settings.batch_size)
        val_dataloader = DataLoader(dataset=val_graphs, batch_size=workflow_settings.batch_size)

        # Training and Validation
        best_mcc = -2
        current_model = {}
        model_with_best_mcc = {}
        bar = tqdm(total=workflow_settings.number_of_epochs, desc="Training and Validation:")
        metrics_data = []
        parameters = ModelParameters(**workflow_settings.model_dump())

        for epoch in range(1, workflow_settings.number_of_epochs + 1):
            model.train()
            arr_loss = []
            for data in train_dataloader:
                optimizer.zero_grad()
                data = data.to(workflow_settings.device)

                if workflow_settings.use_edge_attr:
                    output = model(data.x, data.edge_index, data.edge_attr, data.batch)
                else:
                    output = model(data.x, data.edge_index, None, data.batch)

                out = output[0]
                loss = criterion(out, data.y)
                loss.backward()
                optimizer.step()
                arr_loss.append(loss.item())

            current_model['epoch'] = epoch
            current_model['model'] = model
            current_model['model_state_dict'] = model.state_dict()
            current_model['optimizer_state_dict'] = optimizer.state_dict()
            current_model['scheduler_state_dict'] = scheduler.state_dict()
            current_model['parameters'] = parameters.model_dump()

            model.eval()
            with torch.no_grad():
                total_num = 0
                total_correct = 0
                y_score = []
                y_true = []
                y_pred = []
                arr_loss = []
                for data in val_dataloader:
                    data = data.to(workflow_settings.device)

                    if workflow_settings.use_edge_attr:
                        output = model(data.x, data.edge_index, data.edge_attr, data.batch)
                    else:
                        output = model(data.x, data.edge_index, None, data.batch)

                    out = output[0]

                    loss = criterion(out, data.y)
                    arr_loss.append(loss.item())

                    pred = out.argmax(dim=1)
                    score = F.softmax(out, dim=1)[:, 1]
                    correct = (pred == data.y).sum().float()
                    total_correct += correct
                    total_num += data.num_graphs
                    y_pred.extend(pred.cpu().detach().data.numpy())
                    y_score.extend(score.cpu().detach().data.numpy())
                    y_true.extend(data.y.cpu().detach().data.numpy())

                train_loss = np.mean(arr_loss)
                val_loss = np.mean(arr_loss)

                if workflow_settings.save_ckpt_per_epoch:
                    model_path = workflow_settings.output_setting['checkpoints_path'].joinpath(
                        f"epoch={epoch}_train-loss={train_loss:.2f}_val-loss={val_loss:.2f}.pt"
                    )
                    torch.save(current_model, model_path)

                mcc = classification_metrics.matthews_correlation_coefficient(y_true=y_true, y_pred=y_pred)
                acc = classification_metrics.accuracy(y_true=y_true, y_pred=y_pred)

                if len(set(y_true)) > 1:
                    auc = classification_metrics.roc_auc(y_true=y_true, y_score=y_score)
                else:
                    auc = None

                recall_pos = classification_metrics.sensitivity(y_true=y_true, y_pred=y_pred)
                recall_neg = classification_metrics.specificity(y_true=y_true, y_pred=y_pred)

                bar.update(1)
                bar.set_postfix(
                    Epoch=f"{epoch}",
                    Training_Loss=f"{train_loss:.4f}",
                    Validation_Loss=f"{val_loss:.4f}",
                    Validation_MCC=f"{mcc:.4f}",
                    Validation_ACC=f"{acc:.4f}",
                    Validation_AUC=f"{auc:.4f}" if auc else None,
                    Validation_Recall_Pos=f"{recall_pos:.4f}" if recall_pos and not np.isnan(recall_pos) else None,
                    Validation_Recall_Neg=f"{recall_neg:.4f}" if recall_neg and not np.isnan(recall_neg) else None
                )

                writer.add_scalar('Loss/train', train_loss, global_step=epoch)
                writer.add_scalar('Loss/validation', val_loss, global_step=epoch)
                writer.add_scalar('MCC/validation', mcc, global_step=epoch)
                writer.add_scalar('ACC/validation', acc, global_step=epoch)
                if auc:
                    writer.add_scalar('AUC/validation', auc, global_step=epoch)
                writer.add_scalar('Recall_Pos/validation', recall_pos, global_step=epoch)
                writer.add_scalar('Recall_Neg/validation', recall_neg, global_step=epoch)

                metrics_data.append({
                    'Epoch': epoch,
                    'Loss/Train': train_loss,
                    'Loss/Val': val_loss,
                    'MCC/Val': mcc,
                    'ACC/Val': acc,
                    'AUC/Val': auc,
                    'Recall_Pos/Val': recall_pos,
                    'Recall_Neg/Val': recall_neg,
                })

                if mcc > best_mcc:
                    best_mcc = mcc
                    acc_of_the_best_mcc = acc
                    auc_of_the_best_mcc = auc
                    recall_pos_of_the_best_mcc = recall_pos
                    recall_neg_of_the_best_mcc = recall_neg
                    val_loss_of_the_best_mcc = val_loss
                    train_loss_of_the_best_mcc = train_loss
                    epoch_of_the_best_mcc = epoch
                    model_with_best_mcc = current_model.copy()

            scheduler.step()

        csv_file = workflow_settings.output_setting['metrics_file']
        columns = ['Epoch', 'Loss/Train', 'Loss/Val', 'MCC/Val', 'ACC/Val', 'AUC/Val', 'Recall_Pos/Val',
                   'Recall_Neg/Val']
        metrics_df = pd.DataFrame(metrics_data, columns=columns)
        metrics_df.to_csv(csv_file, index=False)

        if not workflow_settings.save_ckpt_per_epoch:
            model_path = workflow_settings.output_setting['checkpoints_path'].joinpath(
                f"epoch={epoch}_train-loss={train_loss:.2f}_val-loss={val_loss:.2f}.pt"
            )
            torch.save(current_model, model_path)

            model_path = workflow_settings.output_setting['checkpoints_path'].joinpath(
                f"epoch={epoch_of_the_best_mcc}_train-loss={train_loss_of_the_best_mcc:.2f}_val-loss"
                f"={val_loss_of_the_best_mcc:.2f}_(best-mcc).pt"
            )
            torch.save(model_with_best_mcc, model_path)

        bar.set_postfix(
            Epoch_Best_MCC=f"{epoch_of_the_best_mcc}",
            Training_Loss=f"{train_loss_of_the_best_mcc:.4f}",
            Validation_Loss=f"{val_loss_of_the_best_mcc:.4f}",
            Validation_MCC=f"{best_mcc:.4f}",
            Validation_ACC=f"{acc_of_the_best_mcc:.4f}",
            Validation_AUC=f"{auc_of_the_best_mcc:.4f}" if auc_of_the_best_mcc else None,
            Validation_Recall_Pos=f"{recall_pos_of_the_best_mcc:.4f}" if recall_pos_of_the_best_mcc and not np.isnan(
                recall_pos_of_the_best_mcc) else None,
            Validation_Recall_Neg=f"{recall_neg_of_the_best_mcc:.4f}" if recall_neg_of_the_best_mcc and not np.isnan(
                recall_neg_of_the_best_mcc) else None
        )
        bar.close()

    def getting_applicability_domain(self, workflow_settings: ParameterSetter, ad_models: List[Dict],
                                     features: pd.DataFrame):
        pass

    def building_applicability_domain_model(self, workflow_settings: ParameterSetter):
        pass

    def generate_filenames(self, workflow_settings: ParameterSetter, substr: str):
        pass


class PredictionWorkflow(GDLWorkflow, ABC):
    def initialize_model(self, workflow_settings: ParameterSetter, graphs: List):
        super().initialize_model(workflow_settings, graphs)

        checkpoint = torch.load(workflow_settings.gdl_model_path)
        model = checkpoint['model']
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(workflow_settings.device)
        return model

    def computing_features(self, workflow_settings: ParameterSetter,
                           data: pd.DataFrame, graphs: List[Data], perplexities: pd.DataFrame) -> pd.DataFrame:
        if workflow_settings.get_ad:
            return super().computing_features(workflow_settings, data, graphs, perplexities)

    def building_applicability_domain_model(self, workflow_settings: ParameterSetter):
        if workflow_settings.get_ad:
            features_to_build_domain = pd.read_csv(workflow_settings.feature_file_for_ad)

            ad_models = []
            with tqdm(total=len(workflow_settings.methods_for_ad),
                      desc="Building applicability domain models",
                      disable=False) as progress:
                for method_for_ad in workflow_settings.methods_for_ad:
                    features_selected = filter_features(features_to_build_domain, method_for_ad['features'])
                    model_for_ad = build_model(method_for_ad['method_name'], features_selected)

                    ad_models_dict = {
                        'method': method_for_ad,
                        'model': model_for_ad,
                    }
                    ad_models.append(ad_models_dict)

                    progress.update()
            return ad_models

    def getting_applicability_domain(self, workflow_settings: ParameterSetter, ad_models: List[Dict], features: pd.DataFrame):
        if workflow_settings.get_ad:
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

            csv_path = workflow_settings.output_setting['prediction_file']
            prediction = pd.read_csv(csv_path)
            merged_df = pd.merge(prediction, domain, on='sequence', how='inner')
            merged_df.to_csv(csv_path, index=False)

    def generate_filenames(self, workflow_settings: ParameterSetter, substr: str):
        pass


class TestWorkflow(PredictionWorkflow):
    def parameters_setter(self, output_setting: Dict, parameters: Dict):
        super().parameters_setter(output_setting, parameters)
        checkpoint = torch.load(parameters['gdl_model_path'])
        trained_model_parameters = checkpoint['parameters']
        merged_parameters = {**trained_model_parameters, **parameters}
        workflow_settings = ParameterSetter(mode='test', output_setting=output_setting, **merged_parameters)

        return workflow_settings

    def validate_dataset(self, workflow_settings: ParameterSetter, data: pd.DataFrame,
                         dataset_validator: DatasetValidatorContext) -> pd.DataFrame:
        data = super().validate_dataset(workflow_settings, data, dataset_validator)
        return data[data['partition'].isin([3])].reset_index(drop=True)

    def execute(self, workflow_settings: ParameterSetter, graphs: List, model: GATModel,
                classification_metrics: ClassificationMetricsContext, data: pd.DataFrame) -> Dict:
        dataloader = DataLoader(dataset=graphs, batch_size=workflow_settings.batch_size, shuffle=False)

        seed = workflow_settings.seed
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)

        y_true = []
        y_pred = []
        y_score = []

        model.eval()

        with tqdm(total=len(graphs), desc="Testing") as progress:
            with torch.no_grad():
                for data_loader in dataloader:
                    data_loader = data_loader.to(workflow_settings.device)

                    if workflow_settings.use_edge_attr:
                        output = model(data_loader.x, data_loader.edge_index, data_loader.edge_attr, data_loader.batch)
                    else:
                        output = model(data_loader.x, data_loader.edge_index, None, data_loader.batch)

                    out = output[0]

                    pred = out.argmax(dim=1)
                    score = F.softmax(out, dim=1)[:, 1]

                    y_score.extend(score.cpu().detach().data.numpy().tolist())
                    y_true.extend(data_loader.y.cpu().detach().data.numpy().tolist())
                    y_pred.extend(pred.cpu().detach().data.numpy().tolist())

                    progress.update(data_loader.num_graphs)
                    progress.set_postfix(
                        Recall_Pos=f"{'.'}",
                        Recall_Neg=f"{'.'}",
                        Test_MCC=f"{'.'}",
                        Test_ACC=f"{'.'}",
                        Test_AUC=f"{'.'}"
                    )

                mcc = classification_metrics.matthews_correlation_coefficient(y_true=y_true, y_pred=y_pred)
                acc = classification_metrics.accuracy(y_true=y_true, y_pred=y_pred)

                if len(set(y_true)) > 1:
                    auc = classification_metrics.roc_auc(y_true=y_true, y_score=y_score)
                else:
                    auc = None

                recall_pos = classification_metrics.sensitivity(y_true=y_true, y_pred=y_pred)
                recall_neg = classification_metrics.specificity(y_true=y_true, y_pred=y_pred)

                metrics_data = {
                    'MCC/Test': [mcc],
                    'ACC/Test': [acc],
                    'AUC/Test': [auc],
                    'Recall_Pos/Test': [recall_pos],
                    'Recall_Neg/Test': [recall_neg],
                }

                csv_file = workflow_settings.output_setting['metrics_file']
                columns = ['MCC/Test', 'ACC/Test', 'AUC/Test', 'Recall_Pos/Test', 'Recall_Neg/Test']
                metrics_df = pd.DataFrame(metrics_data, columns=columns)
                metrics_df.to_csv(csv_file, index=False)

                progress.set_description("Test result")
                progress.set_postfix(
                    Recall_Pos=f"{recall_pos:.4f}" if recall_pos and not np.isnan(recall_pos) else None,
                    Recall_Neg=f"{recall_neg:.4f}" if recall_neg and not np.isnan(recall_neg) else None,
                    Test_MCC=f"{mcc:.4f}",
                    Test_ACC=f"{acc:.4f}",
                    Test_AUC=f"{auc:.4f}" if auc else None
                )

        res_data = {
            'id': data.id,
            'sequence': data.sequence,
            'sequence_length': data.length,
            'true_activity': data.activity,
            'predicted_activity': y_pred,
            'predicted_activity_score': y_score,
        }
        column_order = ['id', 'sequence', 'sequence_length', 'true_activity', 'predicted_activity',
                        'predicted_activity_score']

        csv_file = workflow_settings.output_setting['prediction_file']
        df = pd.DataFrame(res_data, columns=column_order)
        df.to_csv(csv_file, index=False)

    def generate_filenames(self, workflow_settings: ParameterSetter, substr: str):
        pass


class InferenceWorkflow(PredictionWorkflow):
    def generate_filenames(self, workflow_settings: ParameterSetter, substr: str):
        path = Path(workflow_settings.output_setting['prediction_path'])
        workflow_settings.output_setting['prediction_file'] = path.joinpath(f"Prediction-batch_{substr}.csv")

        path = Path(workflow_settings.output_setting['features_path'])
        workflow_settings.output_setting['features_file'] = path.joinpath(f"Features-batch_{substr}.csv")

    def parameters_setter(self, output_setting: Dict, parameters: Dict):
        super().parameters_setter(output_setting, parameters)
        checkpoint = torch.load(parameters['gdl_model_path'])
        trained_model_parameters = checkpoint['parameters']
        merged_parameters = {**trained_model_parameters, **parameters}
        workflow_settings = ParameterSetter(mode='inference', output_setting=output_setting, **merged_parameters)

        return workflow_settings

    def execute(self, workflow_settings: ParameterSetter, graphs: List, model: GATModel,
                classification_metrics: ClassificationMetricsContext, data: pd.DataFrame) -> Dict:
        dataloader = DataLoader(dataset=graphs, batch_size=workflow_settings.batch_size, shuffle=False)

        seed = workflow_settings.seed
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)

        y_pred = []
        y_score = []

        model.eval()

        with tqdm(total=len(graphs), desc="Inferring") as progress:
            with torch.no_grad():
                for data_loader in dataloader:
                    data_loader = data_loader.to(workflow_settings.device)

                    if workflow_settings.use_edge_attr:
                        output = model(data_loader.x, data_loader.edge_index, data_loader.edge_attr, data_loader.batch)
                    else:
                        output = model(data_loader.x, data_loader.edge_index, None, data_loader.batch)

                    out = output[0]

                    pred = out.argmax(dim=1)
                    score = F.softmax(out, dim=1)[:, 1]

                    y_score.extend(score.cpu().detach().data.numpy().tolist())
                    y_pred.extend(pred.cpu().detach().data.numpy().tolist())

                    progress.update(data_loader.num_graphs)
                    progress.set_postfix()

        res_data = {
            'id': data.id,
            'sequence': data.sequence,
            'sequence_length': data.length,
            'predicted_activity': y_pred,
            'predicted_activity_score': y_score,
        }
        column_order = ['id', 'sequence', 'sequence_length', 'predicted_activity', 'predicted_activity_score']

        csv_file = workflow_settings.output_setting['prediction_file']
        df = pd.DataFrame(res_data, columns=column_order)
        df.to_csv(csv_file, index=False)


def save_to_fasta(df: pd.DataFrame, fasta_file):
    fasta_records = []
    for i, row in df.iterrows():
        sequence_id = row['id']
        sequence = row['sequence']
        activity = int(row['activity'])
        record_id = f"{sequence_id}_{'Pos' if activity == 1 else 'Neg'}"
        record = SeqRecord(Seq(sequence), id=record_id, description="")
        fasta_records.append(record)

    with open(fasta_file, 'w') as output_handle:
        SeqIO.write(fasta_records, output_handle, 'fasta')
