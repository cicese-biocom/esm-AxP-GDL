import logging
from abc import ABC, abstractmethod
from pathlib import Path
from workflow.classification_metrics import ClassificationMetricsContext
from workflow.data_loader import DataLoaderContext
from workflow.dataset_validator import DatasetValidatorContext
from workflow.parameters_setter import ParameterSetter
from graph.construct_graphs import construct_graphs
import pandas as pd
from typing import List, Dict
from sklearn.model_selection import train_test_split
from models.GAT.GAT import GATModel
import torch
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
import random
from utils import json_parser as json_parser
from .application_context import ApplicationContext
from .dto_workflow import ModelParameters, TrainingOutputParameter, EvalOutputParameter
from .logging_handler import LoggingHandler
from .path_creator import PathCreatorContext


class GDLWorkflow(ABC):
    def run_workflow(self, context: ApplicationContext, parameters: Dict) -> None:

        # Initialization of workflow parameters
        output_setting = self.create_path(path_creator_context=context.path_creator, parameters=parameters)

        LoggingHandler.initialize_logger(logger_settings_path=Path('settings').joinpath('logger_setting.json'), log_output_path=output_setting['log_file'])

        workflow_settings = self.parameters_setter(output_setting=output_setting, parameters=parameters)

        # Workflow execution
        data = self.load_data(workflow_settings, context.data_loader, context.dataset_validator)

        graphs, data = construct_graphs(workflow_settings=workflow_settings, data=data)

        graphs = self.getting_graphs_by_partition(graphs=graphs, data=data)

        model = self.initialize_model(workflow_settings=workflow_settings, graphs=graphs)

        self.execute(workflow_settings=workflow_settings, graphs=graphs, model=model, classification_metrics=context.classification_metrics, data=data)

        self.save_parameters(workflow_settings=workflow_settings)

    def create_path(self, path_creator_context: PathCreatorContext, parameters: Dict):
        return path_creator_context.create_path(parameters['output_path'])

    @abstractmethod
    def parameters_setter(self, output_setting: Dict, parameters: Dict):
        pass

    def load_data(self, workflow_settings: ParameterSetter, data_loader: DataLoaderContext,
                  dataset_validator: DatasetValidatorContext) -> pd.DataFrame:
        data = data_loader.read_file(filepath=workflow_settings.dataset)
        data = dataset_validator.processing_dataset(dataset=data,
                                                    output_setting=workflow_settings.output_setting)
        return data

    def getting_graphs_by_partition(self, graphs: List, data: pd.DataFrame) -> List:
        return graphs

    def initialize_model(self, workflow_settings: ParameterSetter, graphs: List):
        logging.getLogger('workflow_logger').info(f"The parameter add_self_loops has been set to {workflow_settings.add_self_loops}")
        return None

    @abstractmethod
    def execute(self, workflow_settings: ParameterSetter, graphs: List, model: GATModel,
                classification_metrics: ClassificationMetricsContext, data: pd.DataFrame) -> Dict:
        pass

    def save_parameters(self, workflow_settings: ParameterSetter):
        json_file = workflow_settings.output_setting['parameter_file']
        parameters = EvalOutputParameter(**workflow_settings.model_dump())
        json_data = parameters.model_dump()
        json_parser.save_json(json_file, json_data)


class TrainingWorkflow(GDLWorkflow):
    def create_path(self, path_creator_context: PathCreatorContext, parameters: Dict):
        return path_creator_context.create_path(parameters['gdl_model_path'])

    def parameters_setter(self, output_setting: Dict, parameters: Dict):
        return ParameterSetter(mode='training', output_setting=output_setting, **parameters)

    def load_data(self, workflow_settings: ParameterSetter, data_loader: DataLoaderContext,
                  dataset_validator: DatasetValidatorContext) -> pd.DataFrame:
        try:
            data = super().load_data(workflow_settings, data_loader, dataset_validator)

            # keeping only the instances belonging to the training and validation sets
            data = data[data['partition'].isin([1, 2])].reset_index(drop=True)
            if data.size == 0:
                raise Exception("The input set does not contain training instances nor validation instances.")

            # there are no training or validation instances
            # therefore, the 'partition' column is removed
            if data.query('partition == 1').size == 0 or data.query('partition == 2').size == 0:
                data.drop(['partition'], axis=1, inplace=True)

            # the 'partition' column was removed above
            # thus, the data are randomly split (80% training, 20%validation),
            #       and the 'partition' column is added with the values 1 or 2 as appropriate
            if 'partition' not in data.columns:
                # split training dataset: 80% train y 20% test, with seed and shuffle
                train_indexes, val_indexes, _, _ = train_test_split(data.index, data['activity'], test_size=0.2, shuffle=True, random_state=41)

                training = data.drop(val_indexes).assign(partition=lambda x: 1)
                validation = data.drop(train_indexes).assign(partition=lambda x: 2)

                data = pd.concat([training, validation])

                data.reset_index(drop=True)

                csv_file = workflow_settings.output_setting['partitioned_data']
                data.to_csv(csv_file, index=False)

                logging.getLogger('workflow_logger'). \
                    warning(f"Split training dataset: 80% train y 20% test, with seed and shuffle. See: {csv_file}")

            return data
        except Exception as e:
            logging.getLogger('workflow_logger').exception(e)
            quit()

    def getting_graphs_by_partition(self, graphs: List, data: pd.DataFrame) -> List:
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
        json_file = workflow_settings.output_setting['parameter_file']
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
                    Validation_Recall_Pos=f"{recall_pos:.4f}" if recall_pos else None,
                    Validation_Recall_Neg=f"{recall_neg:.4f}" if recall_neg else None
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
        columns = ['Epoch', 'Loss/Train', 'Loss/Val', 'MCC/Val', 'ACC/Val', 'AUC/Val', 'Recall_Pos/Val', 'Recall_Neg/Val']
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
            Validation_Recall_Pos=f"{recall_pos_of_the_best_mcc:.4f}" if recall_pos_of_the_best_mcc else None,
            Validation_Recall_Neg=f"{recall_neg_of_the_best_mcc:.4f}" if recall_neg_of_the_best_mcc else None
        )
        bar.close()


class PredictionWorkflow(GDLWorkflow, ABC):
    def initialize_model(self, workflow_settings: ParameterSetter, graphs: List):
        super().initialize_model(workflow_settings, graphs)

        checkpoint = torch.load(workflow_settings.gdl_model_path)
        model = checkpoint['model']
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(workflow_settings.device)
        return model


class TestWorkflow(PredictionWorkflow):
    def parameters_setter(self, output_setting: Dict, parameters: Dict):
        checkpoint = torch.load(parameters['gdl_model_path'])
        trained_model_parameters = checkpoint['parameters']
        merged_parameters = {**trained_model_parameters, **parameters}
        workflow_settings = ParameterSetter(mode='test', output_setting=output_setting, **merged_parameters)
        return workflow_settings

    def load_data(self, workflow_settings: ParameterSetter, data_loader: DataLoaderContext,
                  dataset_validator: DatasetValidatorContext) -> pd.DataFrame:
        data = super().load_data(workflow_settings, data_loader, dataset_validator)
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
                    Recall_Pos=f"{recall_pos:.4f}" if recall_pos else None,
                    Recall_Neg=f"{recall_neg:.4f}" if recall_neg else None,
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


class InferenceWorkflow(PredictionWorkflow):
    def parameters_setter(self, output_setting: Dict, parameters: Dict):
        checkpoint = torch.load(parameters['gdl_model_path'])
        trained_model_parameters = checkpoint['parameters']
        merged_parameters = {**trained_model_parameters, **parameters}
        return ParameterSetter(mode='inference', output_setting=output_setting, **merged_parameters)

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
