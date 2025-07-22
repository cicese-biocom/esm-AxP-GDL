from abc import abstractmethod

import torch
from torch import nn
from torch_geometric.loader import DataLoader
from src.modeling.loss_function import LossFunctionContext
from src_old.workflow.prediction_processor import PredictionProcessorContext


class ModelEvaluator:
    def __init__(self,
                 model: nn.Module,
                 loss: LossFunctionContext,
                 model_runner: ModelRunnerContext,
                 prediction_processor: PredictionProcessorContext,
                 use_edge_attr,
                 device='cpu'
                 ):
        # model
        self.model = model

        # loss
        self.loss = loss.build()

        # device
        self.device = device

        # use_edge_attr
        self.use_edge_attr = use_edge_attr

        # model_runner
        self.model_runner = model_runner

        # prediction processor
        self.prediction_processor = prediction_processor

    @abstractmethod
    def compute_loss(self):
        pass

    @abstractmethod
    def prediction_process(self):
        pass

    def eval(self, data_loader:  DataLoader):
        self.model.eval()
        with torch.no_grad():
            y_true = []
            pred = []
            arr_loss = []
            for batch in data_loader:
                model_output = self.model_runner.forward(batch)

                # compute loss
                model_output_loss = self.loss(model_output, batch.y)
                arr_loss.append(model_output_loss.item())

                # process prediction
                pred.append(self.prediction_processor.process(model_output))
                y_true.extend(batch.y.cpu().detach().data.numpy())



class ClassificationModelEvaluator(ModelEvaluator):
    def compute_loss(self):
        CrossEntropyLoss().compute_loss()

    def prediction_process(self):
        pass


class ReggressionModelEvaluator(ModelEvaluator):
    def compute_loss(self):
        pass

    def prediction_process(self):
        pass