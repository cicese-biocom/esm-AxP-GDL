from typing import Optional, List, Dict

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch_geometric.data import Data

from src.modeling.prediction import PredictionMaking, Prediction
from src.utils.base_parameters import BaseParameters


class ProcessOutput(BaseParameters):
    loss: Optional[float] = None
    y_true: Optional[List] = None
    sequence_info: Optional[Dict] = None
    prediction: Optional[Prediction] = None


class OutputProcessor:
    def __init__(self, model: nn.Module, device: torch.device):
        self._model = model
        self._device = device
        
    @property
    def model(self) -> nn.Module:
        return self._model
        
    def calculate(self, batch: Data) -> ProcessOutput:
        batch = batch.to(device=self._device)

        output = self._model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch
        )

        return output[0]


class TrainingOutputProcessor(OutputProcessor):
    def __init__(
            self,
            model: nn.Module,
            device: torch.device,
            loss_fn: _Loss,
            optimizer: torch.optim.Optimizer,
    ):
        super(TrainingOutputProcessor, self).__init__(model, device)
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        
    @property
    def optimizer(self):
        return self._optimizer

    def calculate(self, batch: Data) -> ProcessOutput:
        self._optimizer.zero_grad()

        output = super().calculate(batch)

        loss = self._loss_fn(output, batch.y)
        loss.backward()
        self._optimizer.step()

        return ProcessOutput(
            loss=loss.item()
        )


class ValidationOutputProcessor(OutputProcessor):
    def __init__(
            self,
            model: nn.Module,
            device: torch.device,
            loss_fn: _Loss,
            prediction: PredictionMaking
    ):
        super(ValidationOutputProcessor, self).__init__(model, device)
        self._loss_fn = loss_fn
        self._prediction = prediction

    def calculate(self, batch: Data) -> ProcessOutput:
        output = super().calculate(batch)

        return ProcessOutput(
            loss=self._loss_fn(output, batch.y).item(),
            y_true=batch.y.tolist(),
            prediction=self._prediction.prediction_making(output)
        )

class TestOutputProcessor(OutputProcessor):
    def __init__(
            self,
            model: nn.Module,
            device: torch.device,
            prediction: PredictionMaking
    ):
        super(TestOutputProcessor, self).__init__(model, device)
        self._prediction = prediction

    def calculate(self, batch: Data) -> ProcessOutput:
        output = super().calculate(batch)

        return ProcessOutput(
            y_true=batch.y.tolist(),
            sequence_info=batch.sequence_info,
            prediction=self._prediction.prediction_making(output)
        )

class InferenceOutputProcessor(OutputProcessor):
    def __init__(
            self,
            model: nn.Module,
            device: torch.device,
            prediction: PredictionMaking
    ):
        super(InferenceOutputProcessor, self).__init__(model, device)
        self._prediction = prediction

    def calculate(self, batch: Data) -> ProcessOutput:
        output = super().calculate(batch)

        return ProcessOutput(
            sequence_info=batch.sequence_info,
            prediction=self._prediction.prediction_making(output)
        )
