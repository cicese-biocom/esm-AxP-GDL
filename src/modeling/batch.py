from typing import Optional, List, Dict

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch_geometric.data import Data

from src.modeling.prediction import PredictionProcessor, PredictionDTO
from src.utils.dto import DTO


class ProcessedBatchDTO(DTO):
    loss: Optional[float] = None
    y_true: Optional[List] = None
    prediction: Optional[PredictionDTO] = None


class BatchProcessor:
    def __init__(self, model: nn.Module, device: torch.device):
        self._model = model
        self._device = device
        
    @property
    def model(self) -> nn.Module:
        return self._model
        
    def process(self, batch: Data) -> ProcessedBatchDTO:
        batch = batch.to(device=self._device)

        output = self._model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch
        )

        return output[0]


class TrainingBatchProcessor(BatchProcessor):
    def __init__(
            self,
            model: nn.Module,
            device: torch.device,
            loss_fn: _Loss,
            optimizer: torch.optim.Optimizer,
    ):
        super(TrainingBatchProcessor, self).__init__(model, device)
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        
    @property
    def optimizer(self):
        return self._optimizer

    def process(self, batch: Data) -> ProcessedBatchDTO:
        self._optimizer.zero_grad()

        output = super().process(batch)

        loss = self._loss_fn(output, batch.y)
        loss.backward()
        self._optimizer.step()

        return ProcessedBatchDTO(
            loss=loss.item()
        )


class ValidationBatchProcessor(BatchProcessor):
    def __init__(
            self,
            model: nn.Module,
            device: torch.device,
            loss_fn: _Loss,
            prediction: PredictionProcessor
    ):
        super(ValidationBatchProcessor, self).__init__(model, device)
        self._loss_fn = loss_fn
        self._prediction = prediction

    def process(self, batch: Data) -> ProcessedBatchDTO:
        output = super().process(batch)

        return ProcessedBatchDTO(
            loss=self._loss_fn(output, batch.y).item(),
            y_true=batch.y.tolist(),
            prediction=self._prediction.process(output)
        )

class TestBatchProcessor(BatchProcessor):
    def __init__(
            self,
            model: nn.Module,
            device: torch.device,
            prediction: PredictionProcessor
    ):
        super(TestBatchProcessor, self).__init__(model, device)
        self._prediction = prediction

    def process(self, batch: Data) -> ProcessedBatchDTO:
        output = super().process(batch)

        return ProcessedBatchDTO(
            y_true=batch.y.tolist(),
            prediction=self._prediction.process(output)
        )

class InferenceBatchProcessor(BatchProcessor):
    def __init__(
            self,
            model: nn.Module,
            device: torch.device,
            prediction: PredictionProcessor
    ):
        super(InferenceBatchProcessor, self).__init__(model, device)
        self._prediction = prediction

    def process(self, batch: Data) -> ProcessedBatchDTO:
        output = super().process(batch)

        return ProcessedBatchDTO(
            prediction=self._prediction.process(output)
        )
