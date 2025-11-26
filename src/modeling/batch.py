from typing import Optional, List, Dict

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch_geometric.data import Data

from src.modeling.prediction import PredictionProcessor, Prediction
from src.utils.base_dto import BaseDataTransferObject


class ProcessedBatch(BaseDataTransferObject):
    loss: Optional[float] = None
    y_true: Optional[List] = None
    sequence_info: Optional[Dict] = None
    prediction: Optional[Prediction] = None


class BatchProcessor:
    def __init__(self, model: nn.Module, device: torch.device):
        self._model = model
        self._device = device
        
    @property
    def model(self) -> nn.Module:
        return self._model
        
    def process(self, batch: Data) -> ProcessedBatch:
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

    def process(self, batch: Data) -> ProcessedBatch:
        self._optimizer.zero_grad()

        output = super().process(batch)

        loss = self._loss_fn(output, batch.y)
        loss.backward()
        self._optimizer.step()

        return ProcessedBatch(
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

    def process(self, batch: Data) -> ProcessedBatch:
        output = super().process(batch)

        return ProcessedBatch(
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

    def process(self, batch: Data) -> ProcessedBatch:
        output = super().process(batch)

        return ProcessedBatch(
            y_true=batch.y.tolist(),
            sequence_info=batch.sequence_info,
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

    def process(self, batch: Data) -> ProcessedBatch:
        output = super().process(batch)

        return ProcessedBatch(
            sequence_info=batch.sequence_info,
            prediction=self._prediction.process(output)
        )
