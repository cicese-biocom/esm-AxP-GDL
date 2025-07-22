from typing import Optional, List, Dict

import torch
from pydantic.v1 import PositiveFloat, PositiveInt

import numpy as np
from torch.nn.modules.loss import _Loss
from torch_geometric.loader import DataLoader
from torch import nn

from src.modeling.batch import BatchProcessor, ProcessedBatchDTO, TrainingBatchProcessor, ValidationBatchProcessor, \
    TestBatchProcessor, InferenceBatchProcessor
from src.modeling.metrics import Metrics
from src.modeling.prediction import PredictionDTO, PredictionProcessor
from src.utils.dto import DTO


class EvaluationOutputDTO(DTO):
    loss: Optional[PositiveFloat]
    metrics: Optional[Dict]
    y_true: Optional[List]
    prediction: Optional[PredictionDTO]
    optimizer_state_dict: Optional[Dict] = None
    scheduler_state_dict: Optional[Dict] = None
    model: nn.Module = None


class BatchesData:
    @staticmethod    
    def get_mean_loss(batches: List[ProcessedBatchDTO]) -> PositiveFloat:
        return np.mean([batch.loss for batch in batches if batch.loss is not None])

    @staticmethod
    def get_y_true(batches: List[ProcessedBatchDTO]) -> List:
        y_true = []
        for batch in batches:
            if batch.y_true is not None:
                y_true.extend(batch.y_true)
        return y_true

    @staticmethod
    def get_prediction(batches: List[ProcessedBatchDTO]) -> PredictionDTO:
        predictions = PredictionDTO(y_pred=[], y_score=[])
        for batch in batches:
            predictions.extend(batch.prediction)
        return predictions


class Evaluator:
    def __init__(self, model: nn.Module, device: torch.device,):
        self._model = model
        self._device = device,
        self._batch_processor = BatchProcessor(model, device)

    def eval(self, data_loader: DataLoader) -> List[ProcessedBatchDTO]:
        processed_batches = []
        for batch in data_loader:
            processed_batches.append(
                self._batch_processor.process(batch)
            )
        return processed_batches


class TrainingModeEvaluator(Evaluator):
    def __init__(
            self,
            model: nn.Module,
            device: torch.device,
            loss_fn: _Loss,
            learning_rate: float,
            weight_decay: float,
            step_size: PositiveInt,
            gamma: float
    ):
        super(TrainingModeEvaluator, self).__init__(model, device)

        # Step: init optimizer
        self._optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Step: init scheduler
        self._scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self._optimizer,
            step_size=step_size,
            gamma=gamma
        )

        self._batch_processor = TrainingBatchProcessor(
            model,
            device,
            loss_fn,
            self._optimizer
        )


    def eval(self, data_loader: DataLoader) -> EvaluationOutputDTO:
        self._model.train()
        processed_batches = super().eval(data_loader)

        return EvaluationOutputDTO(
            loss=BatchesData().get_mean_loss(processed_batches),
            model=self._batch_processor.model,
            optimizer_state_dict=self._optimizer.state_dict(),
            scheduler_state_dict=self._scheduler.state_dict(),
        )


class ValidationModeEvaluator(Evaluator):
    def __init__(
            self,
            model: nn.Module,
            device: torch.device,
            loss_fn: _Loss,
            prediction: PredictionProcessor,
            metrics: Metrics
    ):
        super(ValidationModeEvaluator, self).__init__(model, device)
        self._batch_processor = ValidationBatchProcessor(model, device, loss_fn, prediction)
        self._metrics = metrics


    def eval(self, data_loader: DataLoader) -> EvaluationOutputDTO:
        self._model.eval()
        processed_batches = super().eval(data_loader)

        return EvaluationOutputDTO(
            loss=BatchesData().get_mean_loss(processed_batches),
            metrics=self._metrics.calculate(
                    prediction=BatchesData().get_prediction(processed_batches),
                    y_true=BatchesData().get_y_true(processed_batches)
            ),
        )


class TestModeEvaluator(Evaluator):
    def __init__(
            self,
            model: nn.Module,
            device: torch.device,
            prediction: PredictionProcessor,
            metrics: Metrics
    ):
        super(TestModeEvaluator, self).__init__(model, device)
        self._batch_processor = TestBatchProcessor(model, device, prediction)
        self._metrics = metrics

    def eval(self, data_loader: DataLoader) -> EvaluationOutputDTO:
        self._model.eval()
        processed_batches = super().eval(data_loader)

        prediction = BatchesData().get_prediction(processed_batches)
        y_true = BatchesData().get_y_true(processed_batches)

        return EvaluationOutputDTO(
            metrics=self._metrics.calculate(
                    prediction=prediction,
                    y_true=y_true
                ),
            prediction=prediction,
            y_true=y_true
        )


class InferenceModeEvaluator(Evaluator):
    def __init__(
            self,
            model: nn.Module,
            device: torch.device,
            prediction: PredictionProcessor
    ):
        super(InferenceModeEvaluator, self).__init__(model, device)
        self._batch_processor = InferenceBatchProcessor(model, device, prediction)

    def eval(self, data_loader: DataLoader) -> EvaluationOutputDTO:
        self._model.eval()
        processed_batches = super().eval(data_loader)

        return EvaluationOutputDTO(
            prediction=BatchesData().get_prediction(processed_batches)
        )
