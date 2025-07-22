from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch.nn.modules.loss import _Loss

from src.modeling.batch import TrainingBatchProcessor, ValidationBatchProcessor, TestBatchProcessor, \
    InferenceBatchProcessor
from src.modeling.evaluator import TrainingModeEvaluator, ValidationModeEvaluator, TestModeEvaluator, \
    InferenceModeEvaluator
from src.modeling.metrics import Metrics
from src.modeling.prediction import PredictionProcessor

from src.utils.dto import DTO


class  ModellingFactoryDTO(DTO):
    device: torch.device
    loss_fn: Optional[_Loss]
    optimizer: Optional[torch.optim.Optimizer]
    metrics_calculator: Optional[Metrics]
    prediction_processor: Optional[PredictionProcessor]



class ModellingFactory(ABC):
    @abstractmethod
    def create_evaluator(self, modelling_factory_dto: ModellingFactoryDTO):
        pass


class TrainingModellingFactory(ModellingFactory):
    def create_evaluator(self, modelling_factory_dto: ModellingFactoryDTO):
        return TrainingModeEvaluator(
            batch_processor=TrainingBatchProcessor(
                device=modelling_factory_dto.device,
                loss_fn=modelling_factory_dto.loss_fn,
                optimizer=modelling_factory_dto.optimizer
            )
        )


class ValidationModellingFactory(ModellingFactory):
    def create_evaluator(self, modelling_factory_dto: ModellingFactoryDTO):
        return ValidationModeEvaluator(
            batch_processor=ValidationBatchProcessor(
                device=modelling_factory_dto.device,
                loss_fn=modelling_factory_dto.loss_fn,
                prediction=modelling_factory_dto.prediction_processor
            ),
            metrics=modelling_factory_dto.metrics_calculator
        )


class TestModellingFactory(ModellingFactory):
    def create_evaluator(self, modelling_factory_dto: ModellingFactoryDTO):
        return TestModeEvaluator(
            batch_processor=TestBatchProcessor(
                device=modelling_factory_dto.device,
                prediction=modelling_factory_dto.prediction_processor
            ),
            metrics=modelling_factory_dto.metrics_calculator
        )


class InferenceModellingFactory(ModellingFactory):
    def create_evaluator(self, modelling_factory_dto: ModellingFactoryDTO):
        return InferenceModeEvaluator(
            batch_processor=InferenceBatchProcessor(
                device=modelling_factory_dto.device,
                prediction=modelling_factory_dto.prediction_processor
            )
        )
