from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, List

import torch
from torch.nn import CrossEntropyLoss, MSELoss

from src.config.types import ExecutionMode
from src.data_processing.target_feature_validator import TargetFeatureValidator, ClassificationTargetFeatureValidator, \
    RegressionTargetFeatureValidator, TrainingClassificationTargetFeatureValidator, \
    TestClassificationTargetFeatureValidator
from src.modeling.model_selector import ModelSelector, MaxMCCModelSelector, MinRMSEModelSelector
from src.modeling.metrics import Metrics, BinaryClassificationMetrics, MulticlassClassificationMetrics, RegressionMetrics
from src.modeling.prediction_maker import PredictionMaking, RegressionMaking, BinaryClassificationMaking, MulticlassClassificationMaking


class ExecutionFactory(ABC):
    @abstractmethod
    def create_loss_fn(self):
        pass

    @abstractmethod
    def create_metrics_calculator(self, prediction_maker: PredictionMaking, classes: Optional[List[int]]) -> Metrics:
        pass

    @abstractmethod
    def create_model_selector(self) -> ModelSelector:
        pass

    @abstractmethod
    def create_prediction_maker(self) -> PredictionMaking:
        pass

    @abstractmethod
    def create_target_feature_validator(self, execution_mode: ExecutionMode) -> TargetFeatureValidator:
        pass


class BinaryExecutionFactory(ExecutionFactory):

    def create_prediction_maker(self) -> BinaryClassificationMaking:
        return BinaryClassificationMaking()

    def create_loss_fn(self) -> CrossEntropyLoss:
        return torch.nn.CrossEntropyLoss()

    def create_metrics_calculator(self, prediction_maker: PredictionMaking, classes: Optional[List[int]])-> BinaryClassificationMetrics:
        return BinaryClassificationMetrics(prediction_maker, classes)

    def create_model_selector(self) -> MaxMCCModelSelector:
        return MaxMCCModelSelector()

    def create_target_feature_validator(self, execution_mode: ExecutionMode) -> TargetFeatureValidator:
        if execution_mode == ExecutionMode.TRAIN:
            return TrainingClassificationTargetFeatureValidator()
        elif execution_mode == ExecutionMode.TEST:
            return TestClassificationTargetFeatureValidator()
        return ClassificationTargetFeatureValidator()


class MulticlassExecutionFactory(ExecutionFactory):

    def create_prediction_maker(self) -> MulticlassClassificationMaking:
        return MulticlassClassificationMaking()

    def create_loss_fn(self) -> CrossEntropyLoss:
        return torch.nn.CrossEntropyLoss()

    def create_metrics_calculator(self, prediction_maker: PredictionMaking, classes: Optional[List[int]]) -> MulticlassClassificationMetrics:
        return MulticlassClassificationMetrics(prediction_maker, classes)

    def create_model_selector(self) -> MaxMCCModelSelector:
        return MaxMCCModelSelector()

    def create_target_feature_validator(self, execution_mode: ExecutionMode) -> ClassificationTargetFeatureValidator:
        if execution_mode == ExecutionMode.TRAIN:
            return TrainingClassificationTargetFeatureValidator()
        elif execution_mode == ExecutionMode.TEST:
            return TestClassificationTargetFeatureValidator()
        return ClassificationTargetFeatureValidator()


class RegressionExecutionFactory(ExecutionFactory):

    def create_prediction_maker(self) -> RegressionMaking:
        return RegressionMaking()

    def create_loss_fn(self) -> MSELoss:
        return torch.nn.MSELoss()

    def create_metrics_calculator(self, prediction_maker: PredictionMaking, classes: Optional[List[int]])-> RegressionMetrics:
        return RegressionMetrics(prediction_maker)

    def create_model_selector(self) -> MinRMSEModelSelector:
        return MinRMSEModelSelector()

    def create_target_feature_validator(self, execution_mode: ExecutionMode) -> RegressionTargetFeatureValidator:
        return RegressionTargetFeatureValidator()
