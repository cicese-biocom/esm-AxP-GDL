from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, List

import torch
from torch.nn import CrossEntropyLoss, MSELoss

from src.data_processing.data_validator import ClassValidator, ClassificationClassValidator, RegressionClassValidator
from src.modeling.selector import BestModelSelector, MaximumMCCBestModelSelector, MinimumRMSEBestModelSelector
from src.modeling.metrics import Metrics, BinaryClassificationMetrics, MulticlassClassificationMetrics, RegressionMetrics
from src.modeling.prediction import PredictionProcessor, RegressionPredictionProcessor, BinaryClassificationPredictionProcessor, MulticlassClassificationPredictionProcessor


class ExecutionFactory(ABC):
    @abstractmethod
    def create_loss(self):
        pass

    @abstractmethod
    def create_metrics(self, prediction_processor: PredictionProcessor, classes: Optional[List[int]]) -> Metrics:
        pass

    @abstractmethod
    def create_best_model_selector(self) -> BestModelSelector:
        pass

    @abstractmethod
    def create_prediction_processor(self) -> PredictionProcessor:
        pass

    @abstractmethod
    def create_class_validator(self) -> ClassValidator:
        pass


class BinaryExecutionFactory(ExecutionFactory):

    def create_prediction_processor(self) -> BinaryClassificationPredictionProcessor:
        return BinaryClassificationPredictionProcessor()

    def create_loss(self) -> CrossEntropyLoss:
        return torch.nn.CrossEntropyLoss()

    def create_metrics(self, prediction_processor: PredictionProcessor, classes: Optional[List[int]])-> BinaryClassificationMetrics:
        return BinaryClassificationMetrics(prediction_processor, classes)

    def create_best_model_selector(self) -> MaximumMCCBestModelSelector:
        return MaximumMCCBestModelSelector()

    def create_class_validator(self) -> ClassificationClassValidator:
        return ClassificationClassValidator()


class MulticlassExecutionFactory(ExecutionFactory):

    def create_prediction_processor(self) -> MulticlassClassificationPredictionProcessor:
        return MulticlassClassificationPredictionProcessor()

    def create_loss(self) -> CrossEntropyLoss:
        return torch.nn.CrossEntropyLoss()

    def create_metrics(self, prediction_processor: PredictionProcessor, classes: Optional[List[int]]) -> MulticlassClassificationMetrics:
        return MulticlassClassificationMetrics(prediction_processor, classes)

    def create_best_model_selector(self) -> MaximumMCCBestModelSelector:
        return MaximumMCCBestModelSelector()

    def create_class_validator(self) -> ClassificationClassValidator:
        return ClassificationClassValidator()


class RegressionExecutionFactory(ExecutionFactory):

    def create_prediction_processor(self) -> RegressionPredictionProcessor:
        return RegressionPredictionProcessor()

    def create_loss(self) -> MSELoss:
        return torch.nn.MSELoss()

    def create_metrics(self, prediction_processor: PredictionProcessor, classes: Optional[List[int]])-> RegressionMetrics:
        return RegressionMetrics(prediction_processor)

    def create_best_model_selector(self) -> MinimumRMSEBestModelSelector:
        return MinimumRMSEBestModelSelector()

    def create_class_validator(self) -> RegressionClassValidator:
        return RegressionClassValidator()
