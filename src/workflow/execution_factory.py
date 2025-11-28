from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, List

import torch
from torch.nn import CrossEntropyLoss, MSELoss

from src.data_processing.data_processor import ClassValidator, ClassificationClassValidator, RegressionClassValidator
from src.modeling.selector import ModelSelector, MaxMCCModelSelector, MinRMSEModelSelector
from src.modeling.metrics import Metrics, BinaryClassificationMetrics, MulticlassClassificationMetrics, RegressionMetrics
from src.modeling.prediction import PredictionMaking, RegressionMaking, BinaryClassificationMaking, MulticlassClassificationMaking


class ExecutionFactory(ABC):
    @abstractmethod
    def create_loss(self):
        pass

    @abstractmethod
    def create_metrics(self, prediction_calculateor: PredictionMaking, classes: Optional[List[int]]) -> Metrics:
        pass

    @abstractmethod
    def create_best_model_selector(self) -> ModelSelector:
        pass

    @abstractmethod
    def create_prediction_calculateor(self) -> PredictionMaking:
        pass

    @abstractmethod
    def create_class_validator(self) -> ClassValidator:
        pass


class BinaryExecutionFactory(ExecutionFactory):

    def create_prediction_calculateor(self) -> BinaryClassificationMaking:
        return BinaryClassificationMaking()

    def create_loss(self) -> CrossEntropyLoss:
        return torch.nn.CrossEntropyLoss()

    def create_metrics(self, prediction_calculateor: PredictionMaking, classes: Optional[List[int]])-> BinaryClassificationMetrics:
        return BinaryClassificationMetrics(prediction_calculateor, classes)

    def create_best_model_selector(self) -> MaxMCCModelSelector:
        return MaxMCCModelSelector()

    def create_class_validator(self) -> ClassificationClassValidator:
        return ClassificationClassValidator()


class MulticlassExecutionFactory(ExecutionFactory):

    def create_prediction_calculateor(self) -> MulticlassClassificationMaking:
        return MulticlassClassificationMaking()

    def create_loss(self) -> CrossEntropyLoss:
        return torch.nn.CrossEntropyLoss()

    def create_metrics(self, prediction_calculateor: PredictionMaking, classes: Optional[List[int]]) -> MulticlassClassificationMetrics:
        return MulticlassClassificationMetrics(prediction_calculateor, classes)

    def create_best_model_selector(self) -> MaxMCCModelSelector:
        return MaxMCCModelSelector()

    def create_class_validator(self) -> ClassificationClassValidator:
        return ClassificationClassValidator()


class RegressionExecutionFactory(ExecutionFactory):

    def create_prediction_calculateor(self) -> RegressionMaking:
        return RegressionMaking()

    def create_loss(self) -> MSELoss:
        return torch.nn.MSELoss()

    def create_metrics(self, prediction_calculateor: PredictionMaking, classes: Optional[List[int]])-> RegressionMetrics:
        return RegressionMetrics(prediction_calculateor)

    def create_best_model_selector(self) -> MinRMSEModelSelector:
        return MinRMSEModelSelector()

    def create_class_validator(self) -> RegressionClassValidator:
        return RegressionClassValidator()
