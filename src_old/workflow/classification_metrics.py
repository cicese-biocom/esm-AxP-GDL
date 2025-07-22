from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from sklearn.metrics import matthews_corrcoef, confusion_matrix, accuracy_score, roc_auc_score


class ClassificationMetrics:
    @abstractmethod
    def matthews_correlation_coefficient(self, y_true, y_pred) -> float:
        pass

    @abstractmethod
    def sensitivity(self, y_true, y_pred) -> Tuple[float, float]:
        pass

    @abstractmethod
    def specificity(self, y_true, y_pred) -> Tuple[float, float]:
        pass
    
    @abstractmethod
    def roc_auc(self, y_true, y_score) -> float:
        pass    
    
    @abstractmethod
    def accuracy(self, y_true, y_pred) -> float:
        pass


class BinaryClassificationMetrics(ClassificationMetrics):
    def accuracy(self, y_true, y_pred) -> float:
        return accuracy_score(y_true, y_pred)

    def roc_auc(self, y_true, y_score) -> float:
        return roc_auc_score(y_true, y_score)

    def sensitivity(self, y_true, y_pred) -> float:
        tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1]).ravel()
        if tp + fn == 0:
            sensitivity = np.nan
        else:
            sensitivity = tp / (tp + fn)

        return sensitivity

    def specificity(self, y_true, y_pred) -> float:
        tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1]).ravel()

        if tn + fp == 0:
            specificity = np.nan
        else:
            specificity = tn / (tn + fp)

        return specificity

    def matthews_correlation_coefficient(self, y_true, y_pred) -> float:
        return matthews_corrcoef(y_true, y_pred)


class ClassificationMetricsContext:
    def __init__(self, classification_metrics: ClassificationMetrics) -> None:
        self._classification_metrics = classification_metrics

    def matthews_correlation_coefficient(self, y_true, y_pred) -> float:
        return self._classification_metrics.matthews_correlation_coefficient(y_true, y_pred)

    def sensitivity(self, y_true, y_pred) -> Tuple[float, float]:
        return self._classification_metrics.sensitivity(y_true=y_true, y_pred=y_pred)

    def specificity(self, y_true, y_pred) -> Tuple[float, float]:
        return self._classification_metrics.specificity(y_true=y_true, y_pred=y_pred)

    def roc_auc(self, y_true, y_score) -> float:
        return self._classification_metrics.roc_auc(y_true=y_true, y_score=y_score)

    def accuracy(self, y_true, y_pred) -> float:
        return self._classification_metrics.accuracy(y_true=y_true, y_pred=y_pred)
