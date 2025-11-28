from abc import abstractmethod, ABC
from typing import List, Optional
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    matthews_corrcoef,
    recall_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from src.modeling.prediction import Prediction, PredictionMaking


class Metrics(ABC):
    """
    Abstract base class for metric calculation.
    """

    def __init__(self, prediction_calculateor: PredictionMaking, classes: Optional[List[int]] = None):
        """
        Initializes the Metrics object.

        Args:
            classes (Optional[List[int]]): List of class labels to consider when calculating metrics.
                Useful for multi-class classification tasks to specify which classes to include.
        """
        self._classes = classes
        self._prediction_calculateor = prediction_calculateor

    @abstractmethod
    def calculate(self, prediction: Prediction, y_true: List) -> dict:
        """
        Abstract method to calculate metrics based on model model_output.

        Args:
            y_true:
            prediction: Prediction of model model_output

        Returns:
            dict: A dictionary with metric names as keys and their calculated values.
        """
        pass


class BinaryClassificationMetrics(Metrics):
    """
    Implements metric calculations specific to binary classification tasks.
    Calculates MCC, accuracy, AUC, specificity, and sensitivity.
    """

    def calculate(self, prediction: Prediction, y_true: List) -> dict:
        """
        Computes binary classification metrics from a list of ProcessedOutput.

        Args:
            y_true:
            prediction: Prediction of model model_output

        Returns:
            dict: Dictionary with keys:
                - "MCC": Matthews Correlation Coefficient
                - "ACC": ACCuracy
                - "AUC": Area Under ROC Curve
                - "Recall_Pos": Specificity (recall for class 0)
                - "Recall_Neg": Sensitivity (recall for class 1)
        """

        recall_per_class = recall_score(y_true, prediction.y_pred, average=None, labels=self._classes)
        recall_dict = {f"Recall_class_{c}": r for c, r in zip(self._classes, recall_per_class)}

        return {
            "MCC": matthews_corrcoef(y_true, prediction.y_pred),
            "ACC": accuracy_score(y_true, prediction.y_pred),
            "AUC": roc_auc_score(y_true, prediction.y_score),
            **recall_dict
        }

class MulticlassClassificationMetrics(Metrics):
    """
    Implements metric calculations for multi-class classification tasks.
    Calculates MCC, accuracy, AUC per class, and recall per class.
    """

    def calculate(self, prediction: Prediction, y_true: List) -> dict:
        """
        Computes multi-class classification metrics from calculateed model model_output.

        Args:
            y_true:
            prediction: Prediction of model model_output

        Returns:
            dict: Dictionary containing:
                - "MCC": Matthews Correlation Coefficient
                - "ACC": ACCuracy
                - "AUC": List or array of AUC values per class
                - Recall_class_{class_label}: Recall per each class
        """

        recall_per_class = recall_score(y_true, prediction.y_pred, average=None, labels=self._classes)
        recall_dict = {f"Recall_class_{c}": r for c, r in zip(self._classes, recall_per_class)}

        return {
            "MCC": matthews_corrcoef(y_true, prediction.y_pred),
            "ACC": accuracy_score(y_true, prediction.y_pred),
            "AUC": roc_auc_score(y_true, prediction.y_score, average=None),
            **recall_dict
        }


class RegressionMetrics(Metrics):
    """
    Implements regression metrics calculation.
    Calculates Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared (R²).
    """

    def calculate(self, prediction: Prediction, y_true: List) -> dict:
        """
        Computes regression metrics based on model output.

        Args:
            y_true:
            prediction: Prediction of model model_output

        Returns:
            dict: Dictionary with keys:
                - "RMSE": Root Mean Squared Error
                - "MAE": Mean Absolute Error
                - "R2": Coefficient of Determination (R²)
        """

        return {
            "RMSE": np.sqrt(mean_squared_error(y_true, prediction.y_pred)),
            "MAE": mean_absolute_error(y_true, prediction.y_pred),
            "R2": r2_score(y_true, prediction.y_pred)
        }