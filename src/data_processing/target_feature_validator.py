from abc import ABC, abstractmethod
from typing import List
import pandas as pd



class TargetFeatureValidator(ABC):
    @abstractmethod
    def validate(self, data: pd.DataFrame, classes: List[int] = None) -> pd.DataFrame:
        pass


class ClassificationTargetFeatureValidator(TargetFeatureValidator):
    def validate(self, data: pd.DataFrame, classes: List[int] = None) -> pd.DataFrame:
        """
        Validation rules:
        - Return rows whose 'activity' is not in `classes`.
        - Add synthetic rows for classes missing at least one instance.
        - Output preserves the exact schema of `data` plus a column '__error__'.

        No exceptions are raised. Returning a non-empty DataFrame means errors exist.
        """
        pass


class TrainingClassificationTargetFeatureValidator(ClassificationTargetFeatureValidator):
    """
    Validator for training and validation partitions.
    Partition 1 (training) is checked for missing classes + invalid labels.
    Partition 2 (validation) is checked only for invalid labels.
    """

    def validate(self, data: pd.DataFrame, classes: List[int] = None) -> pd.DataFrame:
        if classes is None:
            classes = []

        # Partition 1: training
        train_data = data[data["partition"] == 1].copy()
        train_invalid = get_rows_with_invalid_classes(train_data, classes)
        train_missing = get_missing_class_instances(train_data, classes)
        train_errors = pd.concat([train_invalid, train_missing], ignore_index=True)

        # Partition 2: validation (only invalid labels)
        val_data = data[data["partition"] == 2].copy()
        val_invalid = get_rows_with_invalid_classes(val_data, classes)

        # Combine results
        return pd.concat([train_errors, val_invalid], ignore_index=True)


class TestClassificationTargetFeatureValidator(ClassificationTargetFeatureValidator):
    """
    Validator for test partition (partition == 3).
    Only invalid labels are reported.
    """

    def validate(self, data: pd.DataFrame, classes: List[int] = None) -> pd.DataFrame:
        if classes is None:
            classes = []

        test_data = data[data["partition"] == 3].copy()
        return get_rows_with_invalid_classes(test_data, classes)


class RegressionTargetFeatureValidator(TargetFeatureValidator):
    def validate(self, data: pd.DataFrame, classes: List[int] = None) -> pd.DataFrame:
        # Return rows where 'activity' is not a float
        return data[~data['activity'].apply(lambda x: isinstance(x, float))]


class TargetFeatureValidatorContext:
    def __init__(self, validator: TargetFeatureValidator) -> None:
        self._validator = validator

    def validate(self, data: pd.DataFrame, classes: List[int] = None):
        return self._validator.validate(data, classes)

# -------------------------------------------------------
# Helper functions
# -------------------------------------------------------
def get_rows_with_invalid_classes(data: pd.DataFrame, classes: List[int]) -> pd.DataFrame:
    """Return rows with 'activity' not in classes."""
    invalid_rows = data[~data["activity"].isin(classes)].copy()
    if not invalid_rows.empty:
        invalid_rows["__error__"] = "Invalid class value"
    return invalid_rows


def get_missing_class_instances(data: pd.DataFrame, classes: List[int]) -> pd.DataFrame:
    """Return synthetic rows for classes with no instances in data."""
    missing_classes = [c for c in classes if (data["activity"] == c).sum() == 0]
    rows = []
    for c in missing_classes:
        row = {col: None for col in data.columns if col != "__error__"}
        row["activity"] = c
        row["__error__"] = f"Missing required class (no instances): {c}"
        rows.append(row)
    return pd.DataFrame(rows, columns=list(data.columns) + ["__error__"])