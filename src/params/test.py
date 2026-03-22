from pydantic.v1 import root_validator

from src.params.prediction import PredictionArguments
from src.config.types import ExecutionMode
from src.utils.path import check_file_exists


class TestArguments(PredictionArguments):
    @root_validator(skip_on_failure=True)
    def validate_and_configure_test_mode(cls, values):
        _set_execution_mode(values)
        _validate_dataset_csv(values)

        return values


def _set_execution_mode(values):
    values['execution_mode'] = ExecutionMode.TEST


def _validate_dataset_csv(values):
    """
    Validates that the dataset exists and is a CSV file.
    """
    dataset = values.get("dataset")
    dataset_path = check_file_exists(dataset)
    if dataset_path.suffix.lower() != ".csv":
        raise ValueError("Dataset must be a CSV file in test mode")
    values["dataset"] = dataset_path

    values["dataset_file_type"] = "CSV"
