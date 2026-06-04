from pydantic.v1 import root_validator

from src.params.prediction import PredictionArguments
from src.config.types import ExecutionMode
from src.utils.path import check_file_exists


class InferenceArguments(PredictionArguments):
    @root_validator(skip_on_failure=True)
    def validate_and_configure_inference_mode(cls, values):
        _set_execution_mode(values)
        _validate_dataset_csv_or_fasta(values)

        return values


def _set_execution_mode(values):
    values['execution_mode'] = ExecutionMode.INFERENCE


def _validate_dataset_csv_or_fasta(values):
    """
    Validates that the dataset exists and is a CSV or FASTA file.
    """
    dataset = values.get("dataset")
    dataset_path = check_file_exists(dataset)
    if dataset_path.suffix.lower() not in [".csv", ".fasta"]:
        raise ValueError("Dataset must be a CSV or FASTA file in inference mode")
    values["dataset"] = dataset_path

    if dataset_path.suffix.lower() == ".csv":
        values["dataset_file_type"] = "CSV"
    else:
        values["dataset_file_type"] = "FASTA"
