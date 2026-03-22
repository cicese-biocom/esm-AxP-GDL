from pydantic.v1 import root_validator

from src.params.prediction import PredictionArguments
from src.config.types import ExecutionMode


class InferenceArguments(PredictionArguments):
    @root_validator(skip_on_failure=True)
    def validate_and_configure(cls, values):
        _set_execution_mode(values)

        return values


def _set_execution_mode(values):
    values['execution_mode'] = ExecutionMode.TEST
