from typing import Union

import pydantic_argparse
from dotenv import load_dotenv

from src.config.types import ExecutionMode
from src.params.inference import InferenceArguments
from src.params.prediction import PredictionArguments
from src.params.training import TrainingArguments
from src.utils.json import save_json


class ExecutionParameters:
    def __init__(self, execution_mode: ExecutionMode):
        self._model = {
            ExecutionMode.TRAIN: TrainingArguments,
            ExecutionMode.TEST: PredictionArguments,
            ExecutionMode.INFERENCE: InferenceArguments
        }[execution_mode]

    def get_parameters(self) -> Union[TrainingArguments, PredictionArguments, InferenceArguments]:
        load_dotenv(dotenv_path='.env')

        parser = pydantic_argparse.ArgumentParser(model=self._model, exit_on_error=True)
        args = parser.parse_typed_args()

        _save_all_parameters(args)
        _save_command_line_parameters(self._model, args)

        return args


def _save_all_parameters(args):
    save_json(
        json_file=args.output_dir['workflow_execution_args_file'],
        json_data=args.dict()
    )


def _save_command_line_parameters(model, args):
    user_fields = model.__fields__.keys()
    user_provided = {
        k: v for k, v in args.dict().items()
        if v is not None and k in user_fields
    }

    save_json(
        json_file=args.output_dir['command_line_build_graphs_parameters'],
        json_data=user_provided
    )