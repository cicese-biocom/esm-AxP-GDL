from collections import defaultdict
from typing import Optional, List, Dict

import torch
from pydantic.v1 import PositiveFloat, PositiveInt

import numpy as np
from torch.nn.modules.loss import _Loss
from torch_geometric.loader import DataLoader
from torch import nn

from src.modeling.output_processor import OutputProcessor, ProcessOutput, TrainingOutputProcessor, ValidationOutputProcessor, \
    TestOutputProcessor, InferenceOutputProcessor
from src.modeling.prediction_maker import Prediction, PredictionMaking
from src.utils.base_parameters import BaseParameters


class Output(BaseParameters):
    loss: Optional[PositiveFloat]
    y_true: Optional[List]
    prediction: Optional[Prediction]
    optimizer_state_dict: Optional[Dict] = None
    scheduler_state_dict: Optional[Dict] = None
    model: nn.Module = None
    sequence_info: Optional[Dict]


class ModeExecutor:
    def __init__(self, model: nn.Module, device: torch.device,):
        self._model = model
        self._device = device,
        self._output_processor = OutputProcessor(model, device)

    @property
    def model(self) -> nn.Module:
        return self._model

    def execute(self, data_loader: DataLoader) -> List[ProcessOutput]:
        processed_output = []
        for batch in data_loader:
            processed_output.append(
                self._output_processor.calculate(batch)
            )
        return processed_output


class ModelTrainer(ModeExecutor):
    def __init__(
            self,
            model: nn.Module,
            device: torch.device,
            loss_fn: _Loss,
            learning_rate: float,
            weight_decay: float,
            step_size: PositiveInt,
            gamma: float
    ):
        super(ModelTrainer, self).__init__(model, device)

        # Step: init optimizer
        self._optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Step: init scheduler
        self._scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self._optimizer,
            step_size=step_size,
            gamma=gamma
        )

        self._output_processor = TrainingOutputProcessor(
            model,
            device,
            loss_fn,
            self._optimizer
        )


    def execute(self, data_loader: DataLoader) -> Output:
        self._model.train()
        processed_output = super().execute(data_loader)

        return Output(
            loss=calculate_mean_loss(processed_output),
            model=self._output_processor.model,
            optimizer_state_dict=self._optimizer.state_dict(),
            scheduler_state_dict=self._scheduler.state_dict(),
        )


class ModelValidator(ModeExecutor):
    def __init__(
            self,
            model: nn.Module,
            device: torch.device,
            loss_fn: _Loss,
            prediction: PredictionMaking,
    ):
        super(ModelValidator, self).__init__(model, device)
        self._output_processor = ValidationOutputProcessor(model, device, loss_fn, prediction)


    def execute(self, data_loader: DataLoader) -> Output:
        self._model.eval()
        processed_output = super().execute(data_loader)

        return Output(
            loss=calculate_mean_loss(processed_output),
            prediction=get_predictions(processed_output),
            y_true=get_y_true(processed_output)
        )


class ModelTester(ModeExecutor):
    def __init__(
            self,
            model: nn.Module,
            device: torch.device,
            prediction: PredictionMaking,
    ):
        super(ModelTester, self).__init__(model, device)
        self._output_processor = TestOutputProcessor(model, device, prediction)

    def execute(self, data_loader: DataLoader) -> Output:
        self._model.eval()
        processed_output = super().execute(data_loader)

        return Output(
            prediction=get_predictions(processed_output),
            y_true=get_y_true(processed_output),
            sequence_info=get_sequence_info(processed_output),
        )


class ModelInference(ModeExecutor):
    def __init__(
            self,
            model: nn.Module,
            device: torch.device,
            prediction: PredictionMaking
    ):
        super(ModelInference, self).__init__(model, device)
        self._output_processor = InferenceOutputProcessor(model, device, prediction)

    def execute(self, data_loader: DataLoader) -> Output:
        self._model.eval()
        processed_output = super().execute(data_loader)

        return Output(
            prediction=get_predictions(processed_output),
            sequence_info=get_sequence_info(processed_output)
        )


def calculate_mean_loss(batches: List[ProcessOutput]) -> PositiveFloat:
    return np.mean([batch.loss for batch in batches if batch.loss is not None])

def get_y_true(batches: List[ProcessOutput]) -> List:
    y_true = []
    for batch in batches:
        if batch.y_true is not None:
            y_true.extend(batch.y_true)
    return y_true

def get_predictions(batches: List[ProcessOutput]) -> Prediction:
    predictions = Prediction(y_pred=[], y_score=[])
    for batch in batches:
        predictions.add_new_predictions(batch.prediction)
    return predictions

def get_sequence_info(batches: List[ProcessOutput]) -> dict:
    merged_info = defaultdict(list)
    for batch in batches:
        for key, value_list in batch.sequence_info.items():
            merged_info[key].extend(value_list)
    return dict(merged_info)