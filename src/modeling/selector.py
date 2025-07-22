from abc import ABC, abstractmethod
from typing import Dict, Optional
from pydantic.v1 import PositiveInt
from torch import nn

from src.utils.dto import DTO


class ModelDTO(DTO):
    epoch: Optional[PositiveInt] = None
    model: Optional[nn.Module] = None
    model_state_dict: Optional[Dict] = None
    optimizer_state_dict: Optional[Dict] = None
    scheduler_state_dict: Optional[Dict] = None
    parameters: Optional[Dict] = None
    metrics: Optional[Dict] = None
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None


class BestModelSelector(ABC):
    @abstractmethod
    def select(self, best_model: ModelDTO, current_model: ModelDTO) -> ModelDTO:
        pass


class MaximumMCCBestModelSelector(BestModelSelector):
    def select(self, best_model: ModelDTO, current_model: ModelDTO) -> ModelDTO:
        if best_model.metrics is None:
            return current_model

        best_mcc = best_model.metrics.get("MCC", float("-inf"))
        current_mcc = current_model.metrics.get("MCC", float("-inf"))
        return current_model if current_mcc > best_mcc else best_model


class MinimumRMSEBestModelSelector(BestModelSelector):
    def select(self, best_model: ModelDTO, current_model: ModelDTO) -> ModelDTO:
        if best_model.metrics is None:
            return current_model

        best_rmse = best_model.metrics.get("RMSE", float("inf"))
        current_rmse = current_model.metrics.get("RMSE", float("inf"))
        return current_model if current_rmse < best_rmse else best_model
