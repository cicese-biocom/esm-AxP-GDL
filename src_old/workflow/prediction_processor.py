from abc import abstractmethod
from dataclasses import dataclass
import numpy as np
from typing import Optional
from torch.nn.functional import softmax

@dataclass
class PredictionResult:
    y_pred: np.ndarray
    y_score: Optional[np.ndarray] = None


class PredictionProcessor:
    @abstractmethod
    def process(self, model_output) -> PredictionResult:
        """
        Processes the raw output of the model and returns (y_pred, y_score).
        - In classification: y_pred: np.ndarray, y_score: np.ndarray
        - In regression: y_pred: np.ndarray, y_score: None
        """
        pass


class ClassificationPrediction(PredictionProcessor):
    def process(self, model_output) -> PredictionResult:
        y_pred = model_output.argmax(dim=1).cpu().detach().numpy()
        y_score = softmax(model_output, dim=1)[:, 1].cpu().detach().numpy()
        return PredictionResult(y_pred=y_pred, y_score=y_score)


class RegressionPrediction(PredictionProcessor):
    def process(self, model_output) -> PredictionResult:
        y_pred = model_output.cpu().detach().numpy()
        return PredictionResult(y_pred=y_pred)


class PredictionProcessorContext:
    def __init__(self, processor: PredictionProcessor) -> None:
        self._processor = processor

    def process(self, model_output) -> PredictionResult:
        return self._processor.process(model_output)

