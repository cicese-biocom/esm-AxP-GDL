from abc import ABC, abstractmethod
from typing import Optional, List

from torch.nn.functional import softmax

from src.utils.base_entity import BaseParameters


class Prediction(BaseParameters):
    y_pred: Optional[List] = None
    y_score: Optional[List] = None

    def extend(self, prediction_dto: "Prediction"):
        # y_pred
        if prediction_dto.y_pred:
            if self.y_pred is None:
                self.y_pred = prediction_dto.y_pred.copy()
            else:
                self.y_pred.extend(prediction_dto.y_pred)

        # y_score
        if prediction_dto.y_score:
            if self.y_score is None:
                self.y_score = prediction_dto.y_score.copy()
            else:
                self.y_score.extend(prediction_dto.y_score)

# YPred
class YPredProcessor(ABC):
    @abstractmethod
    def process(self, model_output):
        pass

class ClassificationYPred(YPredProcessor):
    def process(self, model_output):
        return model_output.argmax(dim=1).cpu().detach().numpy().tolist()

class RegressionYPred(YPredProcessor):
    def process(self, model_output):
        return model_output.cpu().detach().numpy().tolist()


# YScore
class YScoreProcessor(ABC):
    @abstractmethod
    def process(self, model_output):
        pass

class BinaryClassificationYScore(YScoreProcessor):
    def process(self, model_output):
        return softmax(model_output, dim=1)[:, 1].cpu().detach().numpy().tolist() # Probability class 1

class MulticlassClassificationYScore(YScoreProcessor):
    def process(self, model_output):
        return softmax(model_output, dim=1).cpu().detach().numpy().tolist()


class PredictionProcessor(ABC):
    @abstractmethod
    def process(self, model_output) -> Prediction:
        pass

class BinaryClassificationPredictionProcessor(PredictionProcessor):
    def process(self, model_output) -> Prediction:
        return Prediction(
            y_pred=ClassificationYPred().process(model_output),
            y_score=BinaryClassificationYScore().process(model_output)
        )


class MulticlassClassificationPredictionProcessor(PredictionProcessor):
    def process(self, model_output) -> Prediction:
        return Prediction(
            y_pred=ClassificationYPred().process(model_output),
            y_score=MulticlassClassificationYScore().process(model_output)
        )


class RegressionPredictionProcessor(PredictionProcessor):
    def process(self, model_output) -> Prediction:
        return Prediction(
            y_pred=RegressionYPred().process(model_output)
        )
