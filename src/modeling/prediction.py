from abc import ABC, abstractmethod
from typing import Optional, List

from torch.nn.functional import softmax

from src.utils.base_parameters import BaseParameters


class Prediction(BaseParameters):
    y_pred: Optional[List] = None
    y_score: Optional[List] = None

    def add_new_predictions(self, prediction: "Prediction"):
        # y_pred
        if prediction.y_pred:
            if self.y_pred is None:
                self.y_pred = prediction.y_pred.copy()
            else:
                self.y_pred.extend(prediction.y_pred)

        # y_score
        if prediction.y_score:
            if self.y_score is None:
                self.y_score = prediction.y_score.copy()
            else:
                self.y_score.extend(prediction.y_score)

# YPred
class YPredPredictionStats(ABC):
    @abstractmethod
    def calculate(self, model_output):
        pass

class ClassificationYPred(YPredPredictionStats):
    def calculate(self, model_output):
        return model_output.argmax(dim=1).cpu().detach().numpy().tolist()

class RegressionYPred(YPredPredictionStats):
    def calculate(self, model_output):
        return model_output.cpu().detach().numpy().tolist()


# YScore
class YScorePredictionStats(ABC):
    @abstractmethod
    def calculate(self, model_output):
        pass

class BinaryClassificationYScore(YScorePredictionStats):
    def calculate(self, model_output):
        return softmax(model_output, dim=1)[:, 1].cpu().detach().numpy().tolist() # Probability class 1

class MulticlassClassificationYScore(YScorePredictionStats):
    def calculate(self, model_output):
        return softmax(model_output, dim=1).cpu().detach().numpy().tolist()


# prediction making
class PredictionMaking(ABC):
    @abstractmethod
    def prediction_making(self, model_output) -> Prediction:
        pass

class BinaryClassificationMaking(PredictionMaking):
    def prediction_making(self, model_output) -> Prediction:
        return Prediction(
            y_pred=ClassificationYPred().calculate(model_output),
            y_score=BinaryClassificationYScore().calculate(model_output)
        )

class MulticlassClassificationMaking(PredictionMaking):
    def prediction_making(self, model_output) -> Prediction:
        return Prediction(
            y_pred=ClassificationYPred().calculate(model_output),
            y_score=MulticlassClassificationYScore().calculate(model_output)
        )

class RegressionMaking(PredictionMaking):
    def prediction_making(self, model_output) -> Prediction:
        return Prediction(
            y_pred=RegressionYPred().calculate(model_output)
        )
