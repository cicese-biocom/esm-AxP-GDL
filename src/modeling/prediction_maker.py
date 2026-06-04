from abc import ABC, abstractmethod
from typing import Optional, List

from src.modeling.prediction_stats import ClassificationYPred, RegressionYPred, BinaryClassificationYScore, \
    MulticlassClassificationYScore
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
