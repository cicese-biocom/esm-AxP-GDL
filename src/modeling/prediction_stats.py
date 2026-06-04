from abc import ABC, abstractmethod

from torch.nn.functional import softmax


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
