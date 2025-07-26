from abc import ABC, abstractmethod
import numpy as np

from src.config.types import DistanceFunction


class DistanceStrategy(ABC):
    """
    Interface for distance calculation strategies.
    """
    @abstractmethod
    def compute(self, point1, point2):
        pass

class EuclideanStrategy(DistanceStrategy):
    def compute(self, point1, point2):
        return np.sqrt(np.sum(np.power(np.subtract(point1, point2), 2)))


class CanberraStrategy(DistanceStrategy):
    def compute(self, point1, point2):
        return np.sum(np.divide(np.abs(point1 - point2), np.add(np.abs(point1), np.abs(point2))))


class LanceWilliamsStrategy(DistanceStrategy):
    def compute(self, point1, point2):
        return np.divide(np.sum(np.abs(np.subtract(point1, point2))), np.sum(np.add(np.abs(point1), np.abs(point2))))


class ClarkStrategy(DistanceStrategy):
    def compute(self, point1, point2):
        return np.sqrt(np.sum(np.power(np.divide(np.subtract(point1, point2), np.add(np.abs(point1), np.abs(point2))), 2)))


class SoergelStrategy(DistanceStrategy):
    def compute(self, point1, point2):
        return np.divide(np.sum(np.abs(np.subtract(point1, point2))), np.sum(np.maximum(point1, point2)))


class BhattacharyyaStrategy(DistanceStrategy):
    def compute(self, point1, point2):
        return np.sqrt(np.sum(np.power(np.subtract(np.sqrt(point1), np.sqrt(point2)), 2)))


class AngularSeparationStrategy(DistanceStrategy):
    def compute(self, point1, point2):
        return np.subtract(1, np.divide(np.sum(np.multiply(point1, point2)),
                                        np.sqrt(np.dot(np.sum(np.power(point1, 2)), np.sum(np.power(point2, 2))))))


class DistanceContext:
    def __init__(self, distance_function: DistanceFunction):
        self._strategy = {
            DistanceFunction.EUCLIDEAN: EuclideanStrategy(),
            DistanceFunction.CANBERRA: CanberraStrategy(),
            DistanceFunction.LANCE_WILLIAMS: LanceWilliamsStrategy(),
            DistanceFunction.CLARK: ClarkStrategy(),
            DistanceFunction.SOERGEL: SoergelStrategy(),
            DistanceFunction.BHATTACHARYYA: BhattacharyyaStrategy(),
            DistanceFunction.ANGULAR_SEPARATION: AngularSeparationStrategy()
        }[distance_function]

    def compute(self, point1, point2):
        if len(point1) != len(point2):
            raise ValueError("The points do not have the same number of coordinates")
        return self._strategy.compute(point1, point2)