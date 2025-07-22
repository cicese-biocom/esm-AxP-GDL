import numpy as np


class DistanceStrategy:
    """
    Interface for distance calculation strategies.
    """
    def compute(self, point1, point2):
        raise NotImplementedError("Subclasses should implement this method")


class EuclideanDistance(DistanceStrategy):
    def compute(self, point1, point2):
        return np.sqrt(np.sum(np.power(np.subtract(point1, point2), 2)))


class CanberraDistance(DistanceStrategy):
    def compute(self, point1, point2):
        return np.sum(np.divide(np.abs(point1 - point2), np.add(np.abs(point1), np.abs(point2))))


class LanceWilliamsDistance(DistanceStrategy):
    def compute(self, point1, point2):
        return np.divide(np.sum(np.abs(np.subtract(point1, point2))), np.sum(np.add(np.abs(point1), np.abs(point2))))


class ClarkDistance(DistanceStrategy):
    def compute(self, point1, point2):
        return np.sqrt(np.sum(np.power(np.divide(np.subtract(point1, point2), np.add(np.abs(point1), np.abs(point2))), 2)))


class SoergelDistance(DistanceStrategy):
    def compute(self, point1, point2):
        return np.divide(np.sum(np.abs(np.subtract(point1, point2))), np.sum(np.maximum(point1, point2)))


class BhattacharyyaDistance(DistanceStrategy):
    def compute(self, point1, point2):
        return np.sqrt(np.sum(np.power(np.subtract(np.sqrt(point1), np.sqrt(point2)), 2)))


class AngularSeparationDistance(DistanceStrategy):
    def compute(self, point1, point2):
        return np.subtract(1, np.divide(np.sum(np.multiply(point1, point2)),
                                        np.sqrt(np.dot(np.sum(np.power(point1, 2)), np.sum(np.power(point2, 2))))))


class DistanceContext:
    def __init__(self, strategy: DistanceStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: DistanceStrategy):
        self._strategy = strategy

    def compute(self, point1, point2):
        if len(point1) != len(point2):
            raise ValueError("The points do not have the same number of coordinates")
        return self._strategy.compute(point1, point2)


def compute_distance(point1, point2, distance_strategies):
    return distance_strategies.compute(point1, point2)

