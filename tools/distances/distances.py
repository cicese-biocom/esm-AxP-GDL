import numpy as np
#np.set_printoptions(precision=8, suppress=True)

def distance(point1, point2, distance_function):
    """
    Args:
        point1 (tuple): The coordinates of the first point
        point2 (tuple): The coordinates of the second point
        distance_function (str): The type of distance to calculate

    Returns:
        float: The calculated distance between the two points.
    """
    try:
        if len(point1) != len(point2):
            raise ValueError("The points do not have the same number of coordinates")

        if distance_function == 'euclidean':
            return _euclidean(point1, point2)
        elif distance_function == 'canberra':
            return _canberra(point1, point2)
        elif distance_function == 'lance_williams':
            return _lance_william(point1, point2)
        elif distance_function == 'clark':
            return _clark(point1, point2)
        elif distance_function == 'soergel':
            return _soergel(point1, point2)
        elif distance_function == 'bhattacharyya':
            return _bhattacharyya(point1, point2)
        elif distance_function == 'angular_separation':
            return _angular_separation(point1, point2)
        else:
            raise ValueError("Invalid distance name: " + str(distance_function))
    except Exception as e:
        raise ValueError(f"Error calculating distances: {distance_function}" + str(e))


def _euclidean(point1, point2):
    return np.round(np.sqrt(np.sum(np.power(np.subtract(point1, point2), 2))),8)

def _canberra(point1, point2):
    return np.round(np.sum(np.divide(np.abs(point1 - point2), np.add(np.abs(point1), np.abs(point2)))),8)


def _lance_william(point1, point2):
    return np.round(np.divide(np.sum(np.abs(np.subtract(point1, point2))), np.sum(np.add(np.abs(point1), np.abs(point2)))),8)


def _clark(point1, point2):
    return np.round(np.sqrt(np.sum(np.power(np.divide(np.subtract(point1, point2), np.add(np.abs(point1), np.abs(point2))), 2))),8)


def _soergel(point1, point2):
    return np.round(np.divide(np.sum(np.abs(np.subtract(point1, point2))), np.sum(np.maximum(point1, point2))),8)


def _bhattacharyya(point1, point2):
    return np.round(np.sqrt(np.sum(np.power(np.subtract(np.sqrt(point1), np.sqrt(point2)), 2))),8)


def _angular_separation(point1, point2):
    return np.round(np.subtract(1, np.divide(np.sum(np.multiply(point1, point2)), np.sqrt(np.dot(np.sum(np.power(point1, 2)), np.sum(np.power(point2, 2)))))), 8)


def translate_positive_coordinates(coordinates):
    min_x = min(min(coordinate[0] for coordinate in coordinates), 0)
    min_y = min(min(coordinate[1] for coordinate in coordinates), 0)
    min_z = min(min(coordinate[2] for coordinate in coordinates), 0)

    eps = 1e-6
    return [np.float64((coordinate[0] - min_x + eps, coordinate[1] - min_y + eps, coordinate[2] - min_z + eps)) for coordinate in coordinates]