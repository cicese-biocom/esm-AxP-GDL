from typing import Tuple
import numpy as np
from utils.distances import distance
from functools import partial


class Edges:
    """
    """

    def compute_edges(self, atom_coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass


class EmptyEdges(Edges):
    def compute_edges(self, atom_coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        amino_acid_number = len(atom_coordinates)
        return np.zeros((amino_acid_number, amino_acid_number), dtype=int), np.empty((0, 0))


class EdgeConstructionFunction(Edges):
    _edges: Edges = None

    def __init__(self, edges: Edges) -> None:
        self._edges = edges

    @property
    def edges(self) -> Edges:
        return self._edges

    def compute_edges(self, atom_coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self._edges.compute_edges(atom_coordinates=atom_coordinates)


class PeptideBackbone(EdgeConstructionFunction):
    def __init__(self, edges: Edges, distance_function: str):
        super().__init__(edges)
        self._distance_function = distance_function

    @property
    def distance_function(self) -> str:
        return self._distance_function

    def compute_edges(self, atom_coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        adjacency_matrix, weight_matrix = self.edges.compute_edges(atom_coordinates=atom_coordinates)

        amino_acid_number = len(atom_coordinates)
        new_weights_matrix = np.zeros((amino_acid_number, amino_acid_number), dtype=np.float64)

        for i in range(amino_acid_number - 1):
            adjacency_matrix[i][i + 1] = 1
            adjacency_matrix[i + 1][i] = 1

            if self._distance_function:
                dist = distance(atom_coordinates[i], atom_coordinates[i + 1], self._distance_function)
                new_weights_matrix[i][i + 1] = dist
                new_weights_matrix[i + 1][i] = dist

                new_weights_matrix = np.expand_dims(new_weights_matrix, -1)

            if weight_matrix.size > 0:
                new_weights_matrix = np.concatenate((weight_matrix, new_weights_matrix), axis=-1)
                return adjacency_matrix, new_weights_matrix
            else:
                return adjacency_matrix, weight_matrix


class ESM2ContactMap(EdgeConstructionFunction):
    def __init__(self, edges: Edges, esm2_contact_map: Tuple[np.ndarray, np.ndarray]):
        super().__init__(edges)
        self._esm2_contact_map = esm2_contact_map

    @property
    def esm2_contact_map(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._esm2_contact_map

    def compute_edges(self, atom_coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        adjacency_matrix, weight_matrix = self.edges.compute_edges(atom_coordinates=atom_coordinates)

        amino_acid_number = len(atom_coordinates)

        for i in range(amino_acid_number):
            for j in range(i + 1, amino_acid_number):
                adjacency_matrix[i][j] = adjacency_matrix[i][j] or self.esm2_contact_map[i][j]
                adjacency_matrix[j][i] = adjacency_matrix[i][j]

        return adjacency_matrix, weight_matrix


class DistanceThreshold(EdgeConstructionFunction):
    def __init__(self, edges: Edges, distance_function: str, threshold: float):
        super().__init__(edges)
        self._distance_function = distance_function
        self._threshold = threshold

    @property
    def distance_function(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._distance_function

    @property
    def threshold(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._threshold

    def compute_edges(self, atom_coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        adjacency_matrix, weight_matrix = self.edges.compute_edges(atom_coordinates=atom_coordinates)

        amino_acid_number = len(atom_coordinates)
        new_weights_matrix = np.zeros((amino_acid_number, amino_acid_number), dtype=np.float64)

        for i in range(amino_acid_number):
            for j in range(i + 1, amino_acid_number):
                dist = distance(atom_coordinates[i], atom_coordinates[j], self.distance_function)
                if dist <= self.threshold:
                    adjacency_matrix[i][j] = 1
                    adjacency_matrix[j][i] = 1
                    new_weights_matrix[i][j] = dist
                    new_weights_matrix[j][i] = dist

        new_weights_matrix = np.expand_dims(new_weights_matrix, -1)

        if weight_matrix.size > 0:
            new_weights_matrix = np.concatenate((weight_matrix, new_weights_matrix), axis=-1)
        return adjacency_matrix, new_weights_matrix


class EdgeConstructionContext:
    @staticmethod
    def compute_edges(args):
        edge_construction_functions, distance_function, distance_threshold, esm2_contact_map, atom_coordinates = args

        construction_functions = [
            ('distance_based_threshold',
             partial(DistanceThreshold,
                     edges=None,
                     distance_function=distance_function,
                     threshold=distance_threshold)),
            ('esm2_contact_map',
             partial(ESM2ContactMap,
                     edges=None,
                     esm2_contact_map=esm2_contact_map)),
            ('peptide_backbone',
             partial(PeptideBackbone,
                     edges=None,
                     distance_function=distance_function))
        ]

        edges_functions = EmptyEdges()

        for name in edge_construction_functions:
            for func_name, func in construction_functions:
                if func_name == name:
                    params = func.keywords
                    params['edges'] = edges_functions
                    edges_functions = func(**params)
                    break

        return edges_functions.compute_edges(atom_coordinates=atom_coordinates)


if __name__ == "__main__":
    ####################
    # parameters
    atom_coordinates = np.array([
        [-7.186999798, 5.524000168, 13.19900036],
        [-8.18200016, 3.071000099, 10.44299984],
        [-8.44699955, 5.191999912, 7.461999893]
    ])

    esm2_contact_map = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 0, 0]
    ])

    distance_function = 'euclidean'
    threshold = 5

    function_names = ['distance_threshold', 'esm2_contact_map', 'peptide_backbone']

    edge_construction_funcs = [
        ('distance_threshold',
         partial(DistanceThreshold, edges=None, distance_function=distance_function, threshold=threshold)),
        ('esm2_contact_map',
         partial(ESM2ContactMap, edges=None, esm2_contact_map=esm2_contact_map)),
        ('peptide_backbone',
         partial(PeptideBackbone, edges=None, distance_function=distance_function))
    ]

    # run
    edges = EmptyEdges()

    for name in function_names:
        for func_name, func in edge_construction_funcs:
            if func_name == name:
                params = func.keywords
                params['edges'] = edges
                edges = func(**params)
                break

    adjacency_matrix_2, weight_matrix_2 = \
        edges.compute_edges(atom_coordinates=atom_coordinates)

    ####################

    empty_edges = EmptyEdges()
    adjacency_matrix, weight_matrix = empty_edges.compute_edges(atom_coordinates=atom_coordinates)

    peptide_backbone = PeptideBackbone(edges=empty_edges,
                                       distance_function=distance_function)

    peptide_bond_with_distance_threshold = DistanceThreshold(edges=peptide_backbone,
                                                             distance_function=distance_function,
                                                             threshold=threshold)

    peptide_bond_with_esm2_contact_map = ESM2ContactMap(edges=peptide_backbone, esm2_contact_map=esm2_contact_map)

    adjacency_matrix, weight_matrix = \
        peptide_bond_with_esm2_contact_map.compute_edges(atom_coordinates=atom_coordinates)

    print(adjacency_matrix, weight_matrix)
