from typing import Tuple
import numpy as np
from utils.distances import distance
from functools import partial


class Edges:
    """
    """

    def compute_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        pass


class EmptyEdges(Edges):
    def __init__(self, number_of_amino_acid: int) -> None:
        self._number_of_amino_acid = number_of_amino_acid

    @property
    def number_of_amino_acid(self) -> Edges:
        return self._number_of_amino_acid

    def compute_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.zeros((self.number_of_amino_acid, self.number_of_amino_acid), dtype=int), np.empty((0, 0))


class EdgeConstructionFunction(Edges):
    _edges: Edges = None

    def __init__(self, edges: Edges) -> None:
        self._edges = edges

    @property
    def edges(self) -> Edges:
        return self._edges

    def compute_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._edges.compute_edges()


class PeptideBackbone(EdgeConstructionFunction):
    def __init__(self, edges: Edges, distance_function: str, atom_coordinates: np.ndarray, use_edge_attr: bool):
        super().__init__(edges)
        self._distance_function = distance_function
        self._atom_coordinates = atom_coordinates
        self._use_edge_attr = use_edge_attr

    @property
    def distance_function(self) -> str:
        return self._distance_function

    @property
    def use_edge_attr(self) -> str:
        return self._use_edge_attr
    
    @property
    def atom_coordinates(self) -> str:
        return self._atom_coordinates

    def compute_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        adjacency_matrix, weight_matrix = self.edges.compute_edges()

        number_of_amino_acid = len(self.atom_coordinates)
        new_weights_matrix = np.zeros((number_of_amino_acid, number_of_amino_acid), dtype=np.float64)

        for i in range(number_of_amino_acid - 1):
            adjacency_matrix[i][i + 1] = 1
            adjacency_matrix[i + 1][i] = 1

            if self.use_edge_attr and self.distance_function:
                dist = distance(self.atom_coordinates[i], self.atom_coordinates[i + 1], self.distance_function)
                new_weights_matrix[i][i + 1] = dist
                new_weights_matrix[i + 1][i] = dist

        if self.use_edge_attr and self.distance_function:
            new_weights_matrix = np.expand_dims(new_weights_matrix, -1)
            if weight_matrix.size > 0:
                return adjacency_matrix, np.concatenate((weight_matrix, new_weights_matrix), axis=-1)
            else:
                return adjacency_matrix, new_weights_matrix
        else:
            return adjacency_matrix, weight_matrix


class ESM2ContactMap(EdgeConstructionFunction):
    def __init__(self, edges: Edges, esm2_contact_map: Tuple[np.ndarray, np.ndarray], use_edge_attr: bool):
        super().__init__(edges)
        self._esm2_contact_map = esm2_contact_map
        self._use_edge_attr = use_edge_attr

    @property
    def esm2_contact_map(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._esm2_contact_map

    @property
    def use_edge_attr(self) -> str:
        return self._use_edge_attr

    def compute_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        adjacency_matrix, weight_matrix = self.edges.compute_edges()

        number_of_amino_acid = len(self.esm2_contact_map)
        new_weights_matrix = np.zeros((number_of_amino_acid, number_of_amino_acid), dtype=np.float64)

        for i in range(number_of_amino_acid):
            for j in range(i + 1, number_of_amino_acid):
                adjacency_matrix[i][j] = adjacency_matrix[i][j] or (1 if self.esm2_contact_map[i][j] > 0.5 else 0)
                adjacency_matrix[j][i] = adjacency_matrix[i][j]

                if self.use_edge_attr:
                    new_weights_matrix[i][j] = self.esm2_contact_map[i][j]
                    new_weights_matrix[j][i] = self.esm2_contact_map[i][j]

        if self.use_edge_attr:
            new_weights_matrix = np.expand_dims(new_weights_matrix, -1)
            if weight_matrix.size > 0:
                return adjacency_matrix, np.concatenate((weight_matrix, new_weights_matrix), axis=-1)
            else:
                return adjacency_matrix, new_weights_matrix
        else:
            return adjacency_matrix, weight_matrix


class DistanceBasedThreshold(EdgeConstructionFunction):
    def __init__(self, edges: Edges, distance_function: str, threshold: float, atom_coordinates: np.ndarray,
                 use_edge_attr: bool):
        super().__init__(edges)
        self._distance_function = distance_function
        self._threshold = threshold
        self._atom_coordinates = atom_coordinates
        self._use_edge_attr = use_edge_attr

    @property
    def distance_function(self) -> str:
        return self._distance_function

    @property
    def threshold(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._threshold

    @property
    def use_edge_attr(self) -> str:
        return self._use_edge_attr

    @property
    def atom_coordinates(self) -> str:
        return self._atom_coordinates

    def compute_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        adjacency_matrix, weight_matrix = self.edges.compute_edges()

        number_of_amino_acid = len(self.atom_coordinates)
        new_weights_matrix = np.zeros((number_of_amino_acid, number_of_amino_acid), dtype=np.float64)

        for i in range(number_of_amino_acid):
            for j in range(i + 1, number_of_amino_acid):
                dist = distance(self.atom_coordinates[i], self.atom_coordinates[j], self.distance_function)
                if dist <= self.threshold:
                    adjacency_matrix[i][j] = 1
                    adjacency_matrix[j][i] = 1
                    
                    if self.use_edge_attr:
                        new_weights_matrix[i][j] = dist
                        new_weights_matrix[j][i] = dist

        if self.use_edge_attr:
            new_weights_matrix = np.expand_dims(new_weights_matrix, -1)
            if weight_matrix.size > 0:
                return adjacency_matrix, np.concatenate((weight_matrix, new_weights_matrix), axis=-1)
            else:
                return adjacency_matrix, new_weights_matrix
        else:
            return adjacency_matrix, weight_matrix


class EdgeConstructionContext:
    @staticmethod
    def compute_edges(args):
        edge_construction_functions, distance_function, distance_threshold, esm2_contact_map, atom_coordinates, use_edge_attr = args

        construction_functions = [
            ('distance_based_threshold',
             partial(DistanceBasedThreshold,
                     edges=None,
                     distance_function=distance_function,
                     threshold=distance_threshold,
                     atom_coordinates=atom_coordinates,
                     use_edge_attr=use_edge_attr
                     )),
            ('esm2_contact_map',
             partial(ESM2ContactMap,
                     edges=None,                     
                     esm2_contact_map=esm2_contact_map,
                     use_edge_attr=use_edge_attr)),
            ('peptide_backbone',
             partial(PeptideBackbone,
                     edges=None,
                     distance_function=distance_function,
                     atom_coordinates=atom_coordinates,
                     use_edge_attr=use_edge_attr
                     ))
        ]

        number_of_amino_acid = len(atom_coordinates) if atom_coordinates is not None else len(esm2_contact_map)
        edges_functions = EmptyEdges(number_of_amino_acid)

        for name in edge_construction_functions:
            for func_name, func in construction_functions:
                if func_name == name:
                    params = func.keywords
                    params['edges'] = edges_functions
                    edges_functions = func(**params)
                    break

        return edges_functions.compute_edges()


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
    # distance_function = None
    distance_threshold = 5

    # function_names = ['distance_based_threshold', 'esm2_contact_map', 'peptide_backbone']
    edge_construction_functions = ['peptide_backbone', 'esm2_contact_map', 'distance_based_threshold']

    use_edge_attr = True
    args = (edge_construction_functions, distance_function, distance_threshold, esm2_contact_map,
            atom_coordinates, use_edge_attr)

    edges = EdgeConstructionContext()

    adjacency_matrix, weight_matrix = edges.compute_edges(args)

    a = (adjacency_matrix, weight_matrix)
