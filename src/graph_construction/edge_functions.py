from typing import Tuple
import numpy as np
from numpy import ndarray

from src.config.types import EdgeConstructionFunction
from functools import partial

from src.graph_construction.distance import DistanceContext


class EdgesComponent:
    """
    """

    def compute_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        pass


class EmptyEdgesComponent(EdgesComponent):
    def __init__(self, number_of_amino_acid: int) -> None:
        self._number_of_amino_acid = number_of_amino_acid

    def compute_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.zeros((self._number_of_amino_acid, self._number_of_amino_acid), dtype=int), np.empty((0, 0))


class EdgeConstructionFunctionDecorator(EdgesComponent):
    _edges: EdgesComponent = None

    def __init__(self, edges_component: EdgesComponent) -> None:
        self._edges_component = edges_component

    def compute_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._edges_component.compute_edges()


class SequenceBasedDecorator(EdgeConstructionFunctionDecorator):
    def __init__(self, edges_component: EdgesComponent, distance_context: DistanceContext, atom_coordinates: np.ndarray, sequence: str,
                 use_edge_attr: bool):
        super().__init__(edges_component)
        self._distance_context = distance_context
        self._atom_coordinates = atom_coordinates
        self._sequence = sequence
        self._use_edge_attr = use_edge_attr

    @property
    def distance_context(self) -> DistanceContext:
        return self._distance_context

    @property
    def use_edge_attr(self) -> bool:
        return self._use_edge_attr
    
    @property
    def atom_coordinates(self) -> ndarray:
        return self._atom_coordinates

    @property
    def sequence(self) -> str:
        return self._sequence

    def compute_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        adjacency_matrix, weight_matrix = self._edges_component.compute_edges()
        number_of_amino_acid = len(self.sequence)

        use_weights = self.use_edge_attr and self.distance_context
        new_weights_matrix = None

        if use_weights:
            new_weights_matrix = np.zeros((number_of_amino_acid, number_of_amino_acid), dtype=np.float64)

        for i in range(number_of_amino_acid - 1):
            adjacency_matrix[i][i + 1] = 1

            if use_weights:
                dist = self.distance_context.compute(self.atom_coordinates[i], self.atom_coordinates[i + 1])
                new_weights_matrix[i][i + 1] = dist

        if use_weights:
            new_weights_matrix = np.expand_dims(new_weights_matrix, -1)
            if weight_matrix.size > 0:
                weight_matrix = np.concatenate((weight_matrix, new_weights_matrix), axis=-1)
            else:
                weight_matrix = new_weights_matrix

        return adjacency_matrix, weight_matrix


class ESM2ContactMapDecorator(EdgeConstructionFunctionDecorator):
    def __init__(self, edges_component: EdgesComponent, esm2_contact_map: Tuple[np.ndarray, np.ndarray],
                 probability_threshold: float, use_edge_attr: bool):
        super().__init__(edges_component)
        self._esm2_contact_map = esm2_contact_map
        self._probability_threshold = probability_threshold
        self._use_edge_attr = use_edge_attr

    def compute_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        adjacency_matrix, weight_matrix = self._edges_component.compute_edges()

        number_of_amino_acid = len(self._esm2_contact_map)
        new_weights_matrix = np.zeros((number_of_amino_acid, number_of_amino_acid), dtype=np.float64)

        for i in range(number_of_amino_acid):
            for j in range(i + 1, number_of_amino_acid):
                adjacency_matrix[i][j] = adjacency_matrix[i][j] or (
                    1 if self._esm2_contact_map[i][j] > self._probability_threshold else 0)
                adjacency_matrix[j][i] = adjacency_matrix[i][j]

                if self._use_edge_attr:
                    new_weights_matrix[i][j] = self._esm2_contact_map[i][j]
                    new_weights_matrix[j][i] = self._esm2_contact_map[i][j]

        if self._use_edge_attr:
            new_weights_matrix = np.expand_dims(new_weights_matrix, -1)
            if weight_matrix.size > 0:
                return adjacency_matrix, np.concatenate((weight_matrix, new_weights_matrix), axis=-1)
            else:
                return adjacency_matrix, new_weights_matrix
        else:
            return adjacency_matrix, weight_matrix


class DistanceBasedThresholdDecorator(EdgeConstructionFunctionDecorator):
    def __init__(self, edges_component: EdgesComponent, distance_context: DistanceContext, threshold: float, atom_coordinates: np.ndarray,
                 use_edge_attr: bool):
        super().__init__(edges_component)
        self._distance_context= distance_context
        self._threshold = threshold
        self._atom_coordinates = atom_coordinates
        self._use_edge_attr = use_edge_attr


    def compute_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        adjacency_matrix, weight_matrix = self._edges_component.compute_edges()

        number_of_amino_acid = len(self._atom_coordinates)
        new_weights_matrix = np.zeros((number_of_amino_acid, number_of_amino_acid), dtype=np.float64)

        for i in range(number_of_amino_acid):
            for j in range(i + 1, number_of_amino_acid):
                dist = self._distance_context.compute(self._atom_coordinates[i], self._atom_coordinates[j])
                if 0 < dist <= self._threshold:
                    adjacency_matrix[i][j] = 1
                    adjacency_matrix[j][i] = 1
                    
                    if self._use_edge_attr:
                        new_weights_matrix[i][j] = dist
                        new_weights_matrix[j][i] = dist

        if self._use_edge_attr:
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
        edge_construction_functions, distance_function, distance_threshold, atom_coordinates, sequence, \
        esm2_contact_map, probability_threshold, use_edge_attr = args

        distance = DistanceContext(distance_function)

        construction_functions = [
            (EdgeConstructionFunction.DISTANCE_BASED_THRESHOLD,
             partial(DistanceBasedThresholdDecorator,
                     edges_component=None,
                     distance_context=distance,
                     threshold=distance_threshold,
                     atom_coordinates=atom_coordinates,
                     use_edge_attr=use_edge_attr
                     )),
            (EdgeConstructionFunction.ESM2_CONTACT_MAP,
             partial(ESM2ContactMapDecorator,
                     edges_component=None,
                     esm2_contact_map=esm2_contact_map,
                     probability_threshold=probability_threshold,
                     use_edge_attr=use_edge_attr
                     )),
            (EdgeConstructionFunction.SEQUENCE_BASED,
             partial(SequenceBasedDecorator,
                     edges_component=None,
                     distance_context=distance,
                     atom_coordinates=atom_coordinates,
                     sequence=sequence,
                     use_edge_attr=use_edge_attr
                     ))
        ]

        number_of_amino_acid = len(sequence)
        edges_functions = EmptyEdgesComponent(number_of_amino_acid)

        for name in edge_construction_functions:
            for func_name, func in construction_functions:
                if func_name == name:
                    params = func.keywords
                    params['edges_component'] = edges_functions
                    edges_functions = func(**params)
                    break

        return edges_functions.compute_edges()