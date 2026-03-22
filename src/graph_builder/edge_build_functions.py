from typing import Tuple, Dict
import numpy as np
from numpy import ndarray

from src.config.types import EdgeBuildFunction
from functools import partial

from src.graph_builder.distance_functions import DistanceContext


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


class EdgeBuildFunctionDecorator(EdgesComponent):
    _edges: EdgesComponent = None

    def __init__(self, edges_component: EdgesComponent) -> None:
        self._edges_component = edges_component

    def compute_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._edges_component.compute_edges()


class EmptyGraphDecorator(EdgeBuildFunctionDecorator):
    def __init__(self, edges_component: EdgesComponent, atom_coordinates: np.ndarray,
                 sequence: str,
                 use_edge_attr: bool):
        super().__init__(edges_component)

    def compute_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._edges_component.compute_edges()


class SequenceBasedDecorator(EdgeBuildFunctionDecorator):
    def __init__(self, edges_component: EdgesComponent, sequence: str):
        super().__init__(edges_component)
        self._sequence = sequence

    @property
    def sequence(self) -> str:
        return self._sequence

    def compute_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        adjacency_matrix, weight_matrix = self._edges_component.compute_edges()
        number_of_amino_acid = len(self.sequence)

        for i in range(number_of_amino_acid - 1):
            adjacency_matrix[i][i + 1] = 1

        return adjacency_matrix, weight_matrix


class ESM2ContactMapDecorator(EdgeBuildFunctionDecorator):
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
                    1 if self._esm2_contact_map[i][j] >= self._probability_threshold else 0)
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


class DistanceBasedThresholdDecorator(EdgeBuildFunctionDecorator):
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



class EdgeBuildContext:
    @staticmethod
    def compute_edges(args: Dict):
        """
        Constructs the list of edge-building functions based on the selected methods
        and only the parameters provided in 'args'.

        'Args' should be a dictionary with keys for the selected parameters.
        Only the parameters relevant for each method are passed to its decorator.
        """
        edge_build_functions = args.get('edge_build_functions', [])
        functions = []

        # Prepare distance context if distance_function is provided
        distance_context = DistanceContext(args['distance_function']) if 'distance_function' in args else None

        if EdgeBuildFunction.DISTANCE_BASED_THRESHOLD in edge_build_functions:
            functions.append(
                (
                    EdgeBuildFunction.DISTANCE_BASED_THRESHOLD,
                    partial(
                        DistanceBasedThresholdDecorator,
                        edges_component=None,
                        distance_context=distance_context,
                        threshold=args.get('distance_threshold'),
                        atom_coordinates=args.get('atom_coordinates'),
                        use_edge_attr=args.get('use_edge_attr')
                    )
                )
            )

        if EdgeBuildFunction.ESM2_CONTACT_MAP in edge_build_functions:
            functions.append(
                (
                    EdgeBuildFunction.ESM2_CONTACT_MAP,
                    partial(
                        ESM2ContactMapDecorator,
                        edges_component=None,
                        esm2_contact_map=args.get('esm2_contact_map'),
                        probability_threshold=args.get('probability_threshold'),
                        use_edge_attr=args.get('use_edge_attr')
                    )
                )
            )

        if EdgeBuildFunction.SEQUENCE_BASED in edge_build_functions:
            functions.append(
                (
                    EdgeBuildFunction.SEQUENCE_BASED,
                    partial(
                        SequenceBasedDecorator,
                        edges_component=None,
                        sequence=args.get('sequence'),
                    )
                )
            )

        if EdgeBuildFunction.EMPTY_GRAPH in edge_build_functions:
            functions.append(
                (
                    EdgeBuildFunction.EMPTY_GRAPH,
                    partial(
                        EdgeBuildFunctionDecorator,
                        edges_component=None
                    )
                )
            )

        number_of_amino_acid = len(args.get('sequence'))
        edges_functions = EmptyEdgesComponent(number_of_amino_acid)

        for name in edge_build_functions:
            for func_name, func in functions:
                if func_name == name:
                    build_graphs_parameters = func.keywords
                    build_graphs_parameters['edges_component'] = edges_functions
                    edges_functions = func(**build_graphs_parameters)
                    break

        return edges_functions.compute_edges()