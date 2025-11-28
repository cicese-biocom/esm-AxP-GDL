import logging
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
from pydantic.v1 import PositiveFloat
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from src.config.types import (
    ValidationMode,
    ExecutionMode,
    EdgeConstructionFunction, DistanceFunction,
)

from src.graph_construction.edge_functions import EdgeConstructionContext
from src.utils.base_parameters import BaseParameters
from src.graph_construction.tertiary_structure import (
    predict_tertiary_structures,
    load_tertiary_structures,
    Predict3DStructuresParameters,
    Load3DStructuresParameters
)


class RandomCoordinatesParameters(BaseParameters):
    execution_mode: ExecutionMode
    validation_mode: Optional[ValidationMode]
    randomness_percentage: Optional[PositiveFloat]
    atom_coordinates_matrices: List
    data: pd.DataFrame


class BuildEdgesParameters(BaseParameters):
    execution_mode: ExecutionMode
    validation_mode: Optional[ValidationMode]
    randomness_percentage: Optional[PositiveFloat]
    tertiary_structure_method: Optional[str]
    pdb_path: Optional[Path]
    amino_acid_representation: str
    non_pdb_bound_sequences_file: Path
    edge_construction_functions: List[EdgeConstructionFunction]
    distance_function: Optional[DistanceFunction]
    distance_threshold: Optional[PositiveFloat]
    probability_threshold: Optional[PositiveFloat]
    use_edge_attr: Optional[bool]
    data: pd.DataFrame
    esm2_contact_maps: List

class GenerateEdgesParameters(BaseParameters):
    edge_construction_functions: List[EdgeConstructionFunction]
    distance_function: Optional[DistanceFunction]
    distance_threshold: Optional[PositiveFloat]
    probability_threshold: Optional[PositiveFloat]
    use_edge_attr: Optional[bool]
    atom_coordinates_matrices: List
    data: pd.DataFrame
    esm2_contact_maps: List


def build_edges(build_edges_parameters: BuildEdgesParameters):
    if build_edges_parameters.tertiary_structure_method:
        atom_coordinates_matrices = predict_tertiary_structures(Predict3DStructuresParameters(**build_edges_parameters.dict()))
    else:
        atom_coordinates_matrices = load_tertiary_structures(Load3DStructuresParameters(**build_edges_parameters.dict()))

    atom_coordinates_matrices = _apply_random_coordinates(
        RandomCoordinatesParameters(
            **build_edges_parameters.dict(),
            atom_coordinates_matrices=atom_coordinates_matrices,
        )
    )

    adjacency_matrices, weights_matrices = _generate_edges(
        GenerateEdgesParameters(
            **build_edges_parameters.dict(),
            atom_coordinates_matrices=atom_coordinates_matrices
        )
    )

    return adjacency_matrices, weights_matrices


def _get_range_for_every_coordinate(atom_coordinates_matrices):
    atom_coordinates = np.concatenate(atom_coordinates_matrices, axis=0)
    coordinate_min = np.min(atom_coordinates, axis=0)
    coordinate_max = np.max(atom_coordinates, axis=0)
    return coordinate_min, coordinate_max


def _apply_random_coordinates(random_coordinates_parameters: RandomCoordinatesParameters):
    if (random_coordinates_parameters.validation_mode == ValidationMode.RANDOM_COORDINATES
            and random_coordinates_parameters.execution_mode == ExecutionMode.TRAIN):

        logging.getLogger('workflow_logger'). \
            warning(f"The framework is running in validation mode with workflow_settings.validation_mode: "
                    f"{random_coordinates_parameters.validation_mode.value} and "
                    f"workflow_settings.randomness_percentage: {random_coordinates_parameters.randomness_percentage}")

        partitions = random_coordinates_parameters.data['partition']
        min_values, max_values = _get_range_for_every_coordinate(random_coordinates_parameters.atom_coordinates_matrices)

        with tqdm(range(len(random_coordinates_parameters.atom_coordinates_matrices)), total=len(random_coordinates_parameters.atom_coordinates_matrices),
                  desc="Random coordinates ", disable=False) as progress:
            for i, atom_coordinates_matrix in enumerate(random_coordinates_parameters.atom_coordinates_matrices):
                # only the coordinates belonging to the training set will be randomly created
                # https://dl.acm.org/doi/10.1145/3446776
                if partitions[i] == 1:
                    random_coordinates_parameters.atom_coordinates_matrices[i] = random_coordinate_matrix(
                        atom_coordinates_matrix,
                        random_coordinates_parameters.randomness_percentage,
                        min_values,
                        max_values
                    )
                progress.update(1)

    return random_coordinates_parameters.atom_coordinates_matrices


def _generate_edges(generate_edges_parameters: GenerateEdgesParameters):
    num_cores = multiprocessing.cpu_count()

    sequences = generate_edges_parameters.data['sequence']

    args = [(generate_edges_parameters.edge_construction_functions,
             generate_edges_parameters.distance_function,
             generate_edges_parameters.distance_threshold,
             atom_coordinates,
             sequence,
             esm2_contact_map,
             generate_edges_parameters.probability_threshold,
             generate_edges_parameters.use_edge_attr
             ) for (atom_coordinates, sequence, esm2_contact_map) in
            zip(generate_edges_parameters.atom_coordinates_matrices,
                sequences,
                generate_edges_parameters.esm2_contact_maps
                )]

    with ProcessPoolExecutor(max_workers=num_cores) as pool:
        with tqdm(range(len(args)), total=len(args), desc="Generating adjacency matrices", disable=False) as progress:
            futures = []
            for arg in args:
                future = pool.submit(EdgeConstructionContext.compute_edges, arg)
                future.add_done_callback(lambda p: progress.update())
                futures.append(future)

            adjacency_matrices = [future.result()[0] for future in futures]
            weights_matrices = [future.result()[1] for future in futures]

    return adjacency_matrices, weights_matrices


def random_coordinate_matrix(matrix, randomness_percentage, min, max):
    num_rows = matrix.shape[0]
    num_to_scramble = int(num_rows * (randomness_percentage / 100))
    indices = np.random.choice(num_rows, num_to_scramble, replace=False)

    atom_coordinates = matrix
    coord_0 = np.random.uniform(min[0], max[0], size=num_to_scramble)
    coord_1 = np.random.uniform(min[1], max[1], size=num_to_scramble)
    coord_2 = np.random.uniform(min[2], max[2], size=num_to_scramble)
    for coord_idx, matrix_idx in enumerate(indices):
        atom_coordinates[matrix_idx][0] = coord_0[coord_idx]
        atom_coordinates[matrix_idx][1] = coord_1[coord_idx]
        atom_coordinates[matrix_idx][2] = coord_2[coord_idx]

    return atom_coordinates