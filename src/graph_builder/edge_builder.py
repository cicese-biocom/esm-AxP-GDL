import logging
import multiprocessing
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
from pydantic.v1 import PositiveFloat
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from src.config.types import (
    ValidationMode,
    ExecutionMode,
    EdgeBuildFunction, DistanceFunction,
)

from src.graph_builder.edge_build_functions import EdgeBuildContext
from src.utils.base_parameters import BaseParameters
from src.graph_builder.tertiary_structure import (
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
    load_tertiary_structure: Optional[bool]
    pdb_path: Optional[Path]
    amino_acid_representation: Optional[str]
    non_pdb_bound_sequences_file: Path
    edge_build_functions: List[EdgeBuildFunction]
    distance_function: Optional[DistanceFunction]
    distance_threshold: Optional[PositiveFloat]
    probability_threshold: Optional[PositiveFloat]
    use_edge_attr: Optional[bool]
    data: pd.DataFrame
    esm2_contact_maps: Optional[List]


class GenerateEdgesParameters(BaseParameters):
    edge_build_functions: List[EdgeBuildFunction]
    distance_function: Optional[DistanceFunction]
    distance_threshold: Optional[PositiveFloat]
    probability_threshold: Optional[PositiveFloat]
    use_edge_attr: Optional[bool]
    atom_coordinates_matrices:  Optional[List]
    data: pd.DataFrame
    esm2_contact_maps: Optional[List]


def build_edges(build_edges_parameters: BuildEdgesParameters):
    """
    Builds adjacency and weight matrices for the given sequences and edge construction methods.

    - Loads or predicts tertiary structures only if DISTANCE_BASED_THRESHOLD is selected.
    - Applies random coordinates if requested.
    - Calls _generate_edges with only the parameters required by the selected methods.
    """
    edge_methods = build_edges_parameters.edge_build_functions

    # Prepare atom coordinates only if DISTANCE_BASED_THRESHOLD is required
    if EdgeBuildFunction.DISTANCE_BASED_THRESHOLD in edge_methods:
        if build_edges_parameters.load_tertiary_structure:
            atom_coordinates_matrices = load_tertiary_structures(
                Load3DStructuresParameters(**build_edges_parameters.dict())
            )
        else:
            atom_coordinates_matrices = predict_tertiary_structures(
                Predict3DStructuresParameters(**build_edges_parameters.dict())
            )

        # Apply random coordinates; the function internally checks if it should modify anything
        atom_coordinates_matrices = _apply_random_coordinates(
            RandomCoordinatesParameters(
                **build_edges_parameters.dict(),
                atom_coordinates_matrices=atom_coordinates_matrices
            )
        )

        generate_edges_params = GenerateEdgesParameters(
            **build_edges_parameters.dict(),
            atom_coordinates_matrices=atom_coordinates_matrices
        )
    else:
        # DISTANCE_BASED_THRESHOLD not selected → no atom coordinates
        generate_edges_params = GenerateEdgesParameters(
            **build_edges_parameters.dict()
        )

    # Generate edges passing only the parameters required per method
    adjacency_matrices, weights_matrices = _generate_edges(generate_edges_params)

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
    """
    Generates adjacency and weight matrices for all sequences in parallel,
    constructing per-sequence argument dictionaries that include only
    the parameters required by the selected edge_build_functions.
    """

    edge_methods = generate_edges_parameters.edge_build_functions
    sequences = generate_edges_parameters.data['sequence']
    num_cores = multiprocessing.cpu_count()

    # Build iterables dynamically
    iterables = [sequences]
    keys = ['sequence']

    if EdgeBuildFunction.DISTANCE_BASED_THRESHOLD in edge_methods:
        iterables.append(generate_edges_parameters.atom_coordinates_matrices)
        keys.append('atom_coordinates')

    if EdgeBuildFunction.ESM2_CONTACT_MAP in edge_methods:
        iterables.append(generate_edges_parameters.esm2_contact_maps)
        keys.append('esm2_contact_map')

    args_list = []

    for values in zip(*iterables):
        arg_dict = dict(zip(keys, values))
        arg_dict['edge_build_functions'] = edge_methods

        # DISTANCE_BASED_THRESHOLD parameters
        if EdgeBuildFunction.DISTANCE_BASED_THRESHOLD in edge_methods:
            arg_dict.update({
                'distance_function': generate_edges_parameters.distance_function,
                'distance_threshold': generate_edges_parameters.distance_threshold,
                'use_edge_attr': generate_edges_parameters.use_edge_attr,
            })

        # ESM2_CONTACT_MAP parameters
        if EdgeBuildFunction.ESM2_CONTACT_MAP in edge_methods:
            arg_dict.update({
                'esm2_contact_map': arg_dict['esm2_contact_map'],  # already mapped but explicit
                'probability_threshold': generate_edges_parameters.probability_threshold,
                'use_edge_attr': generate_edges_parameters.use_edge_attr,
            })

        # SEQUENCE_BASED → only sequence (already included)

        # EMPTY_GRAPH → only sequence (already included)

        args_list.append(arg_dict)

    # Parallel execution
    adjacency_matrices = []
    weights_matrices = []

    with ProcessPoolExecutor(max_workers=num_cores) as pool:
        with tqdm(range(len(args_list)), total=len(args_list), desc="Generating adjacency matrices") as progress:
            futures = []
            for arg_dict in args_list:
                future = pool.submit(EdgeBuildContext.compute_edges, arg_dict)
                future.add_done_callback(lambda p: progress.update())
                futures.append(future)

            for future in futures:
                adjacency, weights = future.result()
                adjacency_matrices.append(adjacency)
                weights_matrices.append(weights)

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