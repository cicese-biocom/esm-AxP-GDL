from typing import List
from workflow.parameters_setter import ParameterSetter
import numpy as np
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from graph.tertiary_structure_handler import predict_tertiary_structures, load_tertiary_structures
from graph.edge_construction_functions import EdgeConstructionContext
from utils.scrambling import random_coordinate_matrix


def get_edges(workflow_settings: ParameterSetter, data: pd.DataFrame, esm2_contact_maps):
    if workflow_settings.tertiary_structure_method:
        atom_coordinates_matrices = predict_tertiary_structures(workflow_settings, data)
    else:
        atom_coordinates_matrices, data = load_tertiary_structures(workflow_settings, data)

    if workflow_settings.validation_mode == 'coordinate_scrambling' and workflow_settings.mode == 'training':
        min_values, max_values = _get_intervals_for_coordinate_axes(atom_coordinates_matrices)

        partitions = data['partition']
        with tqdm(range(len(atom_coordinates_matrices)), total=len(atom_coordinates_matrices),
                  desc="Scrambling the coordinates ", disable=False) as progress:
            for i, atom_coordinates_matrix in enumerate(atom_coordinates_matrices):
                # only the coordinates belonging to the training set will be scrambled
                # https://dl.acm.org/doi/10.1145/3446776
                if partitions[i] == 1:
                    atom_coordinates_matrices[i] = random_coordinate_matrix(atom_coordinates_matrix, min_values, max_values)
                progress.update(1)

    adjacency_matrices, weights_matrices = _construct_edges(atom_coordinates_matrices,
                                                            data['sequence'],
                                                            esm2_contact_maps,
                                                            workflow_settings)
    return adjacency_matrices, weights_matrices, data


def _get_intervals_for_coordinate_axes(atom_coordinates_matrices):
    atom_coordinates = np.concatenate(atom_coordinates_matrices, axis=0)
    coordinate_min = np.min(atom_coordinates, axis=0)
    coordinate_max = np.max(atom_coordinates, axis=0)
    return coordinate_min, coordinate_max


def _construct_edges(atom_coordinates_matrices: np.array, sequences: List[str], esm2_contact_maps, workflow_settings: ParameterSetter):
    num_cores = multiprocessing.cpu_count()

    if not workflow_settings.use_esm2_contact_map:
        esm2_contact_maps = [None] * len(atom_coordinates_matrices)

    args = [(workflow_settings.edge_construction_functions,
             workflow_settings.distance_function,
             workflow_settings.distance_threshold,
             atom_coordinates,
             sequence,
             esm2_contact_map,
             workflow_settings.use_edge_attr
             ) for (atom_coordinates, sequence, esm2_contact_map) in
            zip(atom_coordinates_matrices, sequences, esm2_contact_maps)]

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
