from workflow.parameters_setter import ParameterSetter
import numpy as np
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from .tertiary_structure_handler import predict_tertiary_structures, load_tertiary_structures
from .edge_construction_functions import EdgeConstructionContext


def get_edges(workflow_settings: ParameterSetter, data: pd.DataFrame, esm2_contact_maps):
    if workflow_settings.tertiary_structure_method:
        atom_coordinates_matrices = predict_tertiary_structures(workflow_settings, data)
    else:
        atom_coordinates_matrices, data = load_tertiary_structures(workflow_settings, data)

    adjacency_matrices, weights_matrices = _construct_edges(atom_coordinates_matrices, esm2_contact_maps, workflow_settings)
    return adjacency_matrices, weights_matrices, data


def _construct_edges(atom_coordinates_matrices: np.array, esm2_contact_maps, workflow_settings: ParameterSetter):
    num_cores = multiprocessing.cpu_count()

    if workflow_settings.use_esm2_contact_map:
        args = [(workflow_settings.edge_construction_functions,
                 workflow_settings.distance_function,
                 workflow_settings.distance_threshold,
                 esm2_contact_map,
                 atom_coordinates) for (atom_coordinates, esm2_contact_map) in
                zip(atom_coordinates_matrices, esm2_contact_maps)]
    else:
        args = [(workflow_settings.edge_construction_functions,
                 workflow_settings.distance_function,
                 workflow_settings.distance_threshold,
                 None,
                 atom_coordinates) for atom_coordinates in
                atom_coordinates_matrices]

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