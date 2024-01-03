import torch
from torch import hub
import esm
import os
import argparse
from Bio.PDB import PDBParser
import io
import numpy as np
from sklearn.metrics import pairwise_distances
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.simplefilter('ignore', PDBConstructionWarning)
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import pandas as pd


def get_adjacency_and_weights_matrices(load_pdb, data, path, threshold, validation_config):
    num_cores = multiprocessing.cpu_count()
    distance_type = 'euclidean'
    atom_type = 'CA'

    atom_coordinates_matrices = _get_atom_coordinates(load_pdb, data, path, atom_type)

    coordinate_min, coordinate_max = _get_atom_coordinates_intervals(atom_coordinates_matrices)
    validation_config = (*validation_config, coordinate_min, coordinate_max)

    adjacency_matrices, weights_matrices = _compute_adjacency_and_weights_matrices(atom_coordinates_matrices, threshold,
                                                                                   distance_type, atom_type,
                                                                                   validation_config, num_cores)
    return adjacency_matrices, weights_matrices


def _compute_adjacency_and_weights_matrices(atom_coordinates_matrices, threshold, distance_type, atom_type, validation_config, num_cores):
    args = [(atom_coordinates, threshold, distance_type, atom_type, validation_config) for atom_coordinates in
            atom_coordinates_matrices]
    with ProcessPoolExecutor(max_workers=num_cores) as pool:
        with tqdm(range(len(args)), total=len(args), desc="Generating adjacency matrices", disable=False) as progress:
            futures = []
            for arg in args:
                future = pool.submit(_adjacency_and_weights_matrix, arg)
                future.add_done_callback(lambda p: progress.update())
                futures.append(future)

            adjacency_matrices = [future.result()[0] for future in futures]
            weights_matrices = [future.result()[1] for future in futures]

    return adjacency_matrices, weights_matrices


def _adjacency_and_weights_matrix(args):
    atom_coordinates, threshold, distance_type, atom_type, (
    validation_mode, scrambling_percentage, coordinate_min, coordinate_max) = args

    amino_acid_number = len(atom_coordinates)

    if validation_mode == 'coordinates_scrambling':
        atom_coordinates = np.zeros((atom_coordinates.shape))
        atom_coordinates[:, 0] = np.random.uniform(coordinate_min[0], coordinate_max[0], size=amino_acid_number)
        atom_coordinates[:, 1] = np.random.uniform(coordinate_min[1], coordinate_max[1], size=amino_acid_number)
        atom_coordinates[:, 2] = np.random.uniform(coordinate_min[2], coordinate_max[2], size=amino_acid_number)

    adjacency_matrix = np.zeros((amino_acid_number, amino_acid_number), dtype=np.int)
    weights_matrix = np.zeros((amino_acid_number, amino_acid_number), dtype=np.float64)

    if validation_mode == 'sequence_graph':
        for i in range(amino_acid_number-1):
            dist = _distance(atom_coordinates[i], atom_coordinates[i+1], distance_type)
            adjacency_matrix[i][i+1] = 1
            adjacency_matrix[i+1][i] = 1
            weights_matrix[i][i+1] = dist
            weights_matrix[i+1][i] = dist
    else:
        for i in range(amino_acid_number):
            for j in range(i + 1, amino_acid_number):
                dist = _distance(atom_coordinates[i], atom_coordinates[j], distance_type)
                if dist <= threshold:
                    adjacency_matrix[i][j] = 1
                    adjacency_matrix[j][i] = 1
                    weights_matrix[i][j] = dist
                    weights_matrix[j][i] = dist

    adjacency_matrix[np.eye(adjacency_matrix.shape[0]) == 1] = 0
    weights_matrix = np.expand_dims(weights_matrix, -1)

    return adjacency_matrix, weights_matrix

def _coordinates_scrambling():
    atom_coordinates = np.zeros((atom_coordinates.shape))
    atom_coordinates[:, 0] = np.random.uniform(coordinate_min[0], coordinate_max[0], size=amino_acid_number)
    atom_coordinates[:, 1] = np.random.uniform(coordinate_min[1], coordinate_max[1], size=amino_acid_number)
    atom_coordinates[:, 2] = np.random.uniform(coordinate_min[2], coordinate_max[2], size=amino_acid_number)

def _get_atom_coordinates(load_pdb, data, path, atom_type):
    ids = data.id
    atom_coordinates_matrices = []
    if load_pdb:
        with tqdm(range(len(ids)), total=len(ids), desc="Loading pdb files", disable=False) as progress:
            pdbs = []
            for id in ids:
                pdb_file = os.path.join(path, id + '.pdb')
                pdb_str = _open_pdb(pdb_file)
                pdbs.append(pdb_str)
                atom_coordinates_matrices.append(np.array(_get_atom_coordinates_from_pdb(pdb_str, atom_type), dtype=object))
                progress.update(1)
    else:
        pdbs = _predict_structures(data)
        pdb_names = [str(id) for id in ids]

        with tqdm(range(len(pdbs)), total=len(pdbs), desc="Saving pdb files", disable=False) as progress:
            for (pdb_name, pdb_str) in zip(pdb_names, pdbs):
                _save_pdb(pdb_str, pdb_name, path)
                atom_coordinates_matrices.append(np.array(_get_atom_coordinates_from_pdb(pdb_str, atom_type), dtype=object))
                progress.update(1)

    return atom_coordinates_matrices


def _predict_structures(data):
    hub.set_dir(os.getcwd() + os.sep + "models/esmfold/")
    model = esm.pretrained.esmfold_v1()
    model = model.eval().cuda()

    sequences = data.sequence
    with tqdm(range(len(sequences)), total=len(sequences), desc="Generating 3D structure") as progress_bar:
        pdbs = []
        for i, sequence in enumerate(sequences):
            pdb_str = _predict(model, sequence)
            pdbs.append(pdb_str)
            progress_bar.update(1)

    return pdbs

def _predict(model, sequence):
    with torch.no_grad():
        pdb_str = model.infer_pdb(sequence)
    return pdb_str


def _get_atom_coordinates_from_pdb(pdb_str, atom_type='CA'):
    try:
        pdb_filehandle = io.StringIO(pdb_str)
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("pdb", pdb_filehandle)

        atom_coordinates = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.has_id(atom_type):
                        atom = residue[atom_type]
                        atom_coordinates.append(np.float64(atom.coord))

        pdb_filehandle.close()
        return atom_coordinates

    except Exception as e:
        raise ValueError(f"Error parsing the PDB structure: {e}")


def _get_atom_coordinates_intervals(atom_coordinates_matrices):
    atom_coordinates = np.concatenate(atom_coordinates_matrices, axis=0)
    coordinate_min = np.min(atom_coordinates, axis=0)
    coordinate_max = np.max(atom_coordinates, axis=0)
    return coordinate_min, coordinate_max


def _save_pdb(pdb_str, pdb_name, path):
    if not path.endswith(os.sep):
        path = path + os.sep

    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, pdb_name + ".pdb"), "w") as f:
        f.write(pdb_str)

def _open_pdb(pdb_file):
    with open(pdb_file, "r") as f:
        pdb_str = f.read()
        return pdb_str


def _distance(atom1, atom2, distance_type='euclidean'):
    """
    Calculate the distance between two 3D ºpoints.

    Args:
        atom1 (tuple): The coordinates of the first point (x, y, z).
        atom2 (tuple): The coordinates of the second point (x, y, z).
        distance_type (str): The type of distance to calculate ('euclidean', etc.).

    Returns:
        float: The calculated distance between the two points.
    """
    try:
        return pairwise_distances([atom1], [atom2], metric=distance_type)[0][0]
    except Exception as e:
        raise ValueError("Error calculating distance: " + str(e))


def main(args):
    path = args.tertiary_structure_path
    dataset = args.dataset
    data = pd.read_csv(dataset)
    ids = data.id

    pdbs = _predict_structures(data)
    pdb_names = [str(id) for id in ids]

    with tqdm(range(len(pdbs)), total=len(pdbs), desc ="Saving pdb files", disable=False) as progress:
        for (pdb_name, pdb_str) in zip(pdb_names, pdbs):
            _save_pdb(pdb_str, pdb_name, path)
            progress.update(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset in csv format')
    parser.add_argument('--tertiary_structure_path', type=str, required=True,
                       help='Path to save generated tertiary structures')

    args = parser.parse_args()
    main(args)
