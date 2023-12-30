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


def _predict(model, sequence):
    with torch.no_grad():
        pdb_str = model.infer_pdb(sequence)

    return pdb_str


def _atom_coordinates(pdb_str, atom_type='CA'):
    pdb_filehandle = io.StringIO(pdb_str)
    parser = PDBParser()
    structure = parser.get_structure("pdb", pdb_filehandle)

    # Create a list to store the coordinates of Cβ atoms
    atom_coordinates = []

    # Iterate through the PDB structure and extract the coordinates of Cβ atoms
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.has_id(atom_type):
                    atom = residue[atom_type]
                    atom_coordinates.append(atom.coord)

    pdb_filehandle.close()
    return atom_coordinates


def _adjacency_matrix(args):
    pdb_str , threshold, distance_type, atom_type, validation_config = args

    atom_coordinates = np.array(_atom_coordinates(pdb_str, atom_type), dtype=object)

    amino_acid_number = len(atom_coordinates)

    validation_mode, scrambling_percentage = validation_config
    if validation_mode == 'coordinates_scrambling':
        amino_acid_number_to_shuffle = max(int(amino_acid_number * scrambling_percentage), 2)
        indexes = np.random.choice(amino_acid_number, size=amino_acid_number_to_shuffle, replace=False)
        atom_coordinates_percent = atom_coordinates[indexes].copy()
        np.random.shuffle(atom_coordinates_percent)
        atom_coordinates[indexes] = atom_coordinates_percent

    A = np.zeros((amino_acid_number, amino_acid_number), dtype=np.int)
    edges = np.zeros((amino_acid_number, amino_acid_number), dtype=np.float64)

    if validation_mode == 'sequence_graph':
        for i in range(amino_acid_number-1):
            dist = _distance(atom_coordinates[i], atom_coordinates[i+1], distance_type)
            A[i][i+1] = 1
            A[i+1][i] = 1
            edges[i][i+1] = dist
            edges[i+1][i] = dist
    else:
        for i in range(amino_acid_number):
            for j in range(i + 1, amino_acid_number):
                dist = _distance(atom_coordinates[i], atom_coordinates[j], distance_type)

                if dist <= threshold:
                    A[i][j] = 1
                    A[j][i] = 1
                    edges[i][j] = dist
                    edges[j][i] = dist

    A[np.eye(A.shape[0]) == 1] = 0

    edges = np.expand_dims(edges, -1)

    return A, edges


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


def adjacency_matrices(data, path, threshold, validation_config):
    hub.set_dir(os.getcwd() + os.sep + "models/esmfold/")

    model = esm.pretrained.esmfold_v1()
    model = model.eval().cuda()

    sequences = data.sequence
    ids = data.id

    with tqdm(range(len(sequences)), total=len(sequences), desc ="Generating 3D structure") as progress_bar:
        pdbs = []
        for i, sequence in enumerate(sequences):
            pdb_str = _predict(model, sequence)
            pdbs.append(pdb_str)
            progress_bar.update(1)

    pdb_names = [str(id) for id in ids]

    with tqdm(range(len(pdbs)), total=len(pdbs), desc ="Saving pdb files", disable=False) as progress:
        for (pdb_name, pdb_str) in zip(pdb_names, pdbs):
            _save_pdb(pdb_str, pdb_name, path)
            progress.update(1)

    # adjacency matrix
    num_cores = multiprocessing.cpu_count()
    distance_type = 'euclidean'
    atom_type = 'CA'

    args = [(pdb, threshold, distance_type, atom_type, validation_config) for pdb in pdbs]

    with ProcessPoolExecutor(max_workers=num_cores) as pool:
        with tqdm(range(len(pdbs)), total=len(pdbs), desc ="Generating adjacency matrices", disable=False) as progress:
            futures = []
            for arg in args:
                future = pool.submit(_adjacency_matrix, arg)
                future.add_done_callback(lambda p: progress.update())
                futures.append(future)

            list_A = [future.result()[0] for future in futures]
            list_E = [future.result()[1] for future in futures]

    return list_A, list_E


def pdb_adjacency_matrices(data, path, threshold, validation_config):
    #pdb_files = glob.glob(path + "*.pdb", recursive=True)

    ids = data['id']

    #pdb_files_to_load = [f for f in pdb_files if os.path.basename(f) in ]

    #pdb_files = sorted(pdb_files, key=lambda name: int(os.path.basename(name).split("AVP")[1].split("_")[0]))

    # Load pdbs
    with tqdm(range(len(ids)), total=len(ids), desc ="Loading pdb files", disable=False) as progress:
        pdbs_str = []
        for id in ids:
            pdb_file = os.path.join(path, id + '.pdb')
            pdb_str = _open_pdb(pdb_file)
            pdbs_str.append(pdb_str)
            progress.update(1)

    # adjacency matrix
    distance_type = 'euclidean'
    atom_type = 'CA'

    num_cores = multiprocessing.cpu_count()
    args = [(pdb_str, threshold, distance_type, atom_type, validation_config) for pdb_str in pdbs_str]

    with ProcessPoolExecutor(max_workers=num_cores) as pool:
        with tqdm(range(len(pdbs_str)), total=len(pdbs_str), desc ="Generating adjacency matrices") as progress:
            futures = []

            for arg in args:
                future = pool.submit(_adjacency_matrix, arg)
                future.add_done_callback(lambda p: progress.update())
                futures.append(future)

            list_A = [future.result()[0] for future in futures]
            list_E = [future.result()[1] for future in futures]

    return list_A, list_E


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

    hub.set_dir(os.getcwd() + os.sep + "models/esmfold/")

    model = esm.pretrained.esmfold_v1()
    model = model.eval().cuda()

    sequences = data.sequence
    ids = data.id

    with tqdm(range(len(sequences)), total=len(sequences), desc ="Generating 3D structure") as progress_bar:
        pdbs = []
        for i, sequence in enumerate(sequences):
            pdb_str = _predict(model, sequence)
            pdbs.append(pdb_str)

            progress_bar.update(1)

    #pdb_names = [str(id) + '_Pos' if label == 1 else str(id) + '_Neg' for id, label in zip(ids, labels)]
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
