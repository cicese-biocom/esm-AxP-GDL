import torch
from torch import hub
import esm
import os
import argparse
from Bio.PDB import PDBParser
import io
import numpy as np
from sklearn.metrics import pairwise_distances
from tools.data_preprocessing.dataset_processing import load_and_validate_dataset
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.simplefilter('ignore', PDBConstructionWarning)
import datetime
import glob
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


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

    return atom_coordinates


def _adjacency_matrix(args):
    pdb_str , threshold, add_self_loop, distance_type, atom_type = args

    atom_coordinates = _atom_coordinates(pdb_str, atom_type)

    num_atoms = len(atom_coordinates)
    A = np.zeros((num_atoms, num_atoms), dtype=np.int)
    edges = np.zeros((num_atoms, num_atoms), dtype=np.float64)

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            dist = _distance(atom_coordinates[i], atom_coordinates[j], distance_type)

            if dist < threshold:
                A[i][j] = 1
                A[j][i] = 1
                edges[i][j] = dist
                edges[j][i] = dist

    if add_self_loop:
        A[np.eye(A.shape[0]) == 1] = 1
    else:
        A[np.eye(A.shape[0]) == 1] = 0

    edges = np.expand_dims(edges, -1)

    return A, edges


def _save_pdb(pdb_str, pdb_name, path):
    # Save the pdb
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, pdb_name + ".pdb"), "w") as f:
        f.write(pdb_str)

def _open_pdb(pdb_file):
    with open(pdb_file, "r") as f:
        pdb_str = f.read()
        return pdb_str


def adjacency_matrices(data, path, threshold, add_self_loop):
    hub.set_dir(os.getcwd() + os.sep + "models/esmfold/")

    model = esm.pretrained.esmfold_v1()
    model = model.eval().cuda()

    sequences = data.sequence
    ids = data.id
    labels = data.activity

    with tqdm(range(len(sequences)), total=len(sequences), desc ="Generating 3D structure", disable=False) as progress_bar:
        pdbs = []
        for i, sequence in enumerate(sequences):
            pdb_str = _predict(model, sequence)
            pdbs.append(pdb_str)

            progress_bar.update(1)

    pdb_names = [str(id) + '_Pos' if label == 1 else str(id) + '_Neg' for id, label in zip(ids, labels)]

    with tqdm(range(len(pdbs)), total=len(pdbs), desc ="saving pdb files", disable=False) as progress:
        for (pdb_name, pdb_str) in zip(pdb_names, pdbs):
            _save_pdb(pdb_str, pdb_name, path)
            progress.update(1)

    # adjacency matrix
    num_cores = multiprocessing.cpu_count()
    distance_type = 'euclidean'
    atom_type = 'CA'

    args = [(pdb, threshold, add_self_loop, distance_type, atom_type) for pdb in pdbs]

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


def pdb_adjacency_matrices(data, path, threshold, add_self_loop):
    pdb_files = glob.glob(path + "*.pdb", recursive=True)
    pdb_files = sorted(pdb_files, key=lambda name: int(os.path.basename(name).split("AVP")[1].split("_")[0]))

    # Load pdbs
    with tqdm(range(len(pdb_files)), total=len(pdb_files), desc ="Loading pdb files", disable=False) as progress:
        pdbs_str = []
        for pdb_file in pdb_files:
            pdb_str = _open_pdb(pdb_file)
            pdbs_str.append(pdb_str)
            progress.update(1)

    # adjacency matrix
    distance_type = 'euclidean'
    atom_type = 'CA'

    num_cores = multiprocessing.cpu_count()
    args = [(pdb_str, threshold, add_self_loop, distance_type, atom_type) for pdb_str in pdbs_str]

    with ProcessPoolExecutor(max_workers=num_cores) as pool:
        with tqdm(range(len(pdbs_str)), total=len(pdbs_str), desc ="Generating adjacency matrices", disable=False) as progress:
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
    print(f"Pending implementation")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('-dataset', type=str, default='datasets/Test_Data/test_data.csv',
                        help='Path to the dataset in csv format')
    parser.add_argument('-contact_map', type=int, default=True,
                        help='Specify whether to include a contact map in the output. By default, a contact map'
                             ' will be included (True), but you can disable it by specifying this flag (False).')

    args = parser.parse_args()
    main(args)
