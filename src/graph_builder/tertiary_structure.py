import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import io
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings
warnings.simplefilter('ignore', PDBConstructionWarning)

from src.models.esmfold import predict_structures
from src.utils.base_parameters import BaseParameters
from src.utils.pdb import save_pdb, open_pdb


class Predict3DStructuresParameters(BaseParameters):
    pdb_path: Path
    amino_acid_representation: str
    data: pd.DataFrame


class Load3DStructuresParameters(BaseParameters):
    pdb_path: Path
    amino_acid_representation: str
    non_pdb_bound_sequences_file: Path
    data: pd.DataFrame


def load_tertiary_structures(load_3d_structures_parameters: Load3DStructuresParameters):
    if load_3d_structures_parameters.pdb_path is None:
        return [None] * len(load_3d_structures_parameters.data)

    sequences_to_exclude = pd.DataFrame()
    atom_coordinates_matrices = []
    with tqdm(range(len(load_3d_structures_parameters.data)), total=len(load_3d_structures_parameters.data), desc="Loading pdb files", disable=False) as progress:
        pdbs = []
        for index, row in load_3d_structures_parameters.data.iterrows():
            pdb_file = load_3d_structures_parameters.pdb_path.joinpath(f"{row['id']}.pdb")
            try:
                pdb_str = open_pdb(pdb_file)
                pdbs.append(pdb_str)
                coordinates_matrix = np.array(
                    get_atom_coordinates_from_pdb(
                        pdb_str,
                        load_3d_structures_parameters.amino_acid_representation),
                    dtype='float64'
                )
                coordinates_matrix = np.array(_translate_positive_coordinates(coordinates_matrix), dtype='float64')
                atom_coordinates_matrices.append(coordinates_matrix)
                progress.update(1)
            except Exception as e:
                sequences_to_exclude = sequences_to_exclude.append(row)

        if not sequences_to_exclude.empty:
            sequences_to_exclude.to_csv(
                load_3d_structures_parameters.non_pdb_bound_sequences_file,
                index=False
            )

            try:
                raise FileNotFoundError(
                    f"Sequences not linked to PDB or with error when analyzing the PDB structure. "
                    f"See: {load_3d_structures_parameters.non_pdb_bound_sequences_file}"
                )
            except FileNotFoundError as e:
                logging.getLogger("workflow_logger").critical(str(e), exc_info=e)
                raise

        return atom_coordinates_matrices


def predict_tertiary_structures(predict_3d_structures_parameters: Predict3DStructuresParameters):
    pdbs = predict_structures(predict_3d_structures_parameters.data)
    pdb_names = [str(row.id) for (index, row) in predict_3d_structures_parameters.data.iterrows()]
    atom_coordinates_matrices = []
    with tqdm(range(len(pdbs)), total=len(pdbs), desc="Saving pdb files", disable=False) as progress:
        for (pdb_name, pdb_str) in zip(pdb_names, pdbs):
            save_pdb(pdb_str, pdb_name, predict_3d_structures_parameters.pdb_path)
            coordinates_matrix = \
                np.array(get_atom_coordinates_from_pdb(pdb_str, predict_3d_structures_parameters.amino_acid_representation),
                         dtype='float64')
            coordinates_matrix = np.array(_translate_positive_coordinates(coordinates_matrix), dtype='float64')
            atom_coordinates_matrices.append(coordinates_matrix)
            progress.update(1)
    logging.getLogger('workflow_logger'). \
        info(f"Predicted tertiary structures available in: {predict_3d_structures_parameters.pdb_path}")
    return atom_coordinates_matrices


def get_atom_coordinates_from_pdb(pdb_str, atom_type: str):
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


def _get_random_coordinates(atom_coordinates, coordinate_min, coordinate_max):
    random_atom_coordinates = np.zeros(atom_coordinates.shape)
    random_atom_coordinates[:, 0] = \
        np.random.uniform(coordinate_min[0], coordinate_max[0], size=atom_coordinates.shape[0])
    random_atom_coordinates[:, 1] = \
        np.random.uniform(coordinate_min[1], coordinate_max[1], size=atom_coordinates.shape[0])
    random_atom_coordinates[:, 2] = \
        np.random.uniform(coordinate_min[2], coordinate_max[2], size=atom_coordinates.shape[0])

    return random_atom_coordinates


def _translate_positive_coordinates(coordinates):
    min_x = min(min(coordinate[0] for coordinate in coordinates), 0)
    min_y = min(min(coordinate[1] for coordinate in coordinates), 0)
    min_z = min(min(coordinate[2] for coordinate in coordinates), 0)

    eps = 1e-6
    return [np.float64((coordinate[0] - min_x + eps, coordinate[1] - min_y + eps, coordinate[2] - min_z + eps)) for coordinate in coordinates]