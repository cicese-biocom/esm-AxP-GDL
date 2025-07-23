import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config.enum import AminoAcidRepresentation
from src.models.esmfold import predict_structures
from src.utils.distance import translate_positive_coordinates
from src.utils.dto import DTO
from src.utils.pdb import get_atom_coordinates_from_pdb, save_pdb, open_pdb


class PredictTertiaryStructuresDTO(DTO):
    pdb_path: Path
    amino_acid_representation: AminoAcidRepresentation
    data: pd.DataFrame


class LoadTertiaryStructuresDTO(DTO):
    pdb_path: Path
    amino_acid_representation: AminoAcidRepresentation
    non_pdb_bound_sequences_file: Path
    data: pd.DataFrame



def load_tertiary_structures(load_tertiary_structures_dto: LoadTertiaryStructuresDTO):
    if load_tertiary_structures_dto.pdb_path is None:
        return [None] * len(load_tertiary_structures_dto.data)

    sequences_to_exclude = pd.DataFrame()
    atom_coordinates_matrices = []
    with tqdm(range(len(load_tertiary_structures_dto.data)), total=len(load_tertiary_structures_dto.data), desc="Loading pdb files", disable=False) as progress:
        pdbs = []
        for index, row in load_tertiary_structures_dto.data.iterrows():
            pdb_file = load_tertiary_structures_dto.pdb_path.joinpath(f"{row['id']}.pdb")
            try:
                pdb_str = open_pdb(pdb_file)
                pdbs.append(pdb_str)
                coordinates_matrix = np.array(
                    get_atom_coordinates_from_pdb(
                        pdb_str,
                        load_tertiary_structures_dto.amino_acid_representation),
                    dtype='float64'
                )
                coordinates_matrix = np.array(translate_positive_coordinates(coordinates_matrix), dtype='float64')
                atom_coordinates_matrices.append(coordinates_matrix)
                progress.update(1)
            except Exception as e:
                sequences_to_exclude = sequences_to_exclude.append(row)

        if not sequences_to_exclude.empty:
            sequences_to_exclude.to_csv(
                load_tertiary_structures_dto.non_pdb_bound_sequences_file,
                index=False
            )

            try:
                raise FileNotFoundError(
                    f"Sequences not linked to PDB or with error when analyzing the PDB structure. "
                    f"See: {load_tertiary_structures_dto.non_pdb_bound_sequences_file}"
                )
            except FileNotFoundError as e:
                logging.getLogger("workflow_logger").critical(str(e), exc_info=e)
                raise

        return atom_coordinates_matrices


def predict_tertiary_structures(predict_tertiary_structures_dto: PredictTertiaryStructuresDTO):
    pdbs = predict_structures(predict_tertiary_structures_dto.data)
    pdb_names = [str(row.id) for (index, row) in predict_tertiary_structures_dto.data.iterrows()]
    atom_coordinates_matrices = []
    with tqdm(range(len(pdbs)), total=len(pdbs), desc="Saving pdb files", disable=False) as progress:
        for (pdb_name, pdb_str) in zip(pdb_names, pdbs):
            save_pdb(pdb_str, pdb_name, predict_tertiary_structures_dto.pdb_path)
            coordinates_matrix = \
                np.array(get_atom_coordinates_from_pdb(pdb_str, predict_tertiary_structures_dto.amino_acid_representation),
                         dtype='float64')
            coordinates_matrix = np.array(translate_positive_coordinates(coordinates_matrix), dtype='float64')
            atom_coordinates_matrices.append(coordinates_matrix)
            progress.update(1)
    logging.getLogger('workflow_logger'). \
        info(f"Predicted tertiary structures available in: {predict_tertiary_structures_dto.pdb_path}")
    return atom_coordinates_matrices


