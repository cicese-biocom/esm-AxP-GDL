import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from models.esmfold import esmfold_model_handler as esmfold
from workflow.parameters_setter import ParameterSetter
from utils import pdb_parser, distances


def load_tertiary_structures(workflow_settings: ParameterSetter, data: pd.DataFrame):
    if workflow_settings.pdb_path is None:
        return [None] * len(data), data

    sequences_to_exclude = pd.DataFrame()
    atom_coordinates_matrices = []
    with tqdm(range(len(data)), total=len(data), desc="Loading pdb files", disable=False) as progress:
        pdbs = []
        for index, row in data.iterrows():
            pdb_file = workflow_settings.pdb_path.joinpath(f"{row['id']}.pdb")
            try:
                pdb_str = pdb_parser.open_pdb(pdb_file)
                pdbs.append(pdb_str)
                coordinates_matrix = \
                    np.array(pdb_parser.get_atom_coordinates_from_pdb(pdb_str,
                                                                      workflow_settings.amino_acid_representation),
                             dtype='float64')
                coordinates_matrix = np.array(distances.translate_positive_coordinates(coordinates_matrix),
                                              dtype='float64')
                atom_coordinates_matrices.append(coordinates_matrix)
                progress.update(1)
            except Exception as e:
                sequences_to_exclude = sequences_to_exclude.append(row)

        if not sequences_to_exclude.empty:
            csv_file = workflow_settings.output_setting['non_pdb_bound_sequences_file']
            sequences_to_exclude.to_csv(csv_file, index=False)
            data = data.drop(sequences_to_exclude.index)

            logging.getLogger('workflow_logger'). \
                critical(f"Sequences not linked to PDB or with error when analyzing the PDB structure. See: {csv_file}")
            quit()

        if data.empty:
            logging.getLogger('workflow_logger').critical('Dataset with erroneous sequences')
            quit()

        return atom_coordinates_matrices, data


def predict_tertiary_structures(workflow_settings: ParameterSetter, data: pd.DataFrame):
    pdbs = esmfold.predict_structures(data)
    pdb_names = [str(row.id) for (index, row) in data.iterrows()]
    atom_coordinates_matrices = []
    with tqdm(range(len(pdbs)), total=len(pdbs), desc="Saving pdb files", disable=False) as progress:
        for (pdb_name, pdb_str) in zip(pdb_names, pdbs):
            pdb_parser.save_pdb(pdb_str, pdb_name, workflow_settings.pdb_path)
            coordinates_matrix = \
                np.array(pdb_parser.get_atom_coordinates_from_pdb(pdb_str, workflow_settings.amino_acid_representation),
                         dtype='float64')
            coordinates_matrix = np.array(distances.translate_positive_coordinates(coordinates_matrix), dtype='float64')
            atom_coordinates_matrices.append(coordinates_matrix)
            progress.update(1)
    logging.getLogger('workflow_logger'). \
        info(f"Predicted tertiary structures available in: {workflow_settings.pdb_path}")
    return atom_coordinates_matrices


