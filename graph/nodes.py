import numpy as np
import pandas as pd
from tqdm import tqdm
from models.esm2 import esm2_model_handler as esm2_model_handler
from utils.scrambling import random_node_features
from workflow.parameters_setter import ParameterSetter


def esm2_derived_features(workflow_settings: ParameterSetter, data: pd.DataFrame):
    """
    esm2_derived_features
    :param workflow_settings:
    :param data: (ids: sequences identifier, sequences: sequences itself)
    :return:
        residual_level_features: residual-level features vector
    """

    models = esm2_model_handler.get_models(workflow_settings.esm2_representation)

    node_features = []
    if not models.empty:
        for model_info in models:
            model_name = model_info["model"]
            reduced_features = model_info["reduced_features"]
            reduced_features = [x - 1 for x in reduced_features]

            embeddings, contact_maps = esm2_model_handler.get_embeddings(data,
                                                                         model_name,
                                                                         reduced_features,
                                                                         workflow_settings.validation_mode,
                                                                         workflow_settings.randomness_percentage,
                                                                         workflow_settings.use_esm2_contact_map)

            embeddings = _apply_random_embeddings(workflow_settings, embeddings, data)

            if len(node_features) == 0:
                node_features = np.array(embeddings, dtype=object).copy()
            else:
                # when using more than one ESM-2 model:
                #   1) Concatenate the embedding vectors
                #   2) Averaging the contact maps to get am average contact map
                raise Exception("The use of more than one ESM-2 model is not supported yet!")

    return node_features, contact_maps


def _get_range_for_embeddings(data_tuples):
    atom_coordinates = np.concatenate(data_tuples, axis=0)
    coordinate_min = np.min(np.min(atom_coordinates, axis=0))
    coordinate_max = np.max(np.max(atom_coordinates, axis=0))
    return coordinate_min, coordinate_max


def _apply_random_embeddings(workflow_settings, embeddings, data):
    if workflow_settings.validation_mode == 'random_embeddings' and workflow_settings.mode == 'training':
        partitions = data['partition']
        min_val, max_val = _get_range_for_embeddings(embeddings)

        with tqdm(range(len(embeddings)), total=len(embeddings),
                  desc="Random embeddings ", disable=False) as progress:
            for i, embedding in enumerate(embeddings):
                # only the embeddings belonging to the training set will be randomly created
                # https://dl.acm.org/doi/10.1145/3446776
                if partitions[i] == 1:
                    embeddings[i] = random_node_features(embedding, workflow_settings.randomness_percentage,
                                                         min_val, max_val)
                progress.update(1)
    return embeddings


def _cat(*args):
    """
    :param args: feature matrices
    """
    res = args[0]
    for matrix in args[1:]:
        for i in range(len(matrix)):
            res[i] = np.hstack((res[i], matrix[i]))
    return res