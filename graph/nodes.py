import numpy as np
import pandas as pd
from models.esm2 import esm2_model_handler as esm2_model_handler
from workflow.parameters_setter import ParameterSetter


def esm2_derived_features(workflow_settings: ParameterSetter, data: pd.DataFrame):
    """
    esm2_derived_features
    :param workflow_settings:
    :param data: (ids: sequences identifier, sequences: sequences itself)
    :param esm2_representation: name of the esm2 representation to be used
    :return:
        residual_level_features: residual-level features vector
    """

    models = esm2_model_handler.get_models(workflow_settings.esm2_representation)

    residual_level_features = []
    if not models.empty:
        for model_info in models:
            model_name = model_info["model"]
            reduced_features = model_info["reduced_features"]
            reduced_features = [x - 1 for x in reduced_features]

            embeddings, contact_maps = esm2_model_handler.get_embeddings(data,
                                                                         model_name,
                                                                         reduced_features,
                                                                         workflow_settings.validation_mode,
                                                                         workflow_settings.scrambling_percentage,
                                                                         workflow_settings.use_esm2_contact_map)

            if len(residual_level_features) == 0:
                node_features = np.array(embeddings, dtype=object).copy()
            else:
                node_features = cat(residual_level_features, embeddings)

    return node_features, contact_maps


def cat(*args):
    """
    :param args: feature matrices
    """
    res = args[0]
    for matrix in args[1:]:
        for i in range(len(matrix)):
            res[i] = np.hstack((res[i], matrix[i]))
    return res