import numpy as np
from models.esm2.esm2_representation import *


def esm2_derived_features(data, esm2_representation, normalize_embedding):
    """
    :param data: (ids: sequences identifier, sequences: sequences itself)
    :param esm2_representation: name of the esm2 representation to be used
    :return:
        residual_level_features: residual-level features vector
    """

    models = get_models(esm2_representation)

    residual_level_features = []
    if not models.empty:
        for model_info in models:
            model_name = model_info["model"]
            reduced_features = model_info["reduced_features"]
            reduced_features = [x - 1 for x in reduced_features]

            embeddings = get_embeddings(data, model_name, reduced_features, normalize_embedding)

            if len(residual_level_features) == 0:
                residual_level_features = np.array(embeddings, dtype=object).copy()
            else:
                residual_level_features = cat(residual_level_features, embeddings)

    return residual_level_features


def cat(*args):
    """
    :param args: feature matrices
    """
    res = args[0]
    for matrix in args[1:]:
        for i in range(len(matrix)):
            res[i] = np.hstack((res[i], matrix[i]))
    return res