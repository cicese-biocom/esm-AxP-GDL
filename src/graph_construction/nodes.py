import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from typing import List, Optional
from pydantic.v1 import PositiveFloat

from src.config.types import ExecutionMode, ValidationMode, ESM2Representation
from src.models.esm2 import get_models, get_representations
from src.utils.dto import DTO


class RandomEmbeddingDTO(DTO):
    execution_mode: ExecutionMode
    validation_mode: Optional[ValidationMode]
    randomness_percentage: Optional[PositiveFloat]
    embeddings: List
    data: pd.DataFrame


class ESM2DerivedFeaturesDTO(DTO):
    execution_mode: ExecutionMode
    esm2_representation: ESM2Representation
    data: pd.DataFrame
    device: torch.device


def esm2_derived_features(esm2_derived_features_dto: ESM2DerivedFeaturesDTO):
    models = get_models(
        esm2_derived_features_dto.esm2_representation
    )

    node_features = []
    edge_features = []
    perplexities = None
    if not models.empty:
        for model_info in models:
            model_name = model_info["model"]
            reduced_features = model_info["reduced_features"]
            reduced_features = np.array([x - 1 for x in reduced_features])

            embeddings, contact_maps, perplexities = get_representations(
                data=esm2_derived_features_dto.data,
                model_name=model_name,
                device=esm2_derived_features_dto.device
            )

            # Apply feature reduction (optional)
            embeddings = _apply_feature_reduction(embeddings, reduced_features)

            # Apply embeddings perturbation (optional)
            embeddings = _apply_random_embeddings(
                RandomEmbeddingDTO(**esm2_derived_features_dto.dict(), embeddings=embeddings)
            )

            if not node_features:
                # Initialize node and edge features with the first model's output
                node_features = embeddings.copy()
                edge_features = contact_maps.copy()
            else:
                # When using more than one ESM-2 model:
                #   1) Concatenate the embedding vectors
                node_features = _cat(node_features, embeddings)
                #   2) Averaging the contact maps to get an average contact map
                edge_features = _avg(edge_features, contact_maps)
    return node_features, edge_features, perplexities


def _apply_feature_reduction(embeddings, reduced_features):

    if len(reduced_features) > 0:
        embeddings = [embedding[:, reduced_features] for embedding in embeddings]

    return embeddings


def _get_range_for_embeddings(data_tuples):
    atom_coordinates = np.concatenate(data_tuples, axis=0)
    coordinate_min = np.min(np.min(atom_coordinates, axis=0))
    coordinate_max = np.max(np.max(atom_coordinates, axis=0))
    return coordinate_min, coordinate_max


def _apply_random_embeddings(random_embedding_dto: RandomEmbeddingDTO):
    if random_embedding_dto.validation_mode == ValidationMode.RANDOM_EMBEDDINGS and random_embedding_dto.execution_mode == ExecutionMode.TRAIN:
        partitions = random_embedding_dto.data['partition']
        min_val, max_val = _get_range_for_embeddings(random_embedding_dto.embeddings)

        with tqdm(range(len(random_embedding_dto.embeddings)), total=len(random_embedding_dto.embeddings),
                  desc="Random embeddings ", disable=False) as progress:
            for i, embedding in enumerate(random_embedding_dto.embeddings):
                # only the embeddings belonging to the training set will be randomly created
                # https://dl.acm.org/doi/10.1145/3446776
                if partitions[i] == 1:
                    random_embedding_dto.embeddings[i] = random_node_features(
                        embedding,
                        random_embedding_dto.randomness_percentage,
                        min_val,
                        max_val
                    )
                progress.update(1)
    return random_embedding_dto.embeddings


def random_node_features(feature_matrix, randomness_percentage, min, max):
    total_coefficients = feature_matrix.shape[1]
    total_coefficients_to_build = int(total_coefficients * (randomness_percentage / 100))

    idxs_to_replace = np.random.choice(total_coefficients, total_coefficients_to_build, replace=False)
    for feature_vector in feature_matrix:
        vals_to_replace = np.random.uniform(min, max, size=total_coefficients_to_build)
        feature_vector[idxs_to_replace] = vals_to_replace[range(total_coefficients_to_build)]

    return feature_matrix



def _cat(*args):
    """
    :param args: embeddings
    """
    res = args[0]
    for matrix in args[1:]:
        for i in range(len(matrix)):
            res[i] = np.hstack((res[i], matrix[i]))
    return res


def _avg(*args):
    """
    :param args: contact maps
    """
    return np.mean(np.array(args, dtype=object), axis=0)
