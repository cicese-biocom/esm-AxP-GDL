import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from torch_geometric.data import Data

from pathlib import Path
from typing import List, Optional, Dict
from pydantic.v1 import PositiveFloat

from src.graph_construction.edges import build_edges, BuildEdgesParameters
from src.graph_construction.nodes import compute_esm2_features, ESM2FeatureComputationParameters
from src.models.esm2 import get_models, extract_esm2_representations
from src.config.types import (
    ValidationMode,
    ExecutionMode,
    ESM2ModelForContactMap,
    ESM2Representation,
    EdgeConstructionFunction,
    DistanceFunction
)
from src.utils.base_parameters import BaseParameters


class BuildGraphsParameters(BaseParameters):
    esm2_model_for_contact_map: Optional[ESM2ModelForContactMap]
    esm2_representation: ESM2Representation
    execution_mode: ExecutionMode
    validation_mode: Optional[ValidationMode]
    randomness_percentage: Optional[PositiveFloat]
    tertiary_structure_method: Optional[str]
    pdb_path: Optional[Path]
    amino_acid_representation: str
    non_pdb_bound_sequences_file: Path
    edge_construction_functions: List[EdgeConstructionFunction]
    distance_function: DistanceFunction
    distance_threshold: Optional[PositiveFloat]
    probability_threshold: Optional[PositiveFloat]
    use_edge_attr: Optional[bool]
    data: pd.DataFrame
    device: torch.device
    use_edge_attr: bool


def build_graphs(build_graphs_parameters: BuildGraphsParameters):
    # nodes
    nodes_features, esm2_contact_maps, perplexities_1 = compute_esm2_features(
        ESM2FeatureComputationParameters(
            **build_graphs_parameters.dict()
        )
    )

    # edges
    perplexities_2: pd.DataFrame = pd.DataFrame()

    # If you do not use the edge construction function esm2_contact_map
    if build_graphs_parameters.esm2_model_for_contact_map is None:
        esm2_contact_maps = [None] * len(build_graphs_parameters.data)

    # If the ESM-2 model specified for constructing the graphs and for constructing the edges is different, the following is true
    elif build_graphs_parameters.esm2_model_for_contact_map.value != build_graphs_parameters.esm2_representation.value:
        model = get_models(
            esm2_representation=build_graphs_parameters.esm2_representation
        )

        _, esm2_contact_maps, perplexities_2 = extract_esm2_representations(
            data=build_graphs_parameters.data,
            model_name=model[0]['model'],
            device=build_graphs_parameters.device
        )

    perplexities_output = pd.DataFrame()
    if build_graphs_parameters.esm2_representation == ESM2Representation.ESM2_T36:
        perplexities_output = perplexities_1
    elif build_graphs_parameters.esm2_model_for_contact_map == ESM2Representation.ESM2_T36:
        perplexities_output = perplexities_2

    # If the ESM-2 model specified to build the graphs and to build the edges, the contact maps returned by the
    # function compute_esm2_features are used.
    adjacency_matrices, weights_matrices = build_edges(
        BuildEdgesParameters(
            **build_graphs_parameters.dict(),
            esm2_contact_maps=esm2_contact_maps
        )
    )

    n_samples = len(adjacency_matrices)
    graphs = []

    for i in tqdm(range(n_samples), total=len(adjacency_matrices), desc="Generating graphs"):
        sequence_info = build_graphs_parameters.data.iloc[i]
        graphs.append(
            to_parse_matrix(
                adjacency_matrix=adjacency_matrices[i],
                nodes_features=np.array(nodes_features[i], dtype=np.float32),
                weights_matrix=weights_matrices[i],
                label=sequence_info['activity'] if 'activity' in build_graphs_parameters.data.columns else None,
                sequence_info={
                    "sequence_id": sequence_info['id'],
                    "sequence": sequence_info['sequence'],
                    "sequence_length": sequence_info['length'],
                },
                use_edge_attr=build_graphs_parameters.use_edge_attr
            )
        )

    return graphs, perplexities_output


def to_parse_matrix(adjacency_matrix, nodes_features, weights_matrix, label, sequence_info: Dict, use_edge_attr: bool, eps=1e-6):
    """
    :param label:
    :param sequence_info: Dict
    :param adjacency_matrix: Adjacency matrix with shape (n_nodes, n_nodes)
    :param weights_matrix: Edge matrix with shape (n_nodes, n_nodes, n_edge_features)
    :param nodes_features: node embedding with shape (n_nodes, n_node_features)
    :param use_edge_attr
    :param eps: default eps=1e-6
    :return:
    """

    num_row, num_col = adjacency_matrix.shape
    rows = []
    cols = []
    e_vec = []

    for i in range(num_row):
        for j in range(num_col):
            if adjacency_matrix[i][j] >= eps:
                rows.append(i)
                cols.append(j)
                if weights_matrix.size > 0:
                    e_vec.append(weights_matrix[i][j])
    edge_index = torch.tensor([rows, cols], dtype=torch.int64)
    x = torch.tensor(nodes_features, dtype=torch.float32)
    edge_attr = torch.tensor(np.array(e_vec), dtype=torch.float32) if use_edge_attr else None
    y = torch.tensor([label], dtype=torch.int64) if label is not None else None

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, sequence_info=sequence_info)
    data.validate(raise_on_error=True)
    return data
