import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from torch_geometric.data import Data

from pathlib import Path
from typing import List, Optional
from pydantic.v1 import PositiveFloat

from src.graph_construction.edges import get_edges, GetEdgesDTO
from src.graph_construction.nodes import esm2_derived_features, ESM2DerivedFeaturesDTO
from src.models.esm2 import get_models, get_representations
from src.config.types import (
    ValidationMode,
    ExecutionMode,
    ESM2ModelForContactMap,
    ESM2Representation,
    EdgeConstructionFunctions,
    DistanceFunction
)
from src.utils.dto import DTO


class ConstructGraphDTO(DTO):
    esm2_model_for_contact_map: Optional[ESM2ModelForContactMap]
    esm2_representation: ESM2Representation
    execution_mode: ExecutionMode
    validation_mode: Optional[ValidationMode]
    randomness_percentage: Optional[PositiveFloat]
    tertiary_structure_method: Optional[str]
    pdb_path: Optional[Path]
    amino_acid_representation: str
    non_pdb_bound_sequences_file: Path
    edge_construction_functions: List[EdgeConstructionFunctions]
    distance_function: Optional[DistanceFunction]
    distance_threshold: Optional[PositiveFloat]
    probability_threshold: Optional[PositiveFloat]
    use_edge_attr: Optional[bool]
    data: pd.DataFrame
    device: torch.device


def construct_graphs(construct_graph_dto: ConstructGraphDTO):
    # nodes
    nodes_features, esm2_contact_maps, perplexities_1 = esm2_derived_features(
        ESM2DerivedFeaturesDTO(
            **construct_graph_dto.dict()
        )
    )

    # edges
    perplexities_2: pd.DataFrame = pd.DataFrame()

    # If you do not use the edge construction function esm2_contact_map
    if construct_graph_dto.esm2_model_for_contact_map is None:
        esm2_contact_maps = [None] * len(construct_graph_dto.data)

    # If the ESM-2 model specified for constructing the graphs and for constructing the edges is different, the following is true
    elif construct_graph_dto.esm2_model_for_contact_map.value != construct_graph_dto.esm2_representation.value:
        model = get_models(
            esm2_representation=construct_graph_dto.esm2_representation
        )

        _, esm2_contact_maps, perplexities_2 = get_representations(
            data=construct_graph_dto.data,
            model_name=model[0]['model'],
            device=construct_graph_dto.device
        )

    perplexities_output = pd.DataFrame()
    if construct_graph_dto.esm2_representation == ESM2Representation.ESM2_T36:
        perplexities_output = perplexities_1
    elif construct_graph_dto.esm2_model_for_contact_map == ESM2Representation.ESM2_T36:
        perplexities_output = perplexities_2

    # If the ESM-2 model specified to build the graphs and to build the edges is the contact maps returned by the
    # function esm2_derived_features are used.
    adjacency_matrices, weights_matrices = get_edges(
        GetEdgesDTO(
            **construct_graph_dto.dict(),
            esm2_contact_maps=esm2_contact_maps
        )
    )

    n_samples = len(adjacency_matrices)
    with tqdm(range(n_samples), total=len(adjacency_matrices), desc="Generating graphs", disable=False) as progress:
        graphs = []
        for i in range(n_samples):
            graphs.append(
                to_parse_matrix(
                    adjacency_matrix=adjacency_matrices[i],
                    nodes_features=np.array(nodes_features[i], dtype=np.float32),
                    weights_matrix=weights_matrices[i],
                    label=construct_graph_dto.data.iloc[i]['activity'] if 'activity' in construct_graph_dto.data.columns else None
                )
            )
            progress.update(1)

    return graphs, perplexities_output


def to_parse_matrix(adjacency_matrix, nodes_features, weights_matrix, label, eps=1e-6):
    """
    :param label: label
    :param adjacency_matrix: Adjacency matrix with shape (n_nodes, n_nodes)
    :param weights_matrix: Edge matrix with shape (n_nodes, n_nodes, n_edge_features)
    :param nodes_features: node embedding with shape (n_nodes, n_node_features)
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
    edge_attr = torch.tensor(np.array(e_vec), dtype=torch.float32)
    y = torch.tensor([label], dtype=torch.int64) if label is not None else None

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.validate(raise_on_error=True)
    return data
