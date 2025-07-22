import pandas as pd
from src_old.graph import nodes, edges
from tqdm import tqdm
import numpy as np
from src_old.models.esm2 import esm2_model_handler
from src_old.workflow.parameters_setter import ParameterSetter
import torch
from torch_geometric.data import Data


def construct_graphs(workflow_settings: ParameterSetter, data: pd.DataFrame):
    """
    construct_graphs
    :param workflow_settings:
    :param data: List (id, sequence itself, activity, label)
    :return:
        graphs_representations: list of Data
        labels: list of labels
        partition: identification of the old_data partition each instance belongs to
    """

    # nodes
    nodes_features, esm2_contact_maps, perplexities_1 = nodes.esm2_derived_features(workflow_settings, data)

    # edges
    # If you do not use the edge construction function esm2_contact_map
    if workflow_settings.esm2_model_for_contact_map is None:
        esm2_contact_maps = [None] * len(data)
    # If the ESM-2 model specified for constructing the graphs and for constructing the edges is different, the following is true
    elif workflow_settings.esm2_model_for_contact_map != workflow_settings.esm2_representation:
        model = esm2_model_handler.get_models(workflow_settings.esm2_model_for_contact_map)
        _, esm2_contact_maps, perplexities_2 = esm2_model_handler.get_representations(data, model[0]['model'])

    perplexities_output = pd.DataFrame()
    if workflow_settings.esm2_representation == 'esm2_t36':
        perplexities_output = perplexities_1
    elif workflow_settings.esm2_model_for_contact_map == 'esm2_t36':
        perplexities_output = perplexities_2

    # If the ESM-2 model specified to build the graphs and to build the edges is the contact maps returned by the
    # function esm2_derived_features are used.
    adjacency_matrices, weights_matrices = edges.get_edges(workflow_settings, data, esm2_contact_maps)

    n_samples = len(adjacency_matrices)
    with tqdm(range(n_samples), total=len(adjacency_matrices), desc="Generating graphs", disable=False) as progress:
        graphs = []
        for i in range(n_samples):
            graphs.append(to_parse_matrix(adjacency_matrix=adjacency_matrices[i],
                                          nodes_features=np.array(nodes_features[i], dtype=np.float32),
                                          weights_matrix=weights_matrices[i],
                                          label=data.iloc[i]['activity'] if 'activity' in data.columns else None))
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
