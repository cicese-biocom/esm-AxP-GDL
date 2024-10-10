import pandas as pd
from graph import nodes, edges
from tqdm import tqdm
import numpy as np
from workflow.parameters_setter import ParameterSetter
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
    nodes_features, esm2_contact_maps = nodes.esm2_derived_features(workflow_settings, data)

    # edges
    adjacency_matrices, weights_matrices, data = edges.get_edges(workflow_settings, data, esm2_contact_maps)

    n_samples = len(adjacency_matrices)
    with tqdm(range(n_samples), total=len(adjacency_matrices), desc="Generating graphs", disable=False) as progress:
        graphs = []
        for i in range(n_samples):
            graphs.append(to_parse_matrix(adjacency_matrix=adjacency_matrices[i],
                                          nodes_features=np.array(nodes_features[i], dtype=np.float32),
                                          weights_matrix=weights_matrices[i],
                                          label=data.iloc[i]['activity'] if 'activity' in data.columns else None))
            progress.update(1)

    return graphs, data


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
