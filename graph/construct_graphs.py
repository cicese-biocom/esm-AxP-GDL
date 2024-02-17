import torch
from torch_geometric.data import Data, DataLoader
from graph.nodes import esm2_derived_features
from graph.edges import get_adjacency_and_weights_matrices
from tqdm import tqdm
import numpy as np

def construct_graphs(data, esm2_representation, tertiary_structure_config, distance_function, threshold, validation_config):
    """
    :param data: data (id, sequence itself, activity, label)
    :param esm2_representation: name of the esm2 representation to be used
    :param tertiary_structure_info: Method of generation of 3D structures to be used and path of the tertiary
                                    structures generated
    :param threshold: threshold for build adjacency matrix
    :param add_self_loop: add_self_loop
    :return:
        graphs_representations: list of Data
        labels: list of labels
        partition: identification of the data partition each instance belongs to
    """

    # nodes 
    nodes_features = esm2_derived_features(data, esm2_representation, validation_config)

    # edges
    adjacency_matrices, weights_matrices, similarity = get_adjacency_and_weights_matrices(data, tertiary_structure_config, distance_function, threshold, validation_config)

    labels = data.activity
    n_samples = len(adjacency_matrices)
    with tqdm(range(n_samples), total=len(adjacency_matrices), desc ="Generating graphs", disable=False) as progress:
        graphs = []
        for i in range(n_samples):
            graphs.append(to_parse_matrix(adjacency_matrices[i], nodes_features[i], weights_matrices[i], labels[i]))
            progress.update(1)

    return graphs, similarity


def to_parse_matrix(adjacency_matrix, nodes_features, weights_matrix, label, eps=1e-6):
    """
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
                e_vec.append(weights_matrix[i][j])
    edge_index = torch.tensor([rows, cols], dtype=torch.int64)
    x = torch.tensor(nodes_features, dtype=torch.float32)
    edge_attr = torch.tensor(np.array(e_vec), dtype=torch.float32)
    y = torch.tensor([label], dtype=torch.int64)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.validate(raise_on_error=True)
    return data


