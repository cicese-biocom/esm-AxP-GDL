import torch
from torch_geometric.data import Data, DataLoader
from graph.residues_level_features_encoding import esm2_derived_features
from graph.structure_feature_extraction import contact_map


def construct_graphs(data, esm2_representation, tertiary_structure_info, threshold, add_self_loop=True):
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

    # compute amino acid level feature (esm2 embeddings)
    Xs = esm2_derived_features(data, esm2_representation)

    # load contact map
#    As, Es = contact_map(npz_folder, ids, structural3d_method, threshold, add_self_loop)
    As, Es = contact_map(data, tertiary_structure_info, threshold, add_self_loop)

    graph_representations = []

    labels = data.activity
    n_samples = len(As)
    for i in range(n_samples):
        graph_representations.append(to_parse_matrix(As[i], Xs[i], Es[i], labels[i]))

    return graph_representations


def to_parse_matrix(A, X, E, Y, eps=1e-6):
    """
    :param A: Adjacency matrix with shape (n_nodes, n_nodes)
    :param E: Edge matrix with shape (n_nodes, n_nodes, n_edge_features)
    :param X: node embedding with shape (n_nodes, n_node_features)
    :param eps: default eps=1e-6
    :return:
    """
    num_row, num_col = A.shape
    rows = []
    cols = []
    e_vec = []

    for i in range(num_row):
        for j in range(num_col):
            if A[i][j] >= eps:
                rows.append(i)
                cols.append(j)
                e_vec.append(E[i][j])
    edge_index = torch.tensor([rows, cols], dtype=torch.int64)
    x = torch.tensor(X, dtype=torch.float32)
    edge_attr = torch.tensor(e_vec, dtype=torch.float32)
    y = torch.tensor([Y], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


