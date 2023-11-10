from models.esmfold import esmfold


def adjacency_matrix(data, tertiary_structure_config, threshold, add_self_loop):
    method, path, load_pdb = tertiary_structure_config

    if method == 'esmfold':
        if load_pdb:
            return esmfold.pdb_adjacency_matrices(data, path, threshold, add_self_loop)
        else:
            return esmfold.adjacency_matrices(data, path, threshold, add_self_loop)
    else:
        raise ValueError("Invalid 3D structure generation method")