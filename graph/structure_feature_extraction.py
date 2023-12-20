from models.esmfold import esmfold


def adjacency_matrix(data, tertiary_structure_config, threshold, validation_mode):
    method, path, load_pdb = tertiary_structure_config

    if method == 'esmfold':
        if load_pdb:
            return esmfold.pdb_adjacency_matrices(data, path, threshold, validation_mode)
        else:
            return esmfold.adjacency_matrices(data, path, threshold, validation_mode)
    else:
        raise ValueError("Invalid 3D structure generation method")