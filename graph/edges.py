from models.esmfold import esmfold


def get_adjacency_and_weights_matrices(data, tertiary_structure_config, distance_function, threshold, validation_config):
    method, path, load_pdb = tertiary_structure_config

    if method == 'esmfold':
        return esmfold.get_adjacency_and_weights_matrices(load_pdb, data, path, distance_function, threshold, validation_config)
    else:
        raise ValueError("Invalid 3D structure generation method")