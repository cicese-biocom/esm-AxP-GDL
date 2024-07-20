import numpy as np


def scrambling_matrix_rows(matrix, randomness_percentage):
    num_rows = matrix.shape[0]

    num_to_scramble = int(num_rows * (randomness_percentage / 100))
    index = np.random.choice(num_rows, num_to_scramble, replace=False)
    index_to_scramble = np.random.permutation(index)
    rand_matrix = matrix.copy()
    rand_matrix[index] = matrix[index_to_scramble]

    return rand_matrix


def random_coordinate_matrix(matrix, randomness_percentage, min, max):
    num_rows = matrix.shape[0]
    num_to_scramble = int(num_rows * (randomness_percentage / 100))
    indices = np.random.choice(num_rows, num_to_scramble, replace=False)

    atom_coordinates = matrix
    coord_0 = np.random.uniform(min[0], max[0], size=num_to_scramble)
    coord_1 = np.random.uniform(min[1], max[1], size=num_to_scramble)
    coord_2 = np.random.uniform(min[2], max[2], size=num_to_scramble)
    for coord_idx, matrix_idx in enumerate(indices):
        atom_coordinates[matrix_idx][0] = coord_0[coord_idx]
        atom_coordinates[matrix_idx][1] = coord_1[coord_idx]
        atom_coordinates[matrix_idx][2] = coord_2[coord_idx]

    return atom_coordinates
