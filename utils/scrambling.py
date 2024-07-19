import numpy as np


def scrambling_matrix_rows(matrix, scrambling_percentage):
    num_rows = matrix.shape[0]

    num_to_scramble = int(num_rows * (scrambling_percentage / 100))
    index = np.random.choice(num_rows, num_to_scramble, replace=False)
    index_to_scramble = np.random.permutation(index)
    scrambled_matrix = matrix.copy()
    scrambled_matrix[index] = matrix[index_to_scramble]

    return scrambled_matrix


def random_coordinate_matrix(matrix, min, max):
    atom_coordinates = np.zeros(matrix.shape)
    atom_coordinates[:, 0] = np.random.uniform(min[0], max[0], size=matrix.shape[0])
    atom_coordinates[:, 1] = np.random.uniform(min[1], max[1], size=matrix.shape[0])
    atom_coordinates[:, 2] = np.random.uniform(min[2], max[2], size=matrix.shape[0])
    return atom_coordinates
