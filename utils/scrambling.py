import numpy as np


def scrambling_matrix_rows(matrix, scrambling_percentage):
    num_rows = matrix.shape[0]

    num_to_scramble = int(num_rows * (scrambling_percentage / 100))
    index = np.random.choice(num_rows, num_to_scramble, replace=False)
    index_to_scramble = np.random.permutation(index)
    scrambled_matrix = matrix.copy()
    scrambled_matrix[index] = matrix[index_to_scramble]

    return scrambled_matrix
