import numpy as np
from models.esmfold import esmfold


def adjacency_matrix(data, tertiary_structure_config, threshold, add_self_loop):
    method, path, mode = tertiary_structure_config

    if method == 'trRosetta':
        if mode == 'load':
            return cmap_tr(path, data.id, threshold, add_self_loop)
        else:
            raise ValueError("Invalid mode. Please choose 'load'.")

    if method == 'esmfold':
        if mode == 'generate':
            return esmfold.adjacency_matrices(data, path, threshold, add_self_loop)
        elif mode == 'load':
            return esmfold.pdb_adjacency_matrices(data, path, threshold, add_self_loop)
        else:
            raise ValueError("Invalid mode. Please choose 'generate' or 'load'.")

def cmap_tr(npz_folder, ids, threshold, add_self_loop=True):
    if npz_folder[-1] != '/':
        npz_folder += '/'

    list_A = []
    list_E = []

    for id in ids:
        npz = id[0:] + '.npz'
        f = np.load(npz_folder + npz)

        mat_dist = f['dist']
        mat_omega = f['omega']
        mat_theta = f['theta']
        mat_phi = f['phi']

        """ 
        The distance range (2 to 20 Å) is binned into 36 equally spaced segments, 0.5 Å each, 
        plus one bin indicating that residues are not in contact.
            - Improved protein structure prediction using predicted interresidue orientations: 
        """
        dist = np.argmax(mat_dist, axis=2)  # 37 equally spaced segments
        omega = np.argmax(mat_omega, axis=2)
        theta = np.argmax(mat_theta, axis=2)
        phi = np.argmax(mat_phi, axis=2)

        A = np.zeros(dist.shape, dtype=np.int)

        A[dist < threshold] = 1
        A[dist == 0] = 0
        # A[omega < threshold] = 1
        if add_self_loop:
            A[np.eye(A.shape[0]) == 1] = 1
        else:
            A[np.eye(A.shape[0]) == 1] = 0

        dist[A == 0] = 0
        omega[A == 0] = 0
        theta[A == 0] = 0
        phi[A == 0] = 0

        dist = np.expand_dims(dist, -1)
        omega = np.expand_dims(omega, -1)
        theta = np.expand_dims(theta, -1)
        phi = np.expand_dims(phi, -1)

        edges = dist
        edges = np.concatenate((edges, omega), axis=-1)
        edges = np.concatenate((edges, theta), axis=-1)
        edges = np.concatenate((edges, phi), axis=-1)

        list_A.append(A)
        list_E.append(edges)

    return list_A, list_E