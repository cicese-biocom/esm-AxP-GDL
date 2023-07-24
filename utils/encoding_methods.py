import os
import numpy as np
import torch as torch

def esm2_embbeding(ids, esm2_dir):
    """
    parser esm2 features
    """
    if esm2_dir[-1] != '/': esm2_dir += '/'
    esm2_fs = os.listdir(esm2_dir)
    res = []
    for id in ids:
        name = id
        if id[0] == '>': name = id[1:]
        file_pt = name + '.pt'
        assert file_pt in esm2_fs

        #  Get the full path to the .pt file
        file_pt_path = os.path.join(esm2_dir, file_pt)

        # Load the .pt file
        emb_pt = torch.load(file_pt_path)

        assert 'representations' in emb_pt

        """ 
        representations is a dictionary object with a single key, whose key name varies depending on the layer number
        of the esm2 model used.representations is a dictionary object with a single key, whose key name varies
        depending on the layer number of the esm2 model used.
        """
        representations = emb_pt["representations"]
        key = list(representations.keys())[0]
        tensor = representations[key]

        res.append(tensor.numpy())

    return res


def cat(*args):
    """
    :param args: feature matrices
    """
    res = args[0]
    for matrix in args[1:]:
        for i in range(len(matrix)):
            res[i] = np.hstack((res[i], matrix[i]))
    return res