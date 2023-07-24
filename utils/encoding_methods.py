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

        # representations is a dictionary object with a single key, whose key name varies depending on the layer number
        # of the esm2 model used.representations is a dictionary object with a single key, whose key name varies
        # depending on the layer number of the esm2 model used.
        representations = emb_pt["representations"]
        key = list(representations.keys())[0]
        tensor = representations[key]

        res.append(tensor.numpy())

    return res


def onehot_encoding(seqs):
    residues = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K',
                'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

    encoding_map = np.eye(len(residues))

    residues_map = {}
    for i, r in enumerate(residues):
        residues_map[r] = encoding_map[i]

    res_seqs = []

    for seq in seqs:
        tmp_seq = [residues_map[r] for r in seq]
        res_seqs.append(np.array(tmp_seq))

    return res_seqs


def position_encoding(seqs):
    """
    Position encoding features introduced in "Attention is all your need",
    the b is changed to 1000 for the short length of peptides.
    """
    d = 20
    b = 1000
    res = []
    for seq in seqs:
        N = len(seq)
        value = []
        for pos in range(N):
            tmp = []
            for i in range(d // 2):
                tmp.append(pos / (b ** (2 * i / d)))
            value.append(tmp)
        value = np.array(value)
        pos_encoding = np.zeros((N, d))
        pos_encoding[:, 0::2] = np.sin(value[:, :])
        pos_encoding[:, 1::2] = np.cos(value[:, :])
        res.append(pos_encoding)
    return res


def add(e1, e2):
    res = []
    for i in range(len(e1)):
        res.append(e1[i] + e2[i])
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


if __name__ == '__main__':
    position_encoding(['ARFGD', 'AAAAAA'])
