import torch
from torch import hub
import esm
import os
import argparse
import pandas as pd
from tqdm import tqdm
from utils import pdb_parser


def predict_structures(data):
    hub.set_dir(os.getcwd() + os.sep + "models/esmfold/")
    model = esm.pretrained.esmfold_v1()
    model = model.eval().cuda()

    sequences = data.sequence
    with tqdm(range(len(sequences)), total=len(sequences), desc="Generating 3D structure") as progress_bar:
        pdbs = []
        for i, sequence in enumerate(sequences):
            pdb_str = _predict(model, sequence)
            pdbs.append(pdb_str)
            progress_bar.update(1)
    return pdbs


def _predict(model, sequence):
    with torch.no_grad():
        pdb_str = model.infer_pdb(sequence)
    return pdb_str


def main(args):
    path = args.tertiary_structure_path
    dataset = args.dataset
    data = pd.read_csv(dataset)
    ids = data.id

    pdbs = predict_structures(data)
    pdb_names = [str(id) for id in ids]

    with tqdm(range(len(pdbs)), total=len(pdbs), desc="Saving pdb files", disable=False) as progress:
        for (pdb_name, pdb_str) in zip(pdb_names, pdbs):
            pdb_parser.save_pdb(pdb_str, pdb_name, path)
            progress.update(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset in csv format')
    parser.add_argument('--tertiary_structure_path', type=str, required=True,
                        help='Path to save generated tertiary structures')

    args = parser.parse_args()
    main(args)
