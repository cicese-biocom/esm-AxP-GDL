import torch
from torch import hub
import esm
import os
import argparse
from tools.data_preprocessing.dataset_processing import load_and_validate_dataset


def predict(data):
    hub.set_dir(os.getcwd() + os.sep + "models/esmfold/")

    model = esm.pretrained.esmfold_v1()
    model = model.eval().cuda()

    sequences = data.sequence

    for sequence in sequences:
        with torch.no_grad():
            output = model.infer_pdb(sequence)


def main(args):
    dataset = args.dataset

    abs_path = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
    dataset_path = os.path.join(abs_path, dataset)

    # Load and validation data_preprocessing dataset
    data = load_and_validate_dataset(dataset_path)

    hub.set_dir(os.getcwd() + os.sep + "models/esmfold/")

    model = esm.pretrained.esmfold_v1()
    model = model.eval().cuda()

    sequences = data.sequence

    for sequence in sequences:
        with torch.no_grad():
            output = model.infer_pdb(sequence)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('-dataset', type=str, default='datasets/Test_Data/test_data.csv',
                        help='Path to the dataset in csv format')

    args = parser.parse_args()
    main(args)
