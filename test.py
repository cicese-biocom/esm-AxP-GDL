import numpy as np
import torch
from torch_geometric.data import DataLoader
import argparse
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef, confusion_matrix
import torch.nn.functional as F
import pandas as pd
from tools.data_preprocessing.dataset_processing import load_and_validate_dataset
from graph.construct_graphs import construct_graphs
import os

def independent_test(args):

    threshold = args.d
    dataset = args.dataset
    esm2_representation = args.esm2_representation
    tertiary_structure_info = (args.tertiary_structure_method, os.path.join(os.getcwd(), args.tertiary_structure_path))

    # Load and validation data_preprocessing dataset
    data = load_and_validate_dataset(dataset)

    # Filter rows where 'partition' is equal to 3 (test data)
    test_data = data[data['partition'].isin([3])]

    # Check if test_data is empty
    if test_data.empty:
        raise ValueError("No data available for training.")

    # to get the graph representations
    graphs = construct_graphs(test_data, esm2_representation, tertiary_structure_info, threshold,
                              add_self_loop=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torch.load(args.save).to(device)

    test_dataloader = DataLoader(graphs, batch_size=args.b, shuffle=False)
    y_true = []
    y_pred = []
    prob = []

    model.eval()
    with torch.no_grad():
        for data in test_dataloader:
            data = data.to(device)

            output = model(data.x, data.edge_index, data.batch)
            out = output[0]

            pred = out.argmax(dim=1)
            score = F.softmax(out)[:, 1]

            prob.extend(score.cpu().detach().data.numpy().tolist())
            y_true.extend(data.y.cpu().detach().data.numpy().tolist())
            y_pred.extend(pred.cpu().detach().data.numpy().tolist())

        auc = roc_auc_score(y_true, prob)
        acc = accuracy_score(y_true, y_pred)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        f1 = f1_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        sn = tp / (tp + fn)
        sp = tn / (tn + fp)

        print("Test AUC: ", auc)
        print("ACC", acc)
        print("f1", f1)
        print("MCC", mcc)
        print("sn", sn)
        print("sp", sp)

        if args.o is not None:
            res_data = {'AMP_label': y_true, 'score': prob, 'pred': y_pred}
            df = pd.DataFrame(res_data)
            df.to_csv(args.o, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input files
    parser.add_argument('-dataset', type=str, default='datasets/DeepAVPpred/DeepAVPpred.csv',
                        help='Path to the dataset in csv format')

    # methods for graphs construction
    parser.add_argument('-esm2_representation', type=str, default='esm2_t6',
                        help='Representation derived from ESM models to be used')
    parser.add_argument('-tertiary_structure_method', type=str, default='esmfold',
                        help='Method of generation of 3D structures to be used')
    parser.add_argument('-tertiary_structure_path', type=str, default='datasets/DeepAVPpred/trRosetta_output/npz/',
                        help='Path of the tertiary structures generated with the method specified in the parameter tertiary_structure_method')

    # test parameters
    parser.add_argument('-b', type=int, default=512, help='Batch size')
    parser.add_argument('-save', type=str, default='saved_models/samp.model',
                        help='The directory saving the trained models')
    parser.add_argument('-o', type=str, default='test_results.csv', help='Results file')
    parser.add_argument('-drop', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('-hd', type=int, default=64, help='Hidden layer dim')
    parser.add_argument('-heads', type=int, default=8, help='Number of heads')
    parser.add_argument('-d', type=int, default=37, help='Distance threshold')
    args = parser.parse_args()

    independent_test(args)
