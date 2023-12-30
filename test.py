import numpy as np
import torch
from torch_geometric.loader import DataLoader
import argparse
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef, confusion_matrix
import torch.nn.functional as F
import pandas as pd
from tools.data_preprocessing.dataset_processing import load_and_validate_dataset
from graph.construct_graphs import construct_graphs
import os
from tqdm import tqdm
import time

def independent_test(args):
    threshold = args.d
    dataset = args.dataset
    esm2_representation = args.esm2_representation
    tertiary_structure_config = (args.tertiary_structure_method, os.path.join(os.getcwd(), args.tertiary_structure_path), args.tertiary_structure_load_pdbs)

    # Load and validation dataset
    data = load_and_validate_dataset(dataset)

    # Filter rows where 'partition' is equal to 3 (test data)
    test_data = data[data['partition'].isin([3])].reset_index(drop=True)

    # Check if test_data is empty
    if test_data.empty:
        raise ValueError("No data available for training.")

    # to get the graph representations
    graphs = construct_graphs(test_data, esm2_representation, tertiary_structure_config, threshold)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trained_model = os.path.join(os.getcwd(), args.trained_model_path)

    checkpoint = torch.load(trained_model)
    model = checkpoint['model']
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    model.to(device)

    test_dataloader = DataLoader(graphs, batch_size=args.b, shuffle=False)
    y_true = []
    y_pred = []
    prob = []

    model.eval()
    with tqdm(total=len(graphs), desc="Testing") as progress:
        with torch.no_grad():
            for data in test_dataloader:
                data = data.to(device)

                output = model(data.x, data.edge_index, data.batch)
                out = output[0]

                pred = out.argmax(dim=1)
                score = F.softmax(out, dim=1)[:, 1]

                prob.extend(score.cpu().detach().data.numpy().tolist())
                y_true.extend(data.y.cpu().detach().data.numpy().tolist())
                y_pred.extend(pred.cpu().detach().data.numpy().tolist())

                progress.update(data.num_graphs)
                progress.set_postfix(
                    Recall_Pos=f"{'.'}",
                    Recall_Neg=f"{'.'}",
                    Test_MCC=f"{'.'}",
                    Test_ACC=f"{'.'}",
                    Test_AUC=f"{'.'}"
                )

            auc = roc_auc_score(y_true, prob)
            acc = accuracy_score(y_true, y_pred)

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            mcc = matthews_corrcoef(y_true, y_pred)
            sn = tp / (tp + fn)
            sp = tn / (tn + fp)

            progress.set_description("Test result")
            progress.set_postfix(
                Recall_Pos=f"{sp:.4f}",
                Recall_Neg=f"{sn:.4f}",
                Test_MCC=f"{mcc:.4f}",
                Test_ACC=f"{acc:.4f}",
                Test_AUC=f"{auc:.4f}"
            )

            log_file_in_path = os.path.join(os.path.dirname(trained_model), args.log_file_name + '.txt')
            with open(log_file_in_path, 'a') as f:
                localtime = time.asctime(time.localtime(time.time()))
                f.write(str(localtime) + '\n')
                f.write('args: ' + str(args) + '\n')
                f.write('Test MCC result: ' + str(mcc) + '\n')
                f.write('Test ACC result: ' + str(acc) + '\n')
                f.write('Test AUC result: ' + str(auc) + '\n')
                f.write('Recall Pos result: ' + str(sp) + '\n')
                f.write('Recall Neg result: ' + str(sn) + '\n')

            res_data = {'AMP_label': y_true, 'score': prob, 'pred': y_pred}
            df = pd.DataFrame(res_data)
            output_file = os.path.join(os.path.dirname(trained_model), args.test_result_file_name + '.csv')
            df.to_csv(output_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--dataset', type=str, required=True, help='Path to the input dataset in csv format')

    # methods for graphs construction
    parser.add_argument('--esm2_representation', type=str, default='esm2_t33',
                        choices=['esm2_t6', 'esm2_t12', 'esm2_t30', 'esm2_t36', 'esm2_t48'],
                        help='ESM-2 model to be used')

    parser.add_argument('--tertiary_structure_method', type=str, default='esmfold',
                        choices=['esmfold'],
                        help='3D structure prediction method')
    parser.add_argument('--tertiary_structure_path', type=str, required=True,
                        help='Path to load or save the generated tertiary structures')
    parser.add_argument('--tertiary_structure_load_pdbs', action="store_true",
                        help="True if specified, otherwise, False. True indicates to load existing tertiary structures from PDB files.")

    # test parameters
    parser.add_argument('--trained_model_path', type=str, required=True,
                        help='The directory where the trained model to be used for inference is saved')
    parser.add_argument('--b', type=int, default=512, help='Batch size')
    parser.add_argument('--drop', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--hd', type=int, default=128, help='Hidden layer dimension')
    parser.add_argument('--heads', type=int, default=8, help='Number of heads')
    parser.add_argument('--d', type=int, default=15, help='Distance threshold to construct graph edges')

    parser.add_argument('--log_file_name', type=str, default='TestLog', help='Log file name')
    parser.add_argument('--test_result_file_name', type=str, default='TestResult', help='Results file')



    args = parser.parse_args()

    independent_test(args)
