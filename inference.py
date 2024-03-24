import numpy as np
import torch
from torch_geometric.loader import DataLoader
import argparse
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef, confusion_matrix
import torch.nn.functional as F
import pandas as pd
from tools.data_preprocessing.dataset_processing import load_and_validate_dataset
from graph.construct_graphs import construct_graphs
import os
from tqdm import tqdm
import time
import random
from models.GAT.GAT import GATModel


def inference(args):
    distance_function = args.distance
    threshold = args.d
    dataset = args.dataset
    esm2_representation = args.esm2_representation
    tertiary_structure_config = (args.tertiary_structure_method, os.path.join(os.getcwd(), args.tertiary_structure_path), args.tertiary_structure_load_pdbs)

    # Load and validation dataset
    inference_data = load_and_validate_dataset(dataset, mode='inference')
    inference_data['sequence_length'] = inference_data['sequence'].str.len()

    # Check if test_data is empty
    if inference_data.empty:
        raise ValueError("No data available for training.")

    validation_config = (None, 1)

    # to get the graph representations
    graphs, _ = construct_graphs(inference_data, esm2_representation, tertiary_structure_config, distance_function, threshold, validation_config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trained_model = os.path.join(os.getcwd(), args.trained_model_path)

    # Load model
    checkpoint = torch.load(trained_model)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    test_dataloader = DataLoader(graphs, batch_size=args.b, shuffle=False)
    y_pred = []
    prob = []

    model.eval()
   # torch.manual_seed(1)
   # random.seed(1)
   # np.random.seed(1)
   # torch.cuda.manual_seed(1)
   # torch.cuda.manual_seed_all(1)
   # torch.backends.cudnn.deterministic = True
   # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    with tqdm(total=len(graphs), desc="Inferring") as progress:
        with torch.no_grad():
            for data in test_dataloader:
                data = data.to(device)

                if args.use_edge_attr:
                    output = model(data.x, data.edge_index, data.edge_attr, data.batch)
                else:
                    output = model(data.x, data.edge_index, None, data.batch)

                out = output[0]

                pred = out.argmax(dim=1)
                score = F.softmax(out, dim=1)[:, 1]

                prob.extend(score.cpu().detach().data.numpy().tolist())
                y_pred.extend(pred.cpu().detach().data.numpy().tolist())

                progress.update(data.num_graphs)

            log_file_in_path = os.path.join(os.path.dirname(trained_model), args.log_file_name + '.txt')
            with open(log_file_in_path, 'a') as f:
                localtime = time.asctime(time.localtime(time.time()))
                f.write(str(localtime) + '\n')
                f.write('args: ' + str(args) + '\n')

            res_data = {'id': inference_data.id, 'sequence': inference_data.sequence, 'sequence_length': inference_data.sequence_length, 'score': prob, 'predicted_activity': y_pred}
            df = pd.DataFrame(res_data)
            output_file = os.path.join(os.path.dirname(trained_model), args.inference_result_file_name + '.csv')
            df.to_csv(output_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--dataset', type=str, required=False, help='Path to the input dataset in csv format')

    # methods for graphs construction
    parser.add_argument('--esm2_representation', type=str, default='esm2_t33',
                        choices=['esm2_t6', 'esm2_t12', 'esm2_t30', 'esm2_t33', 'esm2_t36', 'esm2_t48'],
                        help='ESM-2 model to be used')

    parser.add_argument('--tertiary_structure_method', type=str, default='esmfold',
                        choices=['esmfold'],
                        help='3D structure prediction method')
    parser.add_argument('--tertiary_structure_path', type=str, required=False,
                        help='Path to load or save the generated tertiary structures')
    parser.add_argument('--tertiary_structure_load_pdbs', action="store_true",
                        help="True if specified, otherwise, False. True indicates to load existing tertiary structures from PDB files.")

    # inference parameters
    parser.add_argument('--trained_model_path', type=str, required=False,
                        help='The directory where the trained model to be used for inference is saved')
    parser.add_argument('--b', type=int, default=512, help='Batch size')
    parser.add_argument('--drop', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--hd', type=int, default=128, help='Hidden layer dimension')
    parser.add_argument('--use_edge_attr', action="store_true",
                        help="True if specified, otherwise, False. True indicates to use edge attributes in graph learning.")
    parser.add_argument('--heads', type=int, default=8, help='Number of heads')

    parser.add_argument('--distance', type=str, default='euclidean',
                        choices=['euclidean', 'canberra', 'lance_williams', 'clark', 'soergel', 'bhattacharyya',
                                 'angular_separation'],
                        help='Distance function to construct graph edges')
    parser.add_argument('--d', type=float, default=15, help='Distance threshold to construct graph edges')

    parser.add_argument('--log_file_name', type=str, default='InferenceLog', help='Log file name')
    parser.add_argument('--inference_result_file_name', type=str, default='InferenceResult', help='Results file')

    args = parser.parse_args()

    #args.dataset = os.path.join(os.getcwd(), 'datasets/Yovani-AVP-dataset/debug/28_seq_AVP.csv')
    #args.esm2_representation = 'esm2_t33'
    #args.tertiary_structure_path = os.path.join(os.getcwd(), 'datasets/Yovani-AVP-dataset/debug/ESMFold_pdbs/')
    #args.tertiary_structure_load_pdbs = True
    #args.hd = 128
    #args.trained_model_path = 'output_models/debug_yovani/AMPDiscover_128_euclidean_10_t33_ckpt_200-model2.pt'
    #args.d = 10
    #args.save_ckpt_per_epoch = False
    #args.use_edge_attr = False
    #args.distance = 'euclidean'
    #args.log_file_name = 'InferenceLog_28_iter15'
    #args.inference_result_file_name = 'InferenceResult_28_iter15'

    inference(args)

