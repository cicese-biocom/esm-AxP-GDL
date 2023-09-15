import numpy as np
import torch
from torch_geometric.data import DataLoader
import argparse
from models.GAT.GAT import GATModel
from tools.data_preprocessing.dataset_processing import load_and_validate_dataset
from graph.graph_construction import graph_representations
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import time
import datetime
import os

def train(args):
    try:
        # load arguments
        threshold = args.d
        dataset = args.dataset
        esm2_representation = args.esm2_representation
        tertiary_structure_info = (args.tertiary_structure_method, os.path.join(os.getcwd(), args.tertiary_structure_path))

        # Load and validation data_preprocessing dataset
        data = load_and_validate_dataset(dataset)

        # Filter rows where 'partition' is equal to 1 (training data) or 2 (validation data)
        train_and_val_data = data[data['partition'].isin([1, 2])]

        # Check if train_and_val_data is empty
        if train_and_val_data.empty:
            raise ValueError("No data available for training.")

        # to get the graph representations
        graphs = graph_representations(train_and_val_data, esm2_representation, tertiary_structure_info, threshold, add_self_loop=True)
        labels = data.activity

        # Apply the mask to 'graph_representations' to training and validation data
        partitions = data.partition
        partition_train = partitions == 1
        partition_val = partitions == 2

        # If training and validation data were provided
        if partition_train.any() and partition_val.any():
            # Training graphs
            graphs_train = graphs[partition_train]
            graphs_train = shuffle(graphs_train)

            # Validation graphs
            graphs_val = graphs[partition_val]

        # If only training or validation data were provided
        else:
            # split training dataset: 80% train y 20% test, with seed and shuffle
            graphs_train, graphs_val, _, _ = train_test_split(graphs, labels, test_size=0.2, shuffle=True,
                                                          random_state=41)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('device:', device)

        node_feature_dim = graphs_train[0].x.shape[1]
        n_class = 2

        # tensorboard, record the change of auc, acc and loss
        writer = SummaryWriter()

        # load model
        if args.pretrained_model == "":
            model = GATModel(node_feature_dim, args.hd, n_class, args.drop, args.heads).to(device)
        else:
            model = torch.load(args.pretrained_model).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

        criterion = torch.nn.CrossEntropyLoss()
        train_dataloader = DataLoader(graphs_train, batch_size=args.b)
        val_dataloader = DataLoader(graphs_val, batch_size=args.b)

        best_acc = 0
        best_auc = 0
        min_loss = 1000
        save_acc = '/'.join(args.save.split('/')[:-1]) + '/acc_' + args.save.split('/')[-1]
        save_auc = '/'.join(args.save.split('/')[:-1]) + '/auc_' + args.save.split('/')[-1]
        save_loss = '/'.join(args.save.split('/')[:-1]) + '/loss_' + args.save.split('/')[-1]

        for epoch in range(args.e):
            print('Epoch ', epoch)
            model.train()
            arr_loss = []
            for i, data in enumerate(train_dataloader):
                optimizer.zero_grad()
                data = data.to(device)

                output = model(data.x, data.edge_index, data.batch)
                out = output[0]
                loss = criterion(out, data.y)
                loss.backward()
                optimizer.step()
                arr_loss.append(loss.item())

            avgl = np.mean(arr_loss)
            print("Training Average loss :", avgl)

            model.eval()
            with torch.no_grad():
                total_num = 0
                total_correct = 0
                preds = []
                y_true = []
                arr_loss = []
                for data in val_dataloader:
                    data = data.to(device)

                    output = model(data.x, data.edge_index, data.batch)
                    out = output[0]

                    loss = criterion(out, data.y)
                    arr_loss.append(loss.item())

                    pred = out.argmax(dim=1)
                    score = F.softmax(out, dim=1)[:, 1]
                    correct = (pred == data.y).sum().float()
                    total_correct += correct
                    total_num += data.num_graphs
                    preds.extend(score.cpu().detach().data.numpy())
                    y_true.extend(data.y.cpu().detach().data.numpy())

                acc = (total_correct / total_num).cpu().detach().data.numpy()
                auc = roc_auc_score(y_true, preds)
                val_loss = np.mean(arr_loss)
                print("Validation accuracy: ", acc)
                print("Validation auc:", auc)
                print("Validation loss:", val_loss)

                writer.add_scalar('Loss', avgl, global_step=epoch)
                writer.add_scalar('acc', acc, global_step=epoch)
                writer.add_scalar('auc', auc, global_step=epoch)

                if acc > best_acc:
                    best_acc = acc
                    torch.save(model, save_acc)

                if auc > best_auc:
                    best_auc = auc
                    torch.save(model, save_auc)

                if np.mean(val_loss) < min_loss:
                    min_loss = val_loss
                    torch.save(model, save_loss)

                print('-' * 50)

            scheduler.step()

        print('best acc:', best_acc)
        print('best auc:', best_auc)
        if args.o is not None:
            with open(args.o, 'a') as f:
                localtime = time.asctime(time.localtime(time.time()))
                f.write(str(localtime) + '\n')
                f.write('args: ' + str(args) + '\n')
                f.write('auc result: ' + str(best_auc) + '\n\n')

    except Exception as e:
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('-dataset', type=str, default='datasets/Test_Data/Test_Data.csv',
                        help='Path to the dataset in csv format')

    # methods for graphs construction
    parser.add_argument('-esm2_representation', type=str, default='esm2_t6',
                        help='Representation derived from ESM models to be used')
    parser.add_argument('-tertiary_structure_method', type=str, default='trRosetta',
                        help='Method of generation of 3D structures to be used')
    parser.add_argument('-tertiary_structure_path', type=str, default='datasets/Test_Data/trRosetta_output/npz/',
                        help='Path of the tertiary structures generated with the method specified in the parameter tertiary_structure_method')

    # training parameters
    # 0.001 for pretrain， 0.0001 for train
    parser.add_argument('-lr', type=float, default=0.001, help='Learning rate') 
    parser.add_argument('-drop', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('-e', type=int, default=50, help='Maximum number of epochs')
    parser.add_argument('-b', type=int, default=512, help='Batch size')
    parser.add_argument('-hd', type=int, default=64, help='Hidden layer dim')

    # model to be used for training and output path
    parser.add_argument('-pretrained_model', type=str, default="",
                        help='The path of pretraining model, if "", the model will be trained from scratch')
    parser.add_argument('-save', type=str, default='output_models/samp.model',
                        help='The path saving the trained models')
    parser.add_argument('-heads', type=int, default=8, help='Number of heads')

    parser.add_argument('-d', type=int, default=37, help='Distance threshold to construct a graph, 0-37, 37 means 20A')

    # log path
    parser.add_argument('-o', type=str, default='log.txt', help='File saving the raw prediction results')

    args = parser.parse_args()

    start_time = datetime.datetime.now()

    train(args)

    end_time = datetime.datetime.now()
    print('End time(min):', (end_time - start_time).seconds / 60)
