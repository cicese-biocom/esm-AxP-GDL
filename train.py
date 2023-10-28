import numpy as np
import torch
from torch_geometric.loader import DataLoader
import argparse
from models.GAT.GAT import GATModel
from tools.data_preprocessing.dataset_processing import load_and_validate_dataset
from graph.construct_graphs import construct_graphs
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score,matthews_corrcoef
import torch.nn.functional as F
import os
from tqdm import tqdm
import time

def train(args):
    try:
        # load arguments
        threshold = args.d
        dataset = args.dataset
        esm2_representation = args.esm2_representation
        tertiary_structure_config = (args.tertiary_structure_method, os.path.join(os.getcwd(), args.tertiary_structure_path), args.tertiary_structure_operation_mode)

        # Load and validation data_preprocessing dataset
        data = load_and_validate_dataset(dataset)

        # Filter rows where 'partition' is equal to 1 (training data) or 2 (validation data)
        train_and_val_data = data[data['partition'].isin([1, 2])].reset_index(drop=True)

        # Check if train_and_val_data is empty
        if train_and_val_data.empty:
            raise ValueError("No data available for training.")

        # to get the graph representations
        graphs = construct_graphs(train_and_val_data, esm2_representation, tertiary_structure_config, threshold, add_self_loop=True)
        labels = data.activity

        # Apply the mask to 'graph_representations' to training and validation data
        partitions = data.partition

        partition_train = any(x == 1 for x in partitions)
        partition_val = any(x == 2 for x in partitions)

        # If training and validation data were provided
        if partition_train and partition_val:
            graphs_train = []
            graphs_val = []

            for graph, group in zip(graphs, partitions):
                if group == 1:
                    graphs_train.append(graph)
                elif group == 2:
                    graphs_val.append(graph)

        # If only training or validation data were provided
        else:
            # split training dataset: 80% train y 20% test, with seed and shuffle
            graphs_train, graphs_val, _, _ = train_test_split(graphs, labels, test_size=0.2, shuffle=True,
                                                          random_state=41)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        node_feature_dim = graphs_train[0].x.shape[1]
        n_class = 2

        # tensorboard, record the change of train_loss, val_loss, mcc, acc and auc
        writer = SummaryWriter()

        # load model
        if args.pretrained_model == "":
            model = GATModel(node_feature_dim, args.hd, n_class, args.drop, args.heads).to(device)
        else:
            model = torch.load(args.pretrained_model).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

        criterion = torch.nn.CrossEntropyLoss()
        train_dataloader = DataLoader(graphs_train, batch_size=args.b)
        val_dataloader = DataLoader(graphs_val, batch_size=args.b)

        model_path = os.path.dirname(args.save)
        if not model_path.endswith(os.sep):
            model_path = model_path + os.sep
        os.makedirs(model_path, exist_ok=True)

        best_mcc = 0
        epochs = args.e
        bar = tqdm(total=epochs, desc="Training and Validation:")
        for epoch in range(1, epochs + 1):
            model.train()
            arr_loss = []
            for data in train_dataloader:
                optimizer.zero_grad()
                data = data.to(device)

                output = model(data.x, data.edge_index, data.batch)
                out = output[0]
                loss = criterion(out, data.y)
                loss.backward()
                optimizer.step()
                arr_loss.append(loss.item())

            torch.save({
                'epoch': epoch,
                'model': model,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }, os.path.join(model_path, f"checkpoint_epoch{epoch}.pt"))

            model.eval()
            with torch.no_grad():
                total_num = 0
                total_correct = 0
                preds = []
                y_true = []
                y_pred = []
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
                    y_pred.extend(pred.cpu().detach().data.numpy())
                    preds.extend(score.cpu().detach().data.numpy())
                    y_true.extend(data.y.cpu().detach().data.numpy())

                train_loss = np.mean(arr_loss)
                val_loss = np.mean(arr_loss)
                mcc = matthews_corrcoef(y_true, y_pred)
                acc = (total_correct / total_num).cpu().detach().data.numpy()
                auc = roc_auc_score(y_true, preds)

                bar.update(1)
                bar.set_postfix(
                    Epoch=f"{epoch}",
                    Training_Loss=f"{train_loss:.4f}",
                    Validation_Loss=f"{val_loss:.4f}",
                    Validation_MCC=f"{mcc:.4f}",
                    Validation_ACC=f"{acc:.4f}",
                    Validation_AUC=f"{auc:.4f}"
                )

                writer.add_scalar('Loss/train', train_loss, global_step=epoch)
                writer.add_scalar('Loss/validation', val_loss, global_step=epoch)
                writer.add_scalar('MCC/validation', mcc, global_step=epoch)
                writer.add_scalar('ACC/validation', acc, global_step=epoch)
                writer.add_scalar('AUC/validation', auc, global_step=epoch)

                if mcc > best_mcc:
                    best_mcc = mcc
                    acc_of_the_best_mcc = acc
                    auc_of_the_best_mcc = auc
                    val_loss_of_the_best_mcc = val_loss
                    train_loss_of_the_best_mcc = train_loss
                    epoch_of_the_best_mcc = epoch

            scheduler.step()

        bar.set_postfix(
            Epoch_Best_MCC=f"{epoch_of_the_best_mcc}",
            Training_Loss=f"{train_loss_of_the_best_mcc:.4f}",
            Validation_Loss=f"{val_loss_of_the_best_mcc:.4f}",
            Validation_MCC=f"{best_mcc:.4f}",
            Validation_ACC=f"{acc_of_the_best_mcc:.4f}",
            Validation_AUC=f"{auc_of_the_best_mcc:.4f}"
        )
        bar.close()

        if args.o is not None:
            with open(args.o, 'a') as f:
               localtime = time.asctime(time.localtime(time.time()))
               f.write(str(localtime) + '\n')
               f.write('args: ' + str(args) + '\n')
               f.write('Best MCC result: ' + str(best_mcc) + '\n\n')

    except Exception as e:
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('-dataset', type=str, default='datasets/DeepAVPpred/DeepAVPpred.csv',
                        help='Path to the dataset in csv format')

    # methods for graphs construction
    parser.add_argument('-esm2_representation', type=str, default='esm2_t6',
                        help='Representation derived from ESM models to be used')

    parser.add_argument('-tertiary_structure_method', type=str, default='esmfold',
                        help='Method of generation of 3D structures to be used')
    parser.add_argument('-tertiary_structure_path', type=str, default='datasets/DeepAVPpred/ESMFold_pdbs/',
                        help='Path to load or save generated tertiary structures')
    parser.add_argument('-tertiary_structure_operation_mode', type=str, default='load',
                        help="Specifies the mode of operation. You can choose between ""load"" "
                             "to load existing tertiary structures or ""generate"" to create new ones")

    # training parameters
    # 0.001 for pretrain， 0.0001 for train
    parser.add_argument('-lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('-drop', type=float, default=0.25, help='Dropout rate')
    parser.add_argument('-e', type=int, default=20, help='Maximum number of epochs')
    parser.add_argument('-b', type=int, default=512, help='Batch size')
    parser.add_argument('-hd', type=int, default=64, help='Hidden layer dim')

    # model to be used for training and output path
    parser.add_argument('-pretrained_model', type=str, default="",
                        help='The path of pretraining model, if "", the model will be trained from scratch')
    parser.add_argument('-save', type=str, default='output_models/',
                        help='The path saving the trained models')
    parser.add_argument('-heads', type=int, default=8, help='Number of heads')

    parser.add_argument('-d', type=int, default=20, help='Distance threshold to construct a graph')

    # log path
    parser.add_argument('-o', type=str, default='log.txt', help='File saving the raw prediction results')

    args = parser.parse_args()

    train(args)

