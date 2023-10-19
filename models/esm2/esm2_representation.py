import numpy as np
import pandas as pd
import torch
from torch import hub
import esm
from esm import FastaBatchedDataset
import os
import json
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from torch.nn.functional import normalize

def get_models(esm2_representation):
    """
    :param esm2_representation: residual-level features representation name
    :return:
        models: models corresponding to the specified esm 2 representation
    """

    esm2_representations_json = os.getcwd() + os.sep + "models/esm2/esm2_representations.json"

    # Read JSON in a DataFrame
    with open(esm2_representations_json, 'r') as json_file:
        data = json.load(json_file)

    # Create a DataFrame
    representations = pd.DataFrame(data["representations"])

    # Filter encoding method
    representation = representations[representations["representation"] == esm2_representation]

    # Check if the DataFrame is empty
    if not representation.empty:
        # Extract the column "models" and create a new DataFrame
        models = representation["models"].explode(ignore_index=True)
    else:
        #  If the DataFrame is empty, throw an exception and stop the code.
        raise Exception(f"'{esm2_representation}' is not a valid coding method name.")

    return models


def get_embeddings(data, model_name, reduced_features, normalize_embedding):
    """
    :param ids: sequences identifiers. Containing multiple sequences.
    :param sequences: sequences itself
    :param model_name: esm2 model name
    :param reduced_features: vector of positions of the features to be used
    :param normalize_embedding: whether to normalize the embedding using Min-Max scaling
    :return:
        embeddings: reduced embedding of each sequence of the fasta file according to reduced_features
    """
    try:
        # esm2 checkpoints
        hub.set_dir(os.getcwd() + os.sep + "models/esm2/")

        no_gpu = False
        model, alphabet = esm.pretrained.load_model_and_alphabet_hub(model_name)
        model.eval()  # disables dropout for deterministic results

        if torch.cuda.is_available() and not no_gpu:
            model = model.cuda()
            #print("Transferred model to GPU")

        dataset = FastaBatchedDataset(data.id, data.sequence)
        batches = dataset.get_batch_indices(toks_per_batch=1, extra_toks_per_seq=1)
        data_loader = torch.utils.data.DataLoader(dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=None)
        #print(f"Read FASTA file with {len(dataset)} sequences")

        scaler = MinMaxScaler()
        repr_layers = model.num_layers
        embeddings = []

        with torch.no_grad():
                for batch_idx, (labels, strs, toks) in tqdm(enumerate(data_loader), total=len(data_loader), desc ="Generating esm2 embeddings"):
                    if torch.cuda.is_available() and not no_gpu:
                        toks = toks.to(device="cuda", non_blocking=True)

                    representation = model(toks, repr_layers=[repr_layers], return_contacts=False)["representations"][repr_layers]

                    for i, label in enumerate(labels):
                        layer_for_i = representation[i, 1:len(strs[i]) + 1]

                        reduced_features = np.array(reduced_features)
                        if len(reduced_features) > 0:
                            layer_for_i = layer_for_i[:, reduced_features]

#                        embedding = layer_for_i.cpu().numpy()

                        if normalize_embedding:
#                            min_values = embedding.min(axis=1)
#                            max_values = embedding.max(axis=1)
#                            embedding = (embedding - min_values[:, np.newaxis]) / (max_values - min_values)[:,
#                                                                                           np.newaxis]
                             layer_for_i = normalize(layer_for_i)

#                        embeddings.append(embedding)
                        embeddings.append(layer_for_i.cpu().numpy())

        return embeddings

    except Exception as e:
        print(f"Error in get_embeddings function: {e}")
