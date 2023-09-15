import numpy as np
import pandas as pd
import torch
from torch import hub
import esm
from esm import FastaBatchedDataset
import os
import json


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
        raise Exception(f"'{representation_name}' is not a valid coding method name.")

    return models


def get_embeddings(data, model_name, reduced_features):
    """
    :param ids: sequences identifiers. Containing multiple sequences.
    :param sequences: sequences itself
    :param model_name: esm2 model name
    :param reduced_features: vector of positions of the features to be used
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
            print("Transferred model to GPU")

        dataset = FastaBatchedDataset(data.id, data.sequence)
        batches = dataset.get_batch_indices(toks_per_batch=1, extra_toks_per_seq=1)
        data_loader = torch.utils.data.DataLoader(dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=None)
        print(f"Read FASTA file with {len(dataset)} sequences")


        repr_layers = model.num_layers
        embeddings = []
        with torch.no_grad():
            for batch_idx, (labels, strs, toks) in enumerate(data_loader):
                print(f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)")

                if torch.cuda.is_available() and not no_gpu:
                    toks = toks.to(device="cuda", non_blocking=True)

                representation = model(toks, repr_layers=[repr_layers], return_contacts=False)["representations"][repr_layers]

                for i, label in enumerate(labels):
                    layer_for_i = representation[i, 1:len(strs[i]) + 1]
                    embedding = layer_for_i.numpy()

                    reduced_features = np.array(reduced_features)
                    if len(reduced_features) > 0:
                        embedding = embedding[:, reduced_features]

                embeddings.append(embedding)
        return embeddings
    except Exception as e:
        print(f"Error in get_embeddings function: {e}")