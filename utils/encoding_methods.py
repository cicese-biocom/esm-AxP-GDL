import numpy as np
import pandas as pd
import torch
from torch import hub
import esm
from esm import FastaBatchedDataset
import json
from config.config import Config


def get_models(esm2_representation):
    """
    :param esm2_representation: residual-level features representation name
    :return:
        models: models corresponding to the specified esm 2 representation
    """
    # json file
    config = Config()
    json_file = config.get('esm2_representations_json')

    # Read JSON in a DataFrame
    with open(json_file, 'r') as json_file:
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

def get_embeddings(fasta_file, model_name, reduced_features):
    """
    :param fasta_file: file path of fasta
    :param model_name: esm2 model name
    :param reduced_features: vector of positions of the features to be used
    :return:
        embeddings: reduced embedding of each sequence of the fasta file according to reduced_features
    """
    try:
        # esm2 checkpoints
        config = Config()
        esm2_checkpoints = config.get('esm2_checkpoints')

        hub.set_dir(esm2_checkpoints)
        no_gpu = False
        model, alphabet = esm.pretrained.load_model_and_alphabet_hub(model_name)
        model.eval()  # disables dropout for deterministic results

        if torch.cuda.is_available() and not no_gpu:
            model = model.cuda()
            print("Transferred model to GPU")

        dataset = FastaBatchedDataset.from_file(fasta_file)
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
                representation = representation.squeeze(0)
                embedding = representation.numpy()

                reduced_features = np.array(reduced_features)

                if len(reduced_features) > 0:
                    embedding = embedding[:, reduced_features]

                embeddings.append(embedding)
        return embeddings
    except Exception as e:
        print(f"Error in get_embeddings function: {e}")

def residual_level_features(fasta_file, esm2_representation):
    """
    :param fasta_file: file path of fasta
    :param esm2_representation: name of the esm2 representation to be used
    :return:
        residual_level_features: residual-level features vector
    """
    models = get_models(esm2_representation)

    residual_level_features = []
    if not models.empty:
        for model_info in models:
            model_name = model_info["model"]
            reduced_features = model_info["reduced_features"]
            reduced_features = [x - 1 for x in reduced_features]

            embeddings = get_embeddings(fasta_file, model_name, reduced_features)

            if len(residual_level_features) == 0:
                residual_level_features = np.copy(embeddings)
            else:
                residual_level_features = cat(residual_level_features, embeddings)

    return residual_level_features

def cat(*args):
    """
    :param args: feature matrices
    """
    res = args[0]
    for matrix in args[1:]:
        for i in range(len(matrix)):
            res[i] = np.hstack((res[i], matrix[i]))
    return res

