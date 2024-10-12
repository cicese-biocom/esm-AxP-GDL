from pathlib import Path

import pandas as pd
import torch
from torch import hub
import esm
from esm import FastaBatchedDataset
import os
from tqdm import tqdm
from utils import json_parser


def get_models(esm2_representation):
    """
    get_models
    :param esm2_representation: residual-level features representation name
    :return:
        models: models corresponding to the specified esm 2 representation
    """

    esm2_representations_json = Path.cwd().joinpath("settings", "esm2_representations.json")
    data = json_parser.load_json(esm2_representations_json)

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


def get_representations(data, model_name):
    """
    get_representations
    :param data:
    :param model_name: esm2 model name
    :return:
        embeddings:
        contact_map:
    """
    try:
        # esm2 checkpoints
        hub.set_dir(os.getcwd() + os.sep + "models/esm2/")

        no_gpu = False
        model, alphabet = esm.pretrained.load_model_and_alphabet_hub(model_name)
        model.eval()  # disables dropout for deterministic results

        if torch.cuda.is_available() and not no_gpu:
            model = model.cuda()

        dataset = FastaBatchedDataset(data.id, data.sequence)
        data_loader = torch.utils.data.DataLoader(dataset, collate_fn=alphabet.get_batch_converter(),
                                                  batch_sampler=None)

        repr_layers = model.num_layers
        embeddings = []
        contact_maps = []

        with torch.no_grad():
            for batch_idx, (labels, strs, toks) in tqdm(enumerate(data_loader),
                                                        total=len(data_loader),
                                                        desc=f"Generating embedding and contact maps using the ESM-2 {model_name} model "):
                if torch.cuda.is_available() and not no_gpu:
                    toks = toks.to(device="cuda", non_blocking=True)

                result = model(toks, repr_layers=[repr_layers], return_contacts=True)
                representation = result["representations"][repr_layers]

                for i, label in enumerate(labels):
                    layer_for_i = representation[i, 1:len(strs[i]) + 1]

                    embedding = layer_for_i.cpu().numpy()
                    embeddings.append(embedding)

                    contact_map = result["contacts"][0]
                    contact_map = contact_map.cpu().numpy()
                    contact_maps.append(contact_map)
        return embeddings, contact_maps

    except Exception as e:
        print(f"Error in get_embeddings function: {e}")
