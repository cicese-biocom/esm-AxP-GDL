import os
from pathlib import Path
from typing import Any, Optional

from pandas import DataFrame
from torcheval import metrics
import pandas as pd
import torch
import esm
from esm import FastaBatchedDataset
from tqdm import tqdm

from src.config.enum import ESM2Representation
from src_old.utils import json_parser
from torch.utils.data import DataLoader


def get_models(esm2_representation: ESM2Representation):
    data = json_parser.load_json(
        Path(os.getenv("ESM2_REPRESENTATION_CONFIG_FILE")).resolve()
    )

    # Create a DataFrame
    representations = pd.DataFrame(data["representations"])

    # Filter encoding method
    representation = representations[representations["representation"] == esm2_representation.value]

    # Check if the DataFrame is empty
    if not representation.empty:
        # Extract the column "models" and create a new DataFrame
        models = representation["models"].explode(ignore_index=True)
    else:
        #  If the DataFrame is empty, throw an exception and stop the code.
        raise Exception(f"'{esm2_representation}' is not a valid coding method name.")

    return models


def get_representations(data, model_name, device, show_pbar=False) -> Optional[tuple[list[Any], list[Any], DataFrame]]:
    try:
        no_gpu = False
        model, alphabet = esm.pretrained.load_model_and_alphabet_hub(model_name)
        model.eval()  # disables dropout for deterministic results

        if torch.cuda.is_available() and not no_gpu:
            model = model.cuda()

        data_loader = DataLoader(
            dataset=FastaBatchedDataset(data.id, data.sequence),
            collate_fn=alphabet.get_batch_converter(),
            batch_sampler=None
        )

        repr_layers = model.num_layers
        embeddings = []
        contact_maps = []
        perplexities = []

        with torch.no_grad():
            for batch_idx, (labels, strs, toks) in tqdm(enumerate(data_loader),
                                                        total=len(data_loader),
                                                        desc=f"Running ESM-2 {model_name} model",
                                                        disable=show_pbar):
                if torch.cuda.is_available() and not no_gpu:
                    toks = toks.to(device="cuda", non_blocking=True)

                result = model(toks, repr_layers=[repr_layers], return_contacts=True)
                representation = result["representations"][repr_layers]

                # embedding
                layer_for_i = representation[0, 1:len(strs[0]) + 1]
                embedding = layer_for_i.cpu().numpy()
                embeddings.append(embedding)

                # contact map
                contact_map = result["contacts"][0]
                contact_map = contact_map.cpu().numpy()
                contact_maps.append(contact_map)

                # perplexity
                input = result["logits"][:, 1:-1, :]
                target = toks[:, 1:-1]

                perplexity_metric = metrics.Perplexity(device=device)
                perplexity_metric.update(input, target)

                perplexities.append(
                    {
                        'sequence': strs[0],
                        'perplexity': perplexity_metric.compute().item()
                    })

                perplexity_metric.reset()

        return embeddings, contact_maps, pd.DataFrame(perplexities)

    except Exception as e:
        print(f"Error in get_embeddings function: {e}")
