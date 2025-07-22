import os
from pathlib import Path
from statistics import mean
import networkx as nx
import torch
from torch_geometric.utils import to_networkx
from typing import List, Optional
from networkx import Graph
from typing import Dict
import pandas as pd
from tqdm import tqdm

from src.config.enum import ESM2Representation
from src.models.esm2 import get_models, get_representations
from src.utils.dto import DTO


class FeatureDTO(DTO):
    features_to_calculate: List
    data: pd.DataFrame
    graphs: List
    perplexities: Optional[pd.DataFrame]
    device: torch.device


class FeatureComponent:
    def compute_features(self, pbar: tqdm) -> pd.DataFrame:
        pass


class EmptyFeatureComponent(FeatureComponent):
    def __init__(self, data: pd.DataFrame) -> pd.DataFrame:
        self._features = data[['sequence']].copy()

    @property
    def features(self) -> str:
        return self._features

    def compute_features(self, pbar: tqdm) -> pd.DataFrame:
        return self.features


class FeatureDecorator(FeatureComponent):
    _feature_component: FeatureComponent = None

    def __init__(self, feature_component: FeatureComponent):
        self._feature_component = feature_component

    @property
    def feature_component(self) -> FeatureComponent:
        return self._feature_component

    def compute_features(self, pbar: tqdm) -> pd.DataFrame:
        return self.feature_component.compute_features(pbar)


class GraphCentralitiesDecorator(FeatureDecorator):
    def __init__(
            self,
            feature_component: FeatureComponent,
            feature_id: str,
            data: pd.DataFrame,
            graphs: List[Graph]):
        super().__init__(feature_component)
        self._feature_id = feature_id
        self._graphs = graphs
        self._sequences = data['sequence'].tolist()

    @property
    def feature_id(self) -> str:
        return self._feature_id

    @property
    def sequences(self) -> List[str]:
        return self._sequences

    @property
    def graphs(self) -> List[Graph]:
        return self._graphs

    def compute_features(self, pbar: tqdm) -> pd.DataFrame:
        features_input = self.feature_component.compute_features(pbar)

        centrality_functions = {
            'degree_centrality': nx.degree_centrality,
            'eigenvector_centrality': nx.eigenvector_centrality_numpy,
            'closeness_centrality': nx.closeness_centrality,
            'betweenness_centrality': nx.betweenness_centrality,
            'harmonic_centrality': nx.harmonic_centrality
        }

        features_list = []  # Collect new features from all graphs

        for graph, sequence in zip(self.graphs, self.sequences):
            graph_nx = to_networkx(graph, to_undirected=True)
            feat = {'sequence': sequence}

            for metric, func in centrality_functions.items():
                result = func(graph_nx)
                feat[f"{self.feature_id}_{metric}"] = mean(result.values())

            features_list.append(feat)

        features_new = pd.DataFrame(features_list)
        features_output = features_input.merge(features_new, how='inner', on='sequence')

        pbar.update(1)

        return features_output


class AminoAcidDescriptorDecorator(FeatureDecorator):
    def __init__(
            self,
            feature_component: FeatureComponent,
            feature_id: str,
            data: pd.DataFrame):
        super().__init__(feature_component)
        self._feature_id = feature_id
        self._sequences = data['sequence'].tolist()

    @property
    def feature_id(self) -> str:
        return self._feature_id

    @property
    def sequences(self) -> List[str]:
        return self._sequences

    def compute_features(self, pbar: tqdm) -> pd.DataFrame:
        features_input = self.feature_component.compute_features(pbar)

        # Load amino acid descriptors
        descriptors = pd.read_csv(Path(os.getenv("AMINO_ACID_DESCRIPTORS_FILE")).resolve())

        features_list = []
        for sequence in self.sequences:
            # Convert sequence to DataFrame for merging
            sequence_df = pd.DataFrame(list(sequence), columns=["code_short"])

            # Merge sequence with descriptors
            sequence_descriptors = sequence_df.merge(descriptors, on="code_short", how="left")

            # Calculate mean of numeric descriptors
            numeric_descriptors = sequence_descriptors.select_dtypes(include=['float64', 'int64'])
            mean_descriptors = numeric_descriptors.mean(axis=0)

            # Create feature dictionary for the current sequence
            features = {f"{self.feature_id}_{col}": val for col, val in mean_descriptors.items()}
            features['sequence'] = sequence  # Include sequence identifier
            features_list.append(features)

        # Convert the list of features to a DataFrame
        features_new = pd.DataFrame(features_list)

        # Combine the new features with the input features
        features_output = features_input.merge(features_new, on="sequence", how="inner")

        pbar.update(1)

        return features_output


class PerplexityDecorator(FeatureDecorator):
    def __init__(
            self,
            feature_component: FeatureComponent,
            perplexities: pd.DataFrame,
            data: pd.DataFrame,
            device: torch.device
    ):
        super().__init__(feature_component)
        self._perplexities = perplexities
        self._data = data.copy(deep=True)
        self._device = device

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def perplexities(self) -> pd.DataFrame:
        return self._perplexities

    @property
    def device(self) -> torch.device:
        return self._device

    def compute_features(self, pbar: tqdm) -> pd.DataFrame:
        features_input = self.feature_component.compute_features(pbar)

        perplexities = self.perplexities
        if perplexities.empty:
            model = get_models(ESM2Representation.ESM2_T33)
            _, _, perplexities = get_representations(self.data, model[0]['model'], device=self.device, show_pbar=True)

        features_output = features_input.merge(perplexities, on="sequence", how="inner")

        pbar.update(1)
        return features_output


class FeaturesContext:
    def __init__(self):
        self.function_mapping = {
            'graph_centralities': GraphCentralitiesDecorator,
            'amino_acid_descriptors': AminoAcidDescriptorDecorator,
            'perplexity': PerplexityDecorator
        }

    def compute_features(self, **kwargs):
        features_component = EmptyFeatureComponent(kwargs['data'])

        features_to_calculate = kwargs['features_to_calculate']

        for feat in features_to_calculate:
            if feat['feature_name'] in self.function_mapping:
                func = self.function_mapping[feat['feature_name']]
                kwargs['feature_component'] = features_component
                kwargs['feature_id'] = feat['feature_id']
                params = {param: kwargs[param] for param in kwargs if param in func.__init__.__code__.co_varnames}

                features_component = func(**params)

        with tqdm(total=len(features_to_calculate), desc="Computing features") as pbar:
            features = features_component.compute_features(pbar)

        return features


def filter_features(
        features: pd.DataFrame,
        features_to_select: List[Dict]
) -> pd.DataFrame:

    group_ids = [feat['feature_id'] for feat in features_to_select if feat['type'] == "group"]
    feature_ids = [feat['feature_name'] for feat in features_to_select if feat['type'] == "feature"]

    filtered_features = features.loc[:, features.columns.str.startswith(tuple(group_ids)) | features.columns.isin(feature_ids)]

    return filtered_features

