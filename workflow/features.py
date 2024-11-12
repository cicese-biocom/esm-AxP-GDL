from pathlib import Path
from statistics import mean
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from typing import List
from networkx import Graph
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Dict
import pandas as pd
from pydantic import FilePath
from tqdm import tqdm


class FeatureComponent:
    def compute_features(self) -> Dict:
        pass


class EmptyFeatureComponent(FeatureComponent):
    def __init__(self, sequence: str) -> None:
        self._sequence = sequence

    @property
    def sequence(self) -> str:
        return self._sequence

    def compute_features(self) -> Dict:
        return {'sequence': self.sequence}


class FeatureDecorator(FeatureComponent):
    _feature_component: FeatureComponent = None

    def __init__(self, feature_component: FeatureComponent):
        self._feature_component = feature_component

    @property
    def feature_component(self) -> FeatureComponent:
        return self._feature_component

    def compute_features(self) -> Dict:
        return self.feature_component.compute_features()


class GraphCentralitiesDecorator(FeatureDecorator):
    def __init__(
            self,
            feature_component: FeatureComponent,
            feature_id: str,
            graph: Graph):
        super().__init__(feature_component)
        self._feature_id = feature_id
        self._graph = graph

    @property
    def feature_id(self) -> str:
        return self._feature_id

    @property
    def graph(self) -> Graph:
        return self._graph

    def compute_features(self) -> Dict:
        features_input = self.feature_component.compute_features()

        centrality_functions = {
            'degree_centrality': nx.degree_centrality,
            'eigenvector_centrality': nx.eigenvector_centrality_numpy,
            'closeness_centrality': nx.closeness_centrality,
            'betweenness_centrality': nx.betweenness_centrality,
            'information_centrality': nx.information_centrality,
            'harmonic_centrality': nx.harmonic_centrality
        }

        features = {}
        for metric, func in centrality_functions.items():
            result = func(self.graph)
            features[f"{self.feature_id}_{metric}"] = mean(result.values())

        return {**features_input, **features}


class AminoAcidDescriptorDecorator(FeatureDecorator):
    def __init__(
            self,
            feature_component: FeatureComponent,
            feature_id: str,
            sequence: List[str]):
        super().__init__(feature_component)
        self._feature_id = feature_id
        self._sequence = sequence

    @property
    def feature_id(self) -> str:
        return self._feature_id

    @property
    def sequence(self) -> List[str]:
        return self._sequence

    def compute_features(self) -> Dict:
        features_input = self.feature_component.compute_features()

        descriptors_path = Path.cwd().joinpath("settings", "amino_acid_descriptors.csv")
        descriptors = pd.read_csv(descriptors_path)

        sequence_df = pd.DataFrame(list(self.sequence), columns=["code_short"])
        sequence_descriptors = sequence_df.merge(descriptors, on="code_short", how="left")

        numeric_descriptors = sequence_descriptors.select_dtypes(include=['float64', 'int64'])
        mean_descriptors = numeric_descriptors.mean(axis=0)

        features = {f"{self.feature_id}_{col}": val for col, val in mean_descriptors.items()}

        # Combine and return all features
        return {**features_input, **features}


class FeaturesContext:
    def __init__(self):
        self.function_mapping = {
            'graph_centralities': GraphCentralitiesDecorator,
            'amino_acid_descriptors': AminoAcidDescriptorDecorator
        }

    def compute_features(self, **kwargs):
        feature = kwargs['feature']
        sequence = kwargs['sequence']
        features_component = EmptyFeatureComponent(sequence)

        for item in feature:
            if item['feature_name'] in self.function_mapping:
                func = self.function_mapping[item['feature_name']]
                params = {param: kwargs[param] for param in kwargs if param in func.__init__.__code__.co_varnames}
                params['feature_component'] = features_component
                params['feature_id'] = item['feature_id']
                features_component = func(**params)

        return features_component.compute_features()


def filter_features(
        features: pd.DataFrame,
        features_to_select: List[Dict]
) -> pd.DataFrame:

    group_ids = [item['feature_id'] for item in features_to_select if item['type'] == "group"]
    feature_ids = [item['feature_name'] for item in features_to_select if item['type'] == "feature"]

    filtered_features = features.loc[:, features.columns.str.startswith(tuple(group_ids)) | features.columns.isin(feature_ids)]

    return filtered_features


def compute_features(
        features_to_calculate: List[Dict],
        csv_path: FilePath,
        sequences: List[str],
        graphs: List[Graph]
) -> Dict:
    num_cores = multiprocessing.cpu_count()

    feature_context = FeaturesContext()

    args = [{'feature': features_to_calculate,
             'sequence': sequence,
             'graph': graph
             } for (graph, sequence) in
            zip(graphs, sequences)]

    with ProcessPoolExecutor(max_workers=num_cores) as pool:
        with tqdm(range(len(graphs)), total=len(graphs), desc="Computing features",
                  disable=False) as progress:
            futures = []
            for arg in args:
                future = pool.submit(feature_context.compute_features, **arg)
                future.add_done_callback(lambda p: progress.update())
                futures.append(future)

            features = [future.result() for future in futures]
            features = pd.DataFrame(features)
            features.to_csv(csv_path, index=False)

            return features


if __name__ == "__main__":
    feature = [{
        'feature_name': 'amino_acid_descriptors',
        'feature_id': 'aad'}]
    sequence = 'GLF'
    edge_index = torch.tensor([[0, 0, 0, 1, 1],
                               [1, 2, 3, 2, 3]], dtype=torch.long)
    num_nodes = torch.max(edge_index).item() + 1
    graph = Data(edge_index=edge_index, num_nodes=num_nodes)
    graph = to_networkx(graph, to_undirected=True)

    feature_context = FeaturesContext()
    features = feature_context.compute_features(feature=feature,
                                                sequence=sequence,
                                                graph=graph)
    print(features)
