from abc import abstractmethod
import pandas as pd
from workflow.parameters_setter import ParameterSetter


class DataLoader:
    @abstractmethod
    def read_file(self, workflow_settings: ParameterSetter) -> pd.DataFrame:
        pass


class CSVLoader(DataLoader):
    def read_file(self, workflow_settings: ParameterSetter) -> pd.DataFrame:
        dataset_df = pd.read_csv(workflow_settings.dataset)
        yield dataset_df


class CSVByChunkLoader(DataLoader):
    def read_file(self, workflow_settings: ParameterSetter) -> pd.DataFrame:
        for data_chunk in pd.read_csv(workflow_settings.dataset,
                                      chunksize=workflow_settings.inference_batch_size):
            yield data_chunk


class FASTALoader(DataLoader):
    def read_file(self, workflow_settings: ParameterSetter) -> pd.DataFrame:
        data = []
        print("Not implemented yet")
        return data


class DataLoaderContext:
    def __init__(self, data_loader: DataLoader) -> None:
        self._data_loader = data_loader

    def read_file(self, workflow_settings: ParameterSetter) -> pd.DataFrame:
        return self._data_loader.read_file(workflow_settings)
