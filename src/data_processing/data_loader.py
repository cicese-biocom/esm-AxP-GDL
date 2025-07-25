from abc import abstractmethod
from typing import Any, Generator
import pandas as pd
from pydantic.v1 import FilePath, PositiveInt


class DataLoader:
    @abstractmethod
    def read_file(self, dataset: FilePath) -> Generator[Any, Any, None]:
        pass


class CSVLoader(DataLoader):
    def read_file(self, dataset: FilePath) -> Generator[Any, Any, None]:
        dataset_df = pd.read_csv(dataset)
        yield dataset_df


class CSVByChunkLoader(DataLoader):
    def __init__(self, prediction_batch_size: PositiveInt):
        self._prediction_batch_size = prediction_batch_size

    def read_file(self, dataset: FilePath) -> Generator[Any, Any, None]:
        for data_chunk in pd.read_csv(dataset, chunksize=self._prediction_batch_size):
            yield data_chunk


class DataLoaderContext:
    def __init__(self, data_loader: DataLoader) -> None:
        self._data_loader = data_loader

    def read_file(self, dataset: FilePath) -> Generator[Any, Any, None]:
        return self._data_loader.read_file(dataset)
