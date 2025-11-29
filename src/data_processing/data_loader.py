from abc import abstractmethod, ABC
from typing import Any, Generator
import pandas as pd
from pydantic.v1 import FilePath, PositiveInt


class DataLoader(ABC):
    @abstractmethod
    def read_file(self, dataset: FilePath) -> pd.DataFrame:
        pass


class CSVLoader(DataLoader):
    def read_file(self, dataset: FilePath) -> pd.DataFrame:
        return pd.read_csv(dataset)


class DataLoaderContext:
    def __init__(self, data_loader: DataLoader) -> None:
        self._data_loader = data_loader

    def read_file(self, dataset: FilePath) -> pd.DataFrame:
        return self._data_loader.read_file(dataset)
