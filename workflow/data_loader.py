from abc import abstractmethod
from typing import List
from pathlib import Path
import pandas as pd


class DataLoader:
    @abstractmethod
    def read_file(self, filepath: Path):
        pass


class CSVLoader(DataLoader):
    def read_file(self, filepath: Path) -> List:
        dataset_df = pd.read_csv(filepath)
        return dataset_df


class FASTALoader(DataLoader):
    def read_file(self, filepath: Path) -> List:
        data = []
        print("Not implemented yet")
        return data


class DataLoaderContext:
    def __init__(self, data_loader: DataLoader) -> None:
        self._data_loader = data_loader

    def read_file(self, filepath: Path) -> pd.DataFrame:
        return self._data_loader.read_file(filepath)
