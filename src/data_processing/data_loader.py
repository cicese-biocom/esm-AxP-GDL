from abc import abstractmethod, ABC
from typing import Any, Generator
import pandas as pd
from Bio import SeqIO
from pydantic.v1 import FilePath, PositiveInt


class DataLoader(ABC):
    @abstractmethod
    def read_file(self, dataset: FilePath) -> pd.DataFrame:
        pass


class CSVLoader(DataLoader):
    def read_file(self, dataset: FilePath) -> pd.DataFrame:
        return pd.read_csv(dataset)



class FastaLoader(DataLoader):
    def read_file(self, dataset: FilePath) -> pd.DataFrame:
        records = []

        for record in SeqIO.parse(str(dataset), "fasta"):
            records.append({
                "id": record.id if record.id else "",
                "sequence": str(record.seq)
            })

        data = pd.DataFrame(records, columns=["id", "sequence"])
        return pd.DataFrame(records, columns=["id", "sequence"])


class DataLoaderContext:
    def __init__(self, data_loader: DataLoader) -> None:
        self._data_loader = data_loader

    def read_file(self, dataset: FilePath) -> pd.DataFrame:
        return self._data_loader.read_file(dataset)
