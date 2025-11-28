from typing import Dict, List
import pandas as pd
import re
from abc import ABC, abstractmethod
import logging
partitions = [1, 2, 3]
pattern = re.compile('[^ARNDCQEGHILKMFPSTWYV]')


class TargetFeatureValidator(ABC):
    @abstractmethod
    def validate(self, data: pd.DataFrame, classes: List[int] = None) -> pd.DataFrame:
        pass


class ClassificationTargetFeatureValidator(TargetFeatureValidator):
    def validate(self, data: pd.DataFrame, classes: List[int] = None) -> pd.DataFrame:
        # Return rows where 'activity' is not a class
        return data[~data['activity'].isin(classes)]


class RegressionTargetFeatureValidator(TargetFeatureValidator):
    def validate(self, data: pd.DataFrame, classes: List[int] = None) -> pd.DataFrame:
        # Return rows where 'activity' is not a float
        return data[~data['activity'].apply(lambda x: isinstance(x, float))]
    

class DatasetProcessor(ABC):       
    def calculate(
            self,
            dataset: pd.DataFrame,
            output_dir: Dict,
            class_validator: TargetFeatureValidator,
            classes: List[int],
    ):
        try:
            valid_df = self.check_duplicated_sequence_ids(dataset, output_dir)
            valid_df = self.check_duplicated_sequences(valid_df, output_dir)
            filtered_df = self.filter_sequences_with_non_natural_amino_acids(valid_df, output_dir)
            self.check_sequences_with_erroneous_activity(filtered_df, output_dir, class_validator, classes)
            self.check_sequences_with_baseless_partitions(filtered_df, output_dir)

            if filtered_df.empty:
                raise Exception(f"Empty dataset")

            transform_df = self.transform(filtered_df)

            return transform_df

        except Exception as e:
            logging.getLogger('workflow_logger').exception(e)
            quit()

    @staticmethod
    def transform(dataset) -> pd.DataFrame:
        sequence_df = dataset.copy()
        return sequence_df.assign(length=sequence_df['sequence'].str.len())

    @staticmethod
    def not_sequences_with_non_natural_amino_acids(sequence):
        return pattern.search(sequence) is not None

    def filter_sequences_with_non_natural_amino_acids(self, dataset: pd.DataFrame,
                                                      output_dir: Dict) -> pd.DataFrame:
        csv_file = output_dir['sequences_with_non_natural_amino_acids_file']
        sequence_df = dataset.copy()
        sequences_to_exclude_mask = sequence_df['sequence'].apply(self.not_sequences_with_non_natural_amino_acids)
        sequences_to_exclude = sequence_df[sequences_to_exclude_mask]

        if not sequences_to_exclude.empty:
            sequences_to_exclude.to_csv(csv_file, index=False)
            sequence_df = sequence_df.drop(sequences_to_exclude.index)
            logging.getLogger('workflow_logger'). \
                warning(f"Sequences with non-natural amino acids were excluded. See: {csv_file}")
        return sequence_df

    def check_sequences_with_erroneous_activity(
            self, dataset: pd.DataFrame,
            output_dir: Dict,
            class_validator: TargetFeatureValidator,
            classes: List
    ) -> None:
        return None

    def check_sequences_with_baseless_partitions(self, dataset: pd.DataFrame, output_dir: Dict) \
            -> pd.DataFrame:
        return dataset

    @staticmethod
    def check_duplicated_sequence_ids(dataset: pd.DataFrame, output_dir: Dict) -> pd.DataFrame:
        csv_file = output_dir['duplicated_sequence_ids_file']
        sequence_df = dataset.copy()
        sequences_to_exclude = sequence_df[sequence_df.duplicated(subset='id', keep=False)]

        if not sequences_to_exclude.empty:
            sequences_to_exclude.sort_values(by='id')
            sequences_to_exclude.to_csv(csv_file, index=False)
            raise ValueError(f"Duplicate sequences IDs. See: {csv_file}")
        return sequence_df

    @staticmethod
    def check_duplicated_sequences(dataset: pd.DataFrame, output_dir: Dict) -> pd.DataFrame:
        csv_file = output_dir['duplicated_sequences_file']
        sequence_df = dataset.copy()
        sequences_to_exclude = sequence_df[sequence_df.duplicated(subset='sequence', keep=False)]

        if not sequences_to_exclude.empty:
            sequences_to_exclude.sort_values(by='sequence')
            sequences_to_exclude.to_csv(csv_file, index=False)
            logging.getLogger('workflow_logger'). \
                warning(f"Duplicate sequences. See: {csv_file}")
        return sequence_df


class LabeledDatasetProcessor(DatasetProcessor):
    def check_sequences_with_erroneous_activity(
            self, dataset: pd.DataFrame,
            output_dir: Dict,
            class_validator: TargetFeatureValidator,
            classes: List = None
    ) -> None:
        csv_file = output_dir['sequences_with_erroneous_activity_file']
        sequence_df = dataset.copy()
        sequences_to_exclude = class_validator.validate(sequence_df, classes)

        if not sequences_to_exclude.empty:
            sequences_to_exclude.to_csv(csv_file, index=False)
            raise ValueError(f"Sequences with erroneous_activity. See: {csv_file}")


    def check_sequences_with_baseless_partitions(self, dataset: pd.DataFrame, output_dir: Dict) -> None:
        csv_file = output_dir['sequences_with_baseless_partitions_file']
        sequence_df = dataset.copy()
        sequences_to_exclude = sequence_df[~sequence_df['partition'].isin(partitions)]

        if not sequences_to_exclude.empty:
            sequences_to_exclude.to_csv(csv_file, index=False)
            raise ValueError(f"Sequences with baseless_partitions. See: {csv_file}")



class DatasetProcessorContext:
    def __init__(self, dataset_validator: DatasetProcessor) -> None:
        self._dataset_validator = dataset_validator

    def calculate(self, dataset: pd.DataFrame, output_dir: Dict):
        return self._dataset_validator.calculate(dataset, output_dir)
