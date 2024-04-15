from typing import Dict

import pandas as pd
import re
from abc import ABC
import logging

from injector import inject

activities = [1, 0]
partitions = [1, 2, 3]
pattern = re.compile('[^ARNDCQEGHILKMFPSTWYV]')


class DatasetValidator(ABC):
    def processing_dataset(self, dataset: pd.DataFrame, output_setting: Dict):
        try:
            valid_df = self.check_duplicated_sequence_ids(dataset, output_setting)
            valid_df = self.check_duplicated_sequences(valid_df, output_setting)
            filtered_df = self.filter_sequences_with_non_natural_amino_acids(valid_df, output_setting)            
            filtered_df = self.filter_sequences_with_erroneous_activity(filtered_df, output_setting)
            filtered_df = self.filter_sequences_with_baseless_partitions(filtered_df, output_setting)

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

    def filter_sequences_with_non_natural_amino_acids(self, dataset: pd.DataFrame, output_setting: Dict) -> pd.DataFrame:
        csv_file = output_setting['sequences_with_non_natural_amino_acids_file']
        sequence_df = dataset.copy()
        sequences_to_exclude_mask = sequence_df['sequence'].apply(self.not_sequences_with_non_natural_amino_acids)
        sequences_to_exclude = sequence_df[sequences_to_exclude_mask]

        if not sequences_to_exclude.empty:
            sequences_to_exclude.to_csv(csv_file, index=False)
            sequence_df = sequence_df.drop(sequences_to_exclude.index)
            logging.getLogger('workflow_logger').\
                warning(f"Sequences with non-natural amino acids were excluded. See: {csv_file}")
        return sequence_df

    def filter_sequences_with_erroneous_activity(self, dataset: pd.DataFrame, output_setting: Dict) \
            -> pd.DataFrame:
        return dataset

    def filter_sequences_with_baseless_partitions(self, dataset: pd.DataFrame, output_setting: Dict) \
            -> pd.DataFrame:
        return dataset

    @staticmethod
    def check_duplicated_sequence_ids(dataset: pd.DataFrame, output_setting: Dict) -> pd.DataFrame:
        csv_file = output_setting['duplicated_sequence_ids_file']
        sequence_df = dataset.copy()
        sequences_to_exclude = sequence_df[sequence_df.duplicated(subset='id', keep=False)]

        if not sequences_to_exclude.empty:
            sequences_to_exclude.sort_values(by='id')
            sequences_to_exclude.to_csv(csv_file, index=False)
            raise ValueError(f"Duplicate sequences IDs. See: {csv_file}")
        return sequence_df

    @staticmethod
    def check_duplicated_sequences(dataset: pd.DataFrame, output_setting: Dict) -> pd.DataFrame:
        csv_file = output_setting['duplicated_sequences_file']
        sequence_df = dataset.copy()
        sequences_to_exclude = sequence_df[sequence_df.duplicated(subset='sequence', keep=False)]

        if not sequences_to_exclude.empty:
            sequences_to_exclude.sort_values(by='sequence')
            sequences_to_exclude.to_csv(csv_file, index=False)
            logging.getLogger('workflow_logger').\
                warning(f"Duplicate sequences. See: {csv_file}")
        return sequence_df


class LabeledDatasetValidator(DatasetValidator):
    def filter_sequences_with_erroneous_activity(self, dataset: pd.DataFrame, output_setting: Dict) \
            -> pd.DataFrame:
        csv_file = output_setting['sequences_with_erroneous_activity_file']
        sequence_df = dataset.copy()
        sequences_to_exclude = sequence_df[~sequence_df['activity'].isin(activities)]

        if not sequences_to_exclude.empty:
            sequences_to_exclude.to_csv(csv_file, index=False)
            sequence_df = sequence_df.drop(sequences_to_exclude.index)
            logging.getLogger('workflow_logger').\
                warning(f"Sequences with erroneous_activity. See: {csv_file}")
        return sequence_df

    def filter_sequences_with_baseless_partitions(self, dataset: pd.DataFrame, output_setting: Dict) \
            -> pd.DataFrame:
        csv_file = output_setting['sequences_with_baseless_partitions_file']
        sequence_df = dataset.copy()
        sequences_to_exclude = sequence_df[~sequence_df['partition'].isin(partitions)]

        if not sequences_to_exclude.empty:
            sequences_to_exclude.to_csv(csv_file, index=False)
            sequence_df = sequence_df.drop(sequences_to_exclude.index)
            logging.getLogger('workflow_logger').\
                warning(f"Sequences with baseless_partitions. See: {csv_file}")
        return sequence_df


class DatasetValidatorContext:
    def __init__(self, dataset_validator: DatasetValidator) -> None:
        self._dataset_validator = dataset_validator

    def processing_dataset(self, dataset: pd.DataFrame, output_setting: Dict):
        return self.dataset_validator.processing_dataset(dataset, output_setting)


if __name__ == '__main__':
    # example: labeled
    labeled_data = {
        'id': ['seq0', 'seq1', 'seq2', 'seq4'],
        'sequence': ['ARNDCQXEG', 'ARNDCQEG', 'RGRRQD', 'ARGGG'],
        'activity': [0, None, 1, 1],
        'partition': [1, 5, 3, 1]
    }
    labeled_data_df = pd.DataFrame(labeled_data)
    
    database_context = DatasetValidatorContext(LabeledDatasetValidator())
    df1 = database_context.processing_dataset(dataset=labeled_data_df, csv_output='.')
    print(df1)

    # example: unlabeled
    unlabeled_data = {
        'id': ['seq0', 'seq1', 'seq2', 'seq4'],
        'sequence': ['ARNDCQXEG', 'ARNDCQEG', 'RGRRQD', 'ARGGG']
    }
    unlabeled_data_df = pd.DataFrame(unlabeled_data)

    database_context2 = DatasetValidatorContext(DatasetValidator())
    df2 = database_context2.processing_dataset(dataset=unlabeled_data_df, csv_output='.')
    print(df2)


