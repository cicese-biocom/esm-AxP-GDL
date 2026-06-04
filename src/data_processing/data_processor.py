from typing import Dict, List
import pandas as pd
import re
from abc import ABC, abstractmethod
import logging

from src.data_processing.data_partitioner import partition_data_training_and_validation
from src.data_processing.target_feature_validator import TargetFeatureValidator

partitions = [1, 2, 3]
pattern = re.compile('[^ARNDCQEGHILKMFPSTWYV]')


class DatasetProcessor(ABC):       
    def process(
            self,
            dataset: pd.DataFrame,
            output_dir: Dict,
            target_feature_validator: TargetFeatureValidator,
            classes: List[int],
            **kwargs
    ):
        try:
            # In training workflows, if the dataset does not already include
            # explicit partitions, it must be automatically split into
            # training (partition = 1) and validation (partition = 2) sets.
            valid_df = self.filter_sequences_by_partition(dataset, output_dir, **kwargs)

            valid_df = self.check_duplicated_sequence_ids(valid_df, output_dir)
            valid_df = self.check_duplicated_sequences(valid_df, output_dir)
            filtered_df = self.filter_sequences_with_non_natural_amino_acids(valid_df, output_dir)
            self.check_sequences_with_erroneous_activity(filtered_df, output_dir, target_feature_validator, classes)
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
            target_feature_validator: TargetFeatureValidator,
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

    def filter_sequences_by_partition(self, dataset, output_dir, **kwargs):
        return dataset


class LabeledDatasetProcessor(DatasetProcessor):
    def filter_sequences_by_partition(self, dataset, output_dir, **kwargs):
        logger = logging.getLogger("workflow_logger")

        dataset = super().filter_sequences_by_partition(dataset, output_dir)

        if "partition" not in dataset.columns:
            logger.error("Dataset does not contain required column 'partition'.")
            raise ValueError("Dataset does not contain required column 'partition'.")

        return dataset

    def check_sequences_with_erroneous_activity(
            self, dataset: pd.DataFrame,
            output_dir: Dict,
            target_feature_validator: TargetFeatureValidator,
            classes: List = None
    ) -> None:
        csv_file = output_dir['sequences_with_erroneous_activity_file']
        sequence_df = dataset.copy()
        sequences_to_exclude = target_feature_validator.validate(sequence_df, classes)

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


class TrainingDatasetProcessor(LabeledDatasetProcessor):
    def filter_sequences_by_partition(self, dataset, output_dir, **kwargs):
        logger = logging.getLogger("workflow_logger")
        features = kwargs.get('features')
        split_method = kwargs.get('split_method')
        split_training_fraction = kwargs.get('split_training_fraction')

        data = super().filter_sequences_by_partition(dataset, output_dir)
        data = data[data['partition'].isin([1, 2])].reset_index(drop=True)

        # Case 1: Training set (partition = 1) must exist
        if data.query('partition == 1').empty:
            logger.critical("The dataset does not contain a training set (partition = 1).")
            raise ValueError("The dataset does not contain a training set (partition = 1).")

        # Case 2: Validation set (partition = 2) is missing
        if data.query('partition == 2').empty:

            # Auto-split the dataset
            if split_method and split_training_fraction:
                # Remove the 'partition' column so the split method can run
                data.drop(['partition'], axis=1, inplace=True)
                return partition_data_training_and_validation(data, output_dir, features, split_method,
                                                              split_training_fraction)

            # Missing validation and auto-split not configured
            elif not split_method:
                logger.critical(
                    "The dataset does not contain a validation set; the parameter 'split_method' must be specified.")
                raise ValueError(
                    "The dataset does not contain a validation set; the parameter 'split_method' must be specified.")

            elif not split_training_fraction:
                logger.critical(
                    "The dataset does not contain a validation set; "
                    "the parameter 'split_training_fraction' must be specified."
                )
                raise

        # Case 3: Dataset already includes both partitions,
        # but the user incorrectly specifies auto-split parameters
        elif split_method or split_training_fraction:

            if split_method:
                logger.critical(
                    "The dataset already contains training and validation sets; 'split_method' must not be specified.")
                raise ValueError(
                    "The dataset already contains training and validation sets; 'split_method' must not be specified.")

            elif split_training_fraction:
                logger.critical(
                    "The dataset already contains training and validation sets; 'split_training_fraction' must not be specified.")
                raise ValueError(
                    "The dataset already contains training and validation sets; 'split_training_fraction' must not be specified.")

        return data
     

class TestDatasetProcessor(LabeledDatasetProcessor):
    def filter_sequences_by_partition(self, dataset, output_dir, **kwargs):
        logger = logging.getLogger("workflow_logger")

        dataset = super().filter_sequences_by_partition(dataset, output_dir)
        test_data = dataset.query('partition == 3').reset_index(drop=True)

        # Raise an error if no test sequences are found
        if test_data.empty:
            logger.critical(
                "The dataset does not contain a test set (partition = 3)."
            )
            raise ValueError("Missing test partition (3).")

        return test_data


class InferenceDatasetProcessor(DatasetProcessor):
    def filter_sequences_by_partition(self, dataset, output_dir, **kwargs):
        return dataset


class DatasetProcessorContext:
    def __init__(self, dataset_processor: DatasetProcessor) -> None:
        self._dataset_processor = dataset_processor

    def process(self, dataset: pd.DataFrame, output_dir: Dict, target_feature_validator: TargetFeatureValidator, classes: List = None, **kwargs):
        return self._dataset_processor.process(dataset, output_dir, target_feature_validator, classes, **kwargs)
