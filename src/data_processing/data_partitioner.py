from abc import ABC, abstractmethod
from pathlib import Path

import weka.core.jvm as jvm
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from weka.core.dataset import create_instances_from_lists
from weka.clusterers import Clusterer
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.types import SplitMethod
import logging


class DataPartitioner(ABC):
    @abstractmethod
    def split(self, **kwargs):
        pass


class RandomPartitioner(DataPartitioner):
    def split(self, **kwargs):
        data = kwargs.get('data')
        train_indexes, val_indexes, _, _ = train_test_split(data.index,
                                                            data['activity'],
                                                            train_size=kwargs.get('split_training_fraction'),
                                                            shuffle=True)

        training = data.drop(val_indexes).assign(partition=lambda x: 1)
        validation = data.drop(train_indexes).assign(partition=lambda x: 2)

        data = pd.concat([training, validation]).sort_index()

        return data


class ExpectationMaximizationPartitioner(DataPartitioner):
    def split(self, **kwargs):
        try:
            jvm.start(max_heap_size='1g', packages=True, auto_install=True)

            data = kwargs.get('data')
            features = kwargs.get('features')
            split_training_fraction = kwargs.get('split_training_fraction')

            columns = list(data.columns)

            # preparing data
            data_merge = data.merge(features, on='sequence', how='inner')
            data = data_merge.copy(deep=True)
            y = data_merge['activity'].tolist()
            data_merge.drop(columns=columns, inplace=True)
            x = data_merge.values.tolist()

            # clustering with expectation-maximization
            weka_data = create_instances_from_lists(x, y)
            em = Clusterer(classname="weka.clusterers.EM")
            em.build_clusterer(weka_data)

            clusters = []
            for i, instance in enumerate(weka_data):
                cluster = em.cluster_instance(instance)
                clusters.append(cluster)

            data['cluster'] = clusters

            partitioned_data = pd.DataFrame(columns=columns)

            for cluster in range(em.number_of_clusters):
                current_cluster = data[data['cluster'] == cluster]

                if not current_cluster.empty:
                    current_cluster = current_cluster[columns]

                    # split train and val
                    train = current_cluster.groupby('activity').sample(frac=split_training_fraction)
                    val = current_cluster.loc[current_cluster.index.difference(train.index)]

                    # assign partition
                    train['partition'] = 1

                    if not val.empty:
                        val['partition'] = 2

                    partitioned_data = pd.concat([partitioned_data, train, val])

            partitioned_data = partitioned_data.sort_index()
            partitioned_data['partition'] = partitioned_data['partition'].astype(int)

            return partitioned_data
        finally:
            jvm.stop()


def split(split_method: SplitMethod, **kwargs):
    methods = {
        SplitMethod.RANDOM: RandomPartitioner,
        SplitMethod.EXPECTATION_MAXIMIZATION: ExpectationMaximizationPartitioner
    }
    data_partitioner = methods[split_method]()
    return data_partitioner.split(**kwargs)


def partition_data_training_and_validation(dataset: pd.DataFrame, output_dir, features, split_method, split_training_fraction) -> pd.DataFrame:
    # partitioning data in training and validation
    data = split(
        split_method=split_method,
        data=dataset,
        features=features,
        split_training_fraction=split_training_fraction
    )

    filtered_data = data[['id', 'sequence', 'activity', 'partition']]

    # Save training and validation data to csv
    _save_partitioned_data_to_csv(filtered_data, output_dir['data_csv'])

    # Save training data to fasta
    training_data = filtered_data[filtered_data['partition'] == 1]
    _save_partitioned_data_to_fasta(training_data, output_dir['training_data_fasta'])

    # Save validation data to fasta
    validation_data = filtered_data[filtered_data['partition'] == 2]
    _save_partitioned_data_to_fasta(validation_data, output_dir['validation_data_fasta'])

    return data


def _save_partitioned_data_to_csv(data: pd.DataFrame, csv_file: Path):
    data.to_csv(csv_file, index=False)
    logging.getLogger('workflow_logger'). \
        info(f"Partitioned dataset saved to CSV file. See: {csv_file}")


def _save_partitioned_data_to_fasta(df: pd.DataFrame, fasta_file: Path):
    fasta_records = []
    for i, row in df.iterrows():
        sequence_id = row['id']
        sequence = row['sequence']
        activity = int(row['activity'])
        record_id = f"{sequence_id}_class_{activity}"
        record = SeqRecord(Seq(sequence), id=record_id, description="")
        fasta_records.append(record)

    with open(fasta_file, 'w') as output_handle:
        SeqIO.write(fasta_records, output_handle, 'fasta')

    logging.getLogger('workflow_logger'). \
        info(f"Training sequences saved to FASTA file. See: {fasta_file}")