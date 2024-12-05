from abc import ABC, abstractmethod
import weka.core.jvm as jvm
from weka.core.dataset import create_instances_from_lists
from weka.clusterers import Clusterer
import pandas as pd
from sklearn.model_selection import train_test_split


class DataPartitioner(ABC):
    @abstractmethod
    def to_partition(self, **kwargs):
        pass


class RandomPartitioner(DataPartitioner):
    def to_partition(self, **kwargs):
        data = kwargs.get('data')
        train_indexes, val_indexes, _, _ = train_test_split(data.index,
                                                            data['activity'],
                                                            split_training_fraction=kwargs.get('split_training_fraction'),
                                                            shuffle=True)

        training = data.drop(val_indexes).assign(partition=lambda x: 1)
        validation = data.drop(train_indexes).assign(partition=lambda x: 2)

        data = pd.concat([training, validation]).sort_index()

        return data


class ExpectationMaximizationPartitioner(DataPartitioner):
    def to_partition(self, **kwargs):
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


def to_partition(split_method: str, **kwargs):
    methods = {
        "random": RandomPartitioner,
        "expectation_maximization": ExpectationMaximizationPartitioner
    }
    data_partitioner = methods[split_method]()
    return data_partitioner.to_partition(**kwargs)
