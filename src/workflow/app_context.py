from injector import Injector

from src.architectures.gnn import GNNFactory
from src.config.types import ModelingTask, ExecutionMode
from src.data_processing.data_loader import DataLoaderContext, CSVLoader, CSVByChunkLoader
from src.data_processing.data_processor import DatasetProcessorContext, LabeledDatasetProcessor, DatasetProcessor
from src.workflow.execution_factory import (
    ExecutionFactory,
    BinaryExecutionFactory,
    MulticlassExecutionFactory,
    RegressionExecutionFactory,
)


class ApplicationContext:
    def __init__(self, **kwargs):
        execution_mode = kwargs.get('execution_mode')
        modeling_task = kwargs.get('modeling_task')
        classes = kwargs.get('classes')

        self.__injector = Injector()

        # BINARY
        if modeling_task == ModelingTask.BINARY_CLASSIFICATION:
            self.__injector.binder.bind(ExecutionFactory, BinaryExecutionFactory)

        # MULTICLASS
        elif modeling_task == ModelingTask.MULTICLASS_CLASSIFICATION:
            self.__injector.binder.bind(ExecutionFactory, MulticlassExecutionFactory)

        # REGRESSION
        elif modeling_task == ModelingTask.REGRESSION:
            self.__injector.binder.bind(ExecutionFactory, RegressionExecutionFactory)

        ml_factory = self.__injector.get(ExecutionFactory)

        self.__gnn_factory = None
        # TRAIN
        if execution_mode == ExecutionMode.TRAIN:
            self.__loss_fn = ml_factory.create_loss_fn()
            self.__prediction_maker = ml_factory.create_prediction_maker()
            self.__metrics_calculator = ml_factory.create_metrics_calculator(prediction_maker=self.__prediction_maker, classes=classes)
            self.__model_selector = ml_factory.create_model_selector()
            self.__target_feature_validator = ml_factory.create_target_feature_validator()
            self.__injector.binder.bind(DatasetProcessorContext, LabeledDatasetProcessor)
            self.__injector.binder.bind(DataLoaderContext, CSVLoader)

            # model
            self.__injector.binder.bind(GNNFactory, to=GNNFactory(kwargs.get("gdl_architecture")))
            self.__gnn_factory = self.__injector.get(GNNFactory)

        # TEST
        elif execution_mode == ExecutionMode.TEST:
            self.__prediction_maker = ml_factory.create_prediction_maker()
            self.__metrics_calculator = ml_factory.create_metrics_calculator(prediction_maker=self.__prediction_maker, classes=classes)
            self.__target_feature_validator = ml_factory.create_target_feature_validator()
            self.__injector.binder.bind(DatasetProcessorContext, LabeledDatasetProcessor)
            self.__injector.binder.bind(DataLoaderContext, to=lambda: CSVByChunkLoader(kwargs.get("prediction_batch_size")))

        # INFERENCE
        elif execution_mode == ExecutionMode.INFERENCE:
            self.__prediction_maker = ml_factory.create_prediction_maker()
            self.__target_feature_validator = ml_factory.create_target_feature_validator()
            self.__injector.binder.bind(DatasetProcessorContext, DatasetProcessor)
            self.__injector.binder.bind(DataLoaderContext, to=lambda: CSVByChunkLoader(kwargs.get("prediction_batch_size")))

        self.__dataset_processor = self.__injector.get(DatasetProcessorContext)
        self.__dataset_loader = self.__injector.get(DataLoaderContext)

    @property
    def metrics_calculator(self):
        return self.__metrics_calculator

    @property
    def loss_fn(self):
        return self.__loss_fn

    @property
    def prediction_maker(self):
        return self.__prediction_maker

    @property
    def model_selector(self):
        return self.__model_selector

    @property
    def target_feature_validator(self):
        return self.__target_feature_validator

    @property
    def dataset_processor(self):
        return self.__dataset_processor

    @property
    def dataset_loader(self):
        return self.__dataset_loader

    @property
    def gnn_factory(self):
        return self.__gnn_factory


if __name__ == '__main__':
    build_graphs_parameters = {
        'execution_mode': ExecutionMode.TRAIN,
        'modeling_task': ModelingTask.BINARY_CLASSIFICATION,
    }

    context = ApplicationContext(**build_graphs_parameters)
    metrics = context.metrics
    loss = context.loss_fn
    model_selector = context.model_selector
    target_feature_validator = context.target_feature_validator
    dataset_calculateor = context.dataset_calculateor
    dataset_loader = context.dataset_loader
    gnn = context.gnn_factory

    print("Done!")