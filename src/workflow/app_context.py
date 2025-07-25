from injector import Injector

from src.architecture.gnn import GNNFactory
from src.config.types import ModelingTask, ExecutionMode
from src.data_processing.data_loader import DataLoaderContext, CSVLoader, CSVByChunkLoader
from src.data_processing.data_validator import DatasetValidatorContext, LabeledDatasetValidator, DatasetValidator
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
            self.__loss_fn = ml_factory.create_loss()
            self.__prediction_processor = ml_factory.create_prediction_processor()
            self.__metrics = ml_factory.create_metrics(classes=classes)
            self.__best_model_selector = ml_factory.create_best_model_selector()
            self.__class_validator = ml_factory.create_class_validator()
            self.__injector.binder.bind(DatasetValidatorContext, LabeledDatasetValidator)
            self.__injector.binder.bind(DataLoaderContext, CSVLoader)

            # model
            self.__injector.binder.bind(GNNFactory, to=GNNFactory(kwargs.get("gdl_architecture")))
            self.__gnn_factory = self.__injector.get(GNNFactory)

        # TEST
        elif execution_mode == ExecutionMode.TEST:
            self.__prediction_processor = ml_factory.create_prediction_processor()
            self.__metrics = ml_factory.create_metrics(classes=classes)
            self.__class_validator = ml_factory.create_class_validator()
            self.__injector.binder.bind(DatasetValidatorContext, LabeledDatasetValidator)
            self.__injector.binder.bind(DataLoaderContext, to=lambda: CSVByChunkLoader(kwargs.get("prediction_batch_size")))

        # INFERENCE
        elif execution_mode == ExecutionMode.INFERENCE:
            self.__class_validator = ml_factory.create_class_validator()
            self.__injector.binder.bind(DatasetValidatorContext, DatasetValidator)
            self.__injector.binder.bind(DataLoaderContext, to=lambda: CSVByChunkLoader(kwargs.get("prediction_batch_size")))

        self.__dataset_validator = self.__injector.get(DatasetValidatorContext)
        self.__data_loader = self.__injector.get(DataLoaderContext)

    @property
    def metrics(self):
        return self.__metrics

    @property
    def loss_fn(self):
        return self.__loss_fn

    @property
    def prediction_processor(self):
        return self.__prediction_processor

    @property
    def best_model_selector(self):
        return self.__best_model_selector

    @property
    def class_validator(self):
        return self.__class_validator

    @property
    def dataset_validator(self):
        return self.__dataset_validator

    @property
    def data_loader(self):
        return self.__data_loader

    @property
    def gnn_factory(self):
        return self.__gnn_factory


if __name__ == '__main__':
    params = {
        'execution_mode': ExecutionMode.TRAIN,
        'modeling_task': ModelingTask.BINARY_CLASSIFICATION,
    }

    context = ApplicationContext(**params)
    metrics = context.metrics
    loss = context.loss_fn
    best_model_selector = context.best_model_selector
    class_validator = context.class_validator
    dataset_validator = context.dataset_validator
    data_loader = context.data_loader
    gnn = context.gnn_factory

    print("Done!")