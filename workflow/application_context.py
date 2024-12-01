from injector import Injector
from workflow.classification_metrics import ClassificationMetricsContext, BinaryClassificationMetrics
from workflow.data_loader import DataLoaderContext, CSVLoader, CSVByChunkLoader
from workflow.dataset_validator import DatasetValidatorContext, LabeledDatasetValidator, DatasetValidator
from workflow.path_creator import TrainingModePathCreator, PathCreatorContext, TestModePathCreator, \
    InferenceModePathCreator


class ApplicationContext:
    def __init__(self, mode: str):
        self.__injector = Injector()

        if mode == 'training':
            self.__injector.binder.bind(PathCreatorContext, TrainingModePathCreator)
            self.__injector.binder.bind(DatasetValidatorContext, LabeledDatasetValidator)
            self.__injector.binder.bind(ClassificationMetricsContext, BinaryClassificationMetrics)
            self.__injector.binder.bind(DataLoaderContext, CSVLoader)
        elif mode == 'test':
            self.__injector.binder.bind(PathCreatorContext, TestModePathCreator)
            self.__injector.binder.bind(DatasetValidatorContext, LabeledDatasetValidator)
            self.__injector.binder.bind(ClassificationMetricsContext, BinaryClassificationMetrics)
            self.__injector.binder.bind(DataLoaderContext, CSVLoader)
        elif mode == 'inference':
            self.__injector.binder.bind(PathCreatorContext, InferenceModePathCreator)
            self.__injector.binder.bind(DatasetValidatorContext, DatasetValidator)
            self.__injector.binder.bind(ClassificationMetricsContext, BinaryClassificationMetrics)
            self.__injector.binder.bind(DataLoaderContext, CSVByChunkLoader)

        self.__path_creator = self.__injector.get(PathCreatorContext)
        self.__dataset_validator = self.__injector.get(DatasetValidatorContext)
        self.__data_loader = self.__injector.get(DataLoaderContext)
        self.__classification_metrics = self.__injector.get(ClassificationMetricsContext)

    @property
    def path_creator(self):
        return self.__path_creator

    @property
    def dataset_validator(self):
        return self.__dataset_validator

    @property
    def data_loader(self):
        return self.__data_loader

    @property
    def classification_metrics(self):
        return self.__classification_metrics


if __name__ == '__main__':
    context = ApplicationContext(mode='training')
    pl = context.path_creator
    dv = context.dataset_validator
    dl = context.data_loader







