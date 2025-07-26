from enum import Enum

from typing_extensions import Literal


class ExecutionMode(Enum):
    TRAIN = 'TRAIN'
    TEST = 'TEST'
    INFERENCE = 'INFERENCE'

class ModelingTask(Enum):
    BINARY_CLASSIFICATION = 'binary_classification'
    MULTICLASS_CLASSIFICATION = 'multiclass_classification'
    REGRESSION = 'regression'


class Partition(Enum):
    TRAINING = 1
    VALIDATION = 2
    TEST = 3

TertiaryStructurePredictionMethod = Literal['esmfold']


AminoAcidRepresentation = Literal['CA']


class DistanceFunction(Enum):
    EUCLIDEAN = 'euclidean'
    CANBERRA = 'canberra'
    LANCE_WILLIAMS = 'lance_williams'
    CLARK = 'clark'
    SOERGEL = 'soergel'
    BHATTACHARYYA = 'bhattacharyya'
    ANGULAR_SEPARATION = 'angular_separation'


class EdgeConstructionFunction(Enum):
    DISTANCE_BASED_THRESHOLD = 'distance_based_threshold'
    SEQUENCE_BASED = 'sequence_based'
    ESM2_CONTACT_MAP = 'esm2_contact_map'


class ESM2Representation(Enum):
    ESM2_T6 = 'esm2_t6'
    ESM2_T12 = 'esm2_t12'
    ESM2_T30 = 'esm2_t30'
    ESM2_T33 = 'esm2_t33'
    ESM2_T36 = 'esm2_t36'
    ESM2_T48 = 'esm2_t48'
    REDUCED_ESM2_T6 = 'reduced_esm2_t6'
    REDUCED_ESM2_T12 = 'reduced_esm2_t12'
    REDUCED_ESM2_T30 = 'reduced_esm2_t30'
    REDUCED_ESM2_T33 = 'reduced_esm2_t33'
    REDUCED_ESM2_T36 = 'reduced_esm2_t36'
    COMBINED_ESM2 = 'combined_esm2'


class ESM2ModelForContactMap(Enum):
    ESM2_T6 = 'esm2_t6'
    ESM2_T12 = 'esm2_t12'
    ESM2_T30 = 'esm2_t30'
    ESM2_T33 = 'esm2_t33'
    ESM2_T36 = 'esm2_t36'
    ESM2_T48 = 'esm2_t48'

class ValidationMode(Enum):
    RANDOM_COORDINATES = 'random_coordinates'
    RANDOM_EMBEDDINGS = 'random_embeddings'

class SplitMethod(Enum):
    RANDOM = 'random'
    EXPECTATION_MAXIMIZATION  = 'expectation_maximization'

class MethodsForAD(Enum):
    PERCENTILE_BASED_GC = 'percentile_based(gc)'
    PERCENTILE_BASED_PERP = 'percentile_based(perp)'
    IF_PERP_AAD = 'IF(perp_aad)'
    IF_GC = 'IF(gc)'
    IF_GC_PERP_AAD = 'IF(gc_perp_aad)'
    IF_AAD = 'IF(aad)'

class GDLArchitecture(Enum):
    GATV1 = 'GATv1'
    GATV2 = 'GATv2'

