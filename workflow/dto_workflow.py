from pydantic import BaseModel, Field, PositiveInt, PositiveFloat
from typing import Optional, List, Dict
from typing_extensions import Annotated, Literal


class ModelParameters(BaseModel):
    esm2_representation: Annotated[Optional[Literal['esm2_t6', 'esm2_t12', 'esm2_t30', 'esm2_t33', 'esm2_t36',
                                                    'esm2_t48', 'reduced_esm2_t6', 'reduced_esm2_t12',
                                                    'reduced_esm2_t30', 'reduced_esm2_t33', 'reduced_esm2_t36',
                                                    'combined_esm2']],
                                   Field(description='ESM-2 representation to be used')] = None

    tertiary_structure_method: Annotated[Optional[Literal['esmfold']],
                                         Field(description='3D structure prediction method. None indicates to load '
                                                           'existing tertiary structures from PDB files, otherwise, '
                                                           'sequences in input CSV file are predicted using the '
                                                           'specified method')] = None

    edge_construction_functions: Annotated[List[Literal['distance_based_threshold', 'sequence_based',
                                                        'esm2_contact_map']],
                                           Field(description='Functions to build edges')]

    distance_function: Annotated[Optional[Literal['euclidean', 'canberra', 'lance_williams', 'clark', 'soergel',
                                                  'bhattacharyya', 'angular_separation']],
                                 Field(description='Distance function to construct graph edges')] = None

    distance_threshold: Annotated[Optional[PositiveFloat],
                                  Field(description='Distance threshold to construct graph edges')] = None

    esm2_model_for_contact_map: Annotated[Optional[Literal['esm2_t6', 'esm2_t12', 'esm2_t30', 'esm2_t33', 'esm2_t36',
                                                           'esm2_t48']],
                                          Field(description='ESM-2 model to be used')] = None

    probability_threshold: Annotated[Optional[PositiveFloat],
                                     Field(
                                         description='Probability threshold for constructing a graph based on ESM-2 contact maps',
                                         ge=0, le=1)] = None

    amino_acid_representation: Annotated[Optional[Literal['CA']],
                                         Field(description='Amino acid representations')] = None

    hidden_layer_dimension: Annotated[Optional[PositiveInt],
                                      Field(description='Hidden layer dimension')] = None

    add_self_loops: Annotated[Optional[bool],
                              Field(strict=True,
                                    description='True if specified, otherwise, False. True indicates to use auto loops '
                                                'in attention layer')] = None

    use_edge_attr: Annotated[Optional[bool], Field(strict=True,
                                                   description='True if specified, otherwise, False. True indicates to '
                                                               'use edge attributes in graph learning.')] = None

    dropout_rate: Annotated[Optional[PositiveFloat], Field(default=0.25, description='Dropout rate')] = None

    validation_mode: Annotated[Optional[Literal['random_coordinates', 'random_embeddings']],
                               Field(description='Graph construction method to validate the performance of the models')] = None

    randomness_percentage: Annotated[Optional[PositiveInt],
                                     Field(description='Percentage of rows to be randomly created')] = None


class TrainingOutputParameter(BaseModel):
    mode: Annotated[Optional[str], Field(description='Execution mode')] = None

    is_binary_class: Annotated[Optional[bool],
                               Field(strict=True,
                                     description='Indicates whether the task is binary classification')] = True

    dataset_name: Annotated[Optional[str], Field(description='Name dataset')] = None

    esm2_representation: Annotated[Optional[Literal['esm2_t6', 'esm2_t12', 'esm2_t30', 'esm2_t33', 'esm2_t36',
                                                    'esm2_t48', 'reduced_esm2_t6', 'reduced_esm2_t12',
                                                    'reduced_esm2_t30', 'reduced_esm2_t33', 'reduced_esm2_t36',
                                                    'combined_esm2']],
                                   Field(description='ESM-2 representation to be used')] = None

    tertiary_structure_method: Annotated[Optional[Literal['esmfold']],
                                         Field(description='3D structure prediction method. None indicates to load '
                                                           'existing tertiary structures from PDB files, otherwise, '
                                                           'sequences in input CSV file are predicted using the '
                                                           'specified method')] = None

    edge_construction_functions: Annotated[List[Literal['distance_based_threshold', 'sequence_based',
                                                        'esm2_contact_map']],
                                           Field(description='Functions to build edges')]

    distance_function: Annotated[Optional[Literal['euclidean', 'canberra', 'lance_williams', 'clark', 'soergel',
                                                  'bhattacharyya', 'angular_separation']],
                                 Field(description='Distance function to construct graph edges')] = None

    distance_threshold: Annotated[Optional[PositiveFloat],
                                  Field(description='Distance threshold to construct graph edges')] = None

    esm2_model_for_contact_map: Annotated[Optional[Literal['esm2_t6', 'esm2_t12', 'esm2_t30', 'esm2_t33', 'esm2_t36',
                                                           'esm2_t48']],
                                          Field(description='ESM-2 model to be used')] = None

    probability_threshold: Annotated[Optional[PositiveFloat],
                                     Field(
                                         description='Probability threshold for constructing a graph based on ESM-2 contact maps',
                                         ge=0, le=1)] = None

    amino_acid_representation: Annotated[Optional[Literal['CA']],
                                         Field(description='Amino acid representations')] = None

    number_of_heads: Annotated[Optional[PositiveInt],
                               Field(description='Number of heads')] = None

    hidden_layer_dimension: Annotated[Optional[PositiveInt],
                                      Field(description='Hidden layer dimension')] = None

    add_self_loops: Annotated[Optional[bool],
                              Field(strict=True,
                                    description='True if specified, otherwise, False. True indicates to use auto loops '
                                                'in attention layer')] = None

    use_edge_attr: Annotated[Optional[bool], Field(strict=True,
                                                   description='True if specified, otherwise, False. True indicates to '
                                                               'use edge attributes in graph learning.')] = None

    pooling_ratio: Annotated[Optional[PositiveInt], Field(description='Pooling ratio')] = None

    learning_rate: Annotated[Optional[PositiveFloat], Field(description='Learning rate')] = None

    dropout_rate: Annotated[Optional[PositiveFloat], Field(default=0.25, description='Dropout rate')] = None

    batch_size: Annotated[Optional[PositiveInt], Field(description='Batch size')] = None

    number_of_epochs: Annotated[Optional[PositiveInt], Field(description='Maximum number of epochs')] = None

    save_ckpt_per_epoch: Annotated[Optional[bool],
                                   Field(strict=True,
                                         description='True if specified, otherwise, False. '
                                                     'True indicates to save the models per epoch.')] = None

    validation_mode: Annotated[Optional[Literal['random_coordinates', 'random_embeddings']],
                               Field(description='Graph construction method to validate the performance of the models')] = None

    randomness_percentage: Annotated[Optional[PositiveInt],
                                     Field(description='Percentage of rows to be randomly created')] = None


class EvalOutputParameter(BaseModel):
    mode: Annotated[Optional[str], Field(description='Execution mode')] = None
    checkpoint: Annotated[Optional[str], Field(description='Checkpoint_name', exclude=False)] = None

    is_binary_class: Annotated[Optional[bool],
                               Field(strict=True,
                                     description='Indicates whether the task is binary classification')] = None

    numbers_of_class: Annotated[Optional[PositiveInt], Field(description='Numbers of class')] = 2

    dataset_name: Annotated[Optional[str], Field(description='Name dataset')] = None

    esm2_representation: Annotated[Optional[Literal['esm2_t6', 'esm2_t12', 'esm2_t30', 'esm2_t33', 'esm2_t36',
                                                    'esm2_t48', 'reduced_esm2_t6', 'reduced_esm2_t12',
                                                    'reduced_esm2_t30', 'reduced_esm2_t33', 'reduced_esm2_t36',
                                                    'combined_esm2']],
                                   Field(description='ESM-2 representation to be used')] = None

    tertiary_structure_method: Annotated[Optional[Literal['esmfold']],
                                         Field(description='3D structure prediction method. None indicates to load '
                                                           'existing tertiary structures from PDB files, otherwise, '
                                                           'sequences in input CSV file are predicted using the '
                                                           'specified method')] = None

    edge_construction_functions: Annotated[List[Literal['distance_based_threshold', 'sequence_based',
                                                        'esm2_contact_map']],
                                           Field(description='Functions to build edges')]

    distance_function: Annotated[Optional[Literal['euclidean', 'canberra', 'lance_williams', 'clark', 'soergel',
                                                  'bhattacharyya', 'angular_separation']],
                                 Field(description='Distance function to construct graph edges')] = None

    distance_threshold: Annotated[Optional[PositiveFloat],
                                  Field(description='Distance threshold to construct graph edges')] = None

    esm2_model_for_contact_map: Annotated[Optional[Literal['esm2_t6', 'esm2_t12', 'esm2_t30', 'esm2_t33', 'esm2_t36',
                                                           'esm2_t48']],
                                          Field(description='ESM-2 model to be used')] = None

    probability_threshold: Annotated[Optional[PositiveFloat],
                                     Field(
                                         description='Probability threshold for constructing a graph based on ESM-2 contact maps',
                                         ge=0, le=1)] = None

    amino_acid_representation: Annotated[Optional[Literal['CA']],
                                         Field(description='Amino acid representations')] = None

    add_self_loops: Annotated[Optional[bool],
                              Field(strict=True,
                                    description='True if specified, otherwise, False. True indicates to use auto loops '
                                                'in attention layer')] = None

    use_edge_attr: Annotated[Optional[bool], Field(strict=True,
                                                   description='True if specified, otherwise, False. True indicates to '
                                                               'use edge attributes in graph learning.')] = None

    dropout_rate: Annotated[Optional[PositiveFloat], Field(default=0.25, description='Dropout rate')] = None

    validation_mode: Annotated[Optional[Literal['random_coordinates', 'random_embeddings']],
                               Field(description='Graph construction method to validate the performance of the models')] = None

    randomness_percentage: Annotated[Optional[PositiveInt],
                                     Field(description='Percentage of rows to be randomly created')] = None

    seed: Annotated[Optional[PositiveInt],
                    Field(description='Percentage of rows to be scrambling')] = None

    feature_types_for_ad: Annotated[
        Optional[List[Dict]],
        Field(
            description='Feature groups used to build the applicability domain (AD) model. Accepts a list of feature names (literals) or a list of feature dictionaries with additional details.')
    ] = None

    methods_for_ad: Annotated[
        Optional[List[Dict]],
        Field(
            description='Methods available for calculating the applicability domain (AD). Accepts a list of method names (literals) or a list of method dictionaries with additional details.')
    ] = None
