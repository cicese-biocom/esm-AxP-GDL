from pydantic import BaseModel, Field, PositiveInt, PositiveFloat
from typing import Optional, Set, List
from typing_extensions import Annotated, Literal


class ModelParameters(BaseModel):
    esm2_representation: Annotated[Optional[Literal['esm2_t6', 'esm2_t12', 'esm2_t30', 'esm2_t33', 'esm2_t36',
                                                    'esm2_t48']],
                                   Field(description='ESM-2 model to be used')] = 'esm2_t33'

    tertiary_structure_method: Annotated[Optional[Literal['esmfold']],
                                         Field(description='3D structure prediction method. None indicates to load '
                                                           'existing tertiary structures from PDB files, otherwise, '
                                                           'sequences in input CSV file are predicted using the '
                                                           'specified method')] = None

    edge_construction_functions: Annotated[List[Literal['distance_based_threshold', 'sequence_based',
                                                        'esm2_contact_map_50', 'esm2_contact_map_60',
                                                        'esm2_contact_map_70', 'esm2_contact_map_80',
                                                        'esm2_contact_map_90']],
                                           Field(description='Functions to build edges')]

    distance_function: Annotated[Optional[Literal['euclidean', 'canberra', 'lance_williams', 'clark', 'soergel',
                                                  'bhattacharyya', 'angular_separation']],
                                 Field(description='Distance function to construct graph edges')] = None

    distance_threshold: Annotated[Optional[PositiveFloat],
                                  Field(description='Distance threshold to construct graph edges')] = None

    amino_acid_representation: Annotated[Optional[Literal['CA']],
                                         Field(description='Amino acid representations')] = 'CA'

    hidden_layer_dimension: Annotated[Optional[PositiveInt],
                                      Field(description='Hidden layer dimension')] = 128

    add_self_loops: Annotated[Optional[bool],
                              Field(strict=True,
                                    description='True if specified, otherwise, False. True indicates to use auto loops '
                                                'in attention layer')] = False

    use_edge_attr: Annotated[Optional[bool], Field(strict=True,
                                                   description='True if specified, otherwise, False. True indicates to '
                                                               'use edge attributes in graph learning.')] = False

    dropout_rate: Annotated[Optional[PositiveFloat], Field(default=0.25, description='Dropout rate')] = 0.25

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
                                                    'esm2_t48']],
                                   Field(description='ESM-2 model to be used')] = 'esm2_t33'

    tertiary_structure_method: Annotated[Optional[Literal['esmfold']],
                                         Field(description='3D structure prediction method. None indicates to load '
                                                           'existing tertiary structures from PDB files, otherwise, '
                                                           'sequences in input CSV file are predicted using the '
                                                           'specified method')] = None

    edge_construction_functions: Annotated[List[Literal['distance_based_threshold', 'sequence_based',
                                                        'esm2_contact_map_50', 'esm2_contact_map_60',
                                                        'esm2_contact_map_70', 'esm2_contact_map_80',
                                                        'esm2_contact_map_90']],
                                           Field(description='Functions to build edges')]

    use_esm2_contact_map: Annotated[Optional[bool],
                                    Field(strict=True,
                                          description='Use esm2 contact map')] = False

    distance_function: Annotated[Optional[Literal['euclidean', 'canberra', 'lance_williams', 'clark', 'soergel',
                                                  'bhattacharyya', 'angular_separation']],
                                 Field(description='Distance function to construct graph edges')] = None

    distance_threshold: Annotated[Optional[PositiveFloat],
                                  Field(description='Distance threshold to construct graph edges')] = None

    amino_acid_representation: Annotated[Optional[Literal['CA']],
                                         Field(description='Amino acid representations')] = 'CA'

    number_of_heads: Annotated[Optional[PositiveInt],
                               Field(description='Number of heads')] = 8

    hidden_layer_dimension: Annotated[Optional[PositiveInt],
                                      Field(description='Hidden layer dimension')] = 128

    add_self_loops: Annotated[Optional[bool],
                              Field(strict=True,
                                    description='True if specified, otherwise, False. True indicates to use auto loops '
                                                'in attention layer')] = True

    use_edge_attr: Annotated[Optional[bool], Field(strict=True,
                                                   description='True if specified, otherwise, False. True indicates to '
                                                               'use edge attributes in graph learning.')] = True

    pooling_ratio: Annotated[Optional[PositiveInt], Field(description='Pooling ratio')] = 10

    learning_rate: Annotated[Optional[PositiveFloat], Field(description='Learning rate')] = 1e-4

    dropout_rate: Annotated[Optional[PositiveFloat], Field(default=0.25, description='Dropout rate')] = 0.25

    batch_size: Annotated[Optional[PositiveInt], Field(description='Batch size')] = 512

    number_of_epochs: Annotated[Optional[PositiveInt], Field(description='Maximum number of epochs')] = 200

    save_ckpt_per_epoch: Annotated[Optional[bool],
                                   Field(strict=True,
                                         description='True if specified, otherwise, False. '
                                                     'True indicates to save the models per epoch.')] = True

    validation_mode: Annotated[Optional[Literal['random_coordinates', 'random_embeddings']],
                               Field(description='Graph construction method to validate the performance of the models')] = None

    randomness_percentage: Annotated[Optional[PositiveInt],
                                     Field(description='Percentage of rows to be randomly created')] = None


class EvalOutputParameter(BaseModel):
    mode: Annotated[Optional[str], Field(description='Execution mode')] = None
    checkpoint: Annotated[Optional[str], Field(description='Checkpoint_name', exclude=False)] = None

    is_binary_class: Annotated[Optional[bool],
                               Field(strict=True,
                                     description='Indicates whether the task is binary classification')] = True

    numbers_of_class: Annotated[Optional[PositiveInt], Field(description='Numbers of class')] = 2

    dataset_name: Annotated[Optional[str], Field(description='Name dataset')] = None

    esm2_representation: Annotated[Optional[Literal['esm2_t6', 'esm2_t12', 'esm2_t30', 'esm2_t33', 'esm2_t36',
                                                    'esm2_t48']],
                                   Field(description='ESM-2 model to be used')] = 'esm2_t33'

    tertiary_structure_method: Annotated[Optional[Literal['esmfold']],
                                         Field(description='3D structure prediction method. None indicates to load '
                                                           'existing tertiary structures from PDB files, otherwise, '
                                                           'sequences in input CSV file are predicted using the '
                                                           'specified method')] = None

    edge_construction_functions: Annotated[List[Literal['distance_based_threshold', 'sequence_based',
                                                        'esm2_contact_map_50', 'esm2_contact_map_60',
                                                        'esm2_contact_map_70', 'esm2_contact_map_80',
                                                        'esm2_contact_map_90']],
                                           Field(description='Functions to build edges')]

    use_esm2_contact_map: Annotated[Optional[bool],
                                    Field(strict=True,
                                          description='Use esm2 contact map')] = False

    distance_function: Annotated[Optional[Literal['euclidean', 'canberra', 'lance_williams', 'clark', 'soergel',
                                                  'bhattacharyya', 'angular_separation']],
                                 Field(description='Distance function to construct graph edges')] = None

    distance_threshold: Annotated[Optional[PositiveFloat],
                                  Field(description='Distance threshold to construct graph edges')] = None

    amino_acid_representation: Annotated[Optional[Literal['CA']],
                                         Field(description='Amino acid representations')] = 'CA'

    add_self_loops: Annotated[Optional[bool],
                              Field(strict=True,
                                    description='True if specified, otherwise, False. True indicates to use auto loops '
                                                'in attention layer')] = True

    use_edge_attr: Annotated[Optional[bool], Field(strict=True,
                                                   description='True if specified, otherwise, False. True indicates to '
                                                               'use edge attributes in graph learning.')] = True

    dropout_rate: Annotated[Optional[PositiveFloat], Field(default=0.25, description='Dropout rate')] = 0.25

    validation_mode: Annotated[Optional[Literal['random_coordinates', 'random_embeddings']],
                               Field(description='Graph construction method to validate the performance of the models')] = None

    randomness_percentage: Annotated[Optional[PositiveInt],
                                     Field(description='Percentage of rows to be randomly created')] = None

    seed: Annotated[Optional[PositiveInt],
                    Field(description='Percentage of rows to be scrambling')] = None
