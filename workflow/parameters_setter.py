import logging
from pathlib import Path
from pydantic import BaseModel, FilePath, DirectoryPath, Field, PositiveFloat, PositiveInt, model_validator
from typing import Optional, List, Dict, Union
from typing_extensions import Annotated, Literal, Type
import torch
from workflow.ad_methods_collection_loader import ADMethodCollectionLoader
from workflow.features_collection_loader import FeaturesCollectionLoader
from utils import file_system_handler as file_system_handler


class ParameterSetter(BaseModel):
    mode: Annotated[Optional[Literal['training', 'test', 'inference']], Field(description='Execution mode')] = None
    checkpoint: Annotated[Optional[str], Field(description='Checkpoint_name', exclude=False)] = None

    is_binary_class: Annotated[Optional[bool],
                               Field(strict=True,
                                     description='Indicates whether the task is binary classification')] = True

    numbers_of_class: Annotated[Optional[PositiveInt], Field(description='Numbers of class')] = 2

    device: Annotated[Optional[Type[torch.device]], Field(
        description='The device to be used for computation (e.g., "cuda:0" for GPU or "cpu"', exclude=True)] = \
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset: Annotated[FilePath, Field(description='Path to the input dataset in CSV format', exclude=True)]

    dataset_name: Annotated[Optional[str], Field(description='Name dataset')] = None
    dataset_extension: Annotated[Optional[str], Field(description='Name extension')] = None

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

    pdb_path: Annotated[Optional[DirectoryPath],
                        Field(description='Path where tertiary structures are saved in or loaded from PDB files',
                              exclude=True)] = None

    edge_construction_functions: Annotated[List[Literal['distance_based_threshold', 'sequence_based',
                                                        'esm2_contact_map']],
                                           Field(description='Functions to build edges')]

    distance_function: Annotated[Optional[Literal['euclidean', 'canberra', 'lance_williams', 'clark', 'soergel',
                                                  'bhattacharyya', 'angular_separation']],
                                 Field(
                                     description='Distance function to construct the edges of the distance-based graph')] = None

    distance_threshold: Annotated[Optional[PositiveFloat],
                                  Field(
                                      description='Distance threshold to construct the edges of the distance-based graph')] = None

    esm2_model_for_contact_map: Annotated[Optional[Literal['esm2_t6', 'esm2_t12', 'esm2_t30', 'esm2_t33', 'esm2_t36',
                                                           'esm2_t48']],
                                          Field(description='ESM-2 model to be used')] = None

    probability_threshold: Annotated[Optional[PositiveFloat],
                                     Field(
                                         description='Probability threshold for constructing a graph based on ESM-2 contact maps',
                                         ge=0, le=1)] = None

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

    gdl_model_path: Annotated[Path, Field(description='Path where trained models are saved, or from where a trained '
                                                      'model is loaded for test/inference mode', exclude=True)]

    save_ckpt_per_epoch: Annotated[Optional[bool],
                                   Field(strict=True,
                                         description='True if specified, otherwise, False. '
                                                     'True indicates to save the models per epoch.')] = True

    validation_mode: Annotated[Optional[Literal['random_coordinates', 'random_embeddings']],
                               Field(
                                   description='Graph construction method to validate the performance of the models')] = None

    randomness_percentage: Annotated[Optional[PositiveInt],
                                     Field(description='Percentage of rows to be randomly created', ge=0,
                                           le=100)] = None

    feature_types_for_ad: Annotated[
        Optional[Union[
            List[Literal['graph_centralities', 'perplexity', 'amino_acid_descriptors']],
            List[Dict]
        ]],
        Field(
            description='Feature groups used to build the applicability domain (AD) model. Accepts a list of feature names (literals) or a list of feature dictionaries with additional details.')
    ] = None

    methods_for_ad: Annotated[
        Optional[Union[
            List[Literal['percentile_based', 'isolation_forest', 'clustering_and_isolation_forest']],
            List[Dict]
        ]],
        Field(
            description='Methods available for calculating the applicability domain (AD). Accepts a list of method names (literals) or a list of method dictionaries with additional details.')
    ] = None

    feature_file_for_ad: Annotated[
        Optional[FilePath],
        Field(description='Path to the file containing features for calculating the applicability domain (AD).')
    ] = None

    output_setting: Annotated[Optional[Dict], Field(description='Output settings', exclude=True)] = None

    seed: Annotated[Optional[PositiveInt],
                    Field(description='Percentage of rows to be scrambling')] = None

    get_ad: Annotated[Optional[bool], Field(description='Get Applicability Domain')] = False

    @model_validator(mode='after')
    def validator(self) -> 'ParameterSetter':
        try:
            # device
            logging.getLogger('workflow_logger').info(f"Device: {self.device}")

            self.gdl_model_path = self.gdl_model_path.resolve()
            self.checkpoint = str(self.gdl_model_path.name)

            # database
            self.dataset = file_system_handler.check_file_exists(self.dataset)
            self.dataset_name = self.dataset.name
            self.dataset_extension = file_system_handler.check_file_format(self.dataset)

            # pdb_path
            if self.pdb_path:
                if 'distance_based_threshold' in self.edge_construction_functions \
                        or 'sequence_based' in self.edge_construction_functions \
                        and self.distance_function is not None:
                    self.pdb_path = file_system_handler.check_file_exists(self.pdb_path)
                else:
                    self.pdb_path = None
                    self.tertiary_structure_method = None
                    logging.getLogger('workflow_logger').warning(
                        f"The specified edge construction function does not require tertiary structures. "
                        f"The parameters pdb_path and tertiary_structure_method were set to None.")

            # distance_function
            if self.distance_function is None:
                if 'distance_based_threshold' in self.edge_construction_functions:
                    raise ValueError('Parameter distance_function is required')
            else:
                if not {'distance_based_threshold', 'sequence_based'}.intersection(self.edge_construction_functions):
                    self.distance_function = None
                    logging.getLogger('workflow_logger').warning(
                        f"Edge construction methods {self.edge_construction_functions} "
                        f"do not require the parameter distance_function. It has been set to None")

            # distance_threshold
            if self.distance_threshold is None:
                if 'distance_based_threshold' in self.edge_construction_functions:
                    raise ValueError('Parameter distance_threshold is required')
            else:
                if not {'distance_based_threshold'}.intersection(self.edge_construction_functions):
                    self.distance_threshold = None
                    logging.getLogger('workflow_logger').warning(
                        f"Edge construction methods {self.edge_construction_functions} "
                        f"do not require the parameter distance_threshold. It has been set to None")

            # esm2_model_for_contact_map
            if self.esm2_model_for_contact_map is None:
                if 'esm2_contact_map' in self.edge_construction_functions:
                    raise ValueError('Parameter esm2_model_for_contact_map is required')
            else:
                if not {'esm2_contact_map'}.intersection(self.edge_construction_functions):
                    self.esm2_model_for_contact_map = None
                    logging.getLogger('workflow_logger').warning(
                        f"Edge construction methods {self.edge_construction_functions} "
                        f"do not require the parameter esm2_model_for_contact_map. It has been set to None")

            # probability_threshold
            if self.probability_threshold is None:
                if 'esm2_contact_map' in self.edge_construction_functions:
                    raise ValueError('Parameter probability_threshold is required')
            else:
                if not {'esm2_contact_map'}.intersection(self.edge_construction_functions):
                    self.probability_threshold = None
                    logging.getLogger('workflow_logger').warning(
                        f"Edge construction methods {self.edge_construction_functions} "
                        f"do not require the parameter probability_threshold. It has been set to None")

            # use_edge_attr
            if self.use_edge_attr:
                if not {'distance_based_threshold', 'esm2_contact_map'}.intersection(self.edge_construction_functions) and 'sequence_based' in self.edge_construction_functions and self.distance_function is None:
                    self.use_edge_attr = False
                    logging.getLogger('workflow_logger').warning(
                        f"Edge construction methods {self.edge_construction_functions} "
                        f"do not generate weight matrices. Parameter use_edge_attr has been set to False")

            # validation_mode
            if self.validation_mode and not self.randomness_percentage:
                raise ValueError('Parameter randomness_percentage is required')

            if not self.validation_mode:
                self.randomness_percentage = None

            # applicability domain
            none_params = [
                param for param in [
                    'feature_types_for_ad',
                    'methods_for_ad',
                    'feature_file_for_ad'
                ] if getattr(self, param) is None
            ]

            if len(none_params) == 0:
                self.get_ad = True
            elif len(none_params) == 3:
                self.get_ad = False
            else:
                logging.getLogger('workflow_logger').critical(
                    f"The following parameters must be specified: {', '.join(none_params)}"
                )
                quit()

            features_collection = FeaturesCollectionLoader()
            ad_methods_collection = ADMethodCollectionLoader()

            if self.mode in 'training':
                self.feature_types_for_ad = features_collection.get_all_features()
            elif self.mode in ('test', 'inference') and self.get_ad:
                self.feature_types_for_ad = features_collection.get_features_by_name(feature_names=self.feature_types_for_ad)
                self.methods_for_ad = ad_methods_collection.get_methods_with_features(method_for_ad=self.methods_for_ad,
                                                                                      features_for_ad=self.feature_types_for_ad)

            # parameters that are only used in training mode
            if self.mode in ('test', 'inference'):
                self.number_of_epochs = None
                self.learning_rate = None
                self.pooling_ratio = None

            return self
        except Exception as e:
            logging.getLogger('workflow_logger').critical(e)
            quit()
