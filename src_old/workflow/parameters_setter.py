import logging
import os
from pathlib import Path
import random
import numpy as np
from pydantic import BaseModel, FilePath, Field, PositiveFloat, PositiveInt, model_validator
from typing import Optional, List, Dict, Union, Set
from typing_extensions import Annotated, Literal, Type
import torch
from src_old.workflow.ad_methods_collection_loader import ADMethodCollectionLoader
from src_old.workflow.features_collection_loader import FeaturesCollectionLoader
from src_old.utils import file_system_handler as file_system_handler


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

    pdb_path: Annotated[Optional[Path],
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
                                         ge=0.5, le=1.0)] = None

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

    inference_batch_size: Annotated[Optional[PositiveInt], Field(description='Dataset batch size')] = 20000

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
            List[Dict]
        ]],
        Field(
            description='Feature groups used to build the applicability domain (AD) model.')
    ] = None

    methods_for_ad: Annotated[
        Optional[Union[
            Set[str],
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

    from pydantic import conint

    seed: Annotated[Optional[conint(ge=0)],
                    Field(description='Seed to run the Test/Inference mode')] = None

    get_ad: Annotated[Optional[bool], Field(description='Get Applicability Domain')] = False

    split_method: Annotated[Optional[Literal['random', 'expectation_maximization']],
                            Field(description='Method for data partition')] = None

    split_training_fraction: Annotated[Optional[PositiveFloat], Field(description='Train size', ge=0.6, le=0.9)] = None

    @model_validator(mode='after')
    def validator(self) -> 'ParameterSetter':
        try:
            # device
            logging.getLogger('workflow_logger').info(f"Device: {self.device}")

            # seed
            if self.seed is not None:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

                torch.manual_seed(self.seed)
                random.seed(self.seed)
                np.random.seed(self.seed)
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

                # The option warn_only=True is used in the call to torch.use_deterministic_algorithms(True, warn_only=True)
                # to ensure that PyTorch attempts to apply deterministic algorithms whenever possible, but without
                # interrupting execution if an operation does not have a deterministic implementation. This setting is
                # especially useful in the current context, as some functions like scatter_add_cuda_kernel (used internally
                # by PyTorch Geometric) do not have deterministic versions available.
                torch.use_deterministic_algorithms(True, warn_only=True)

            self.gdl_model_path = self.gdl_model_path.resolve()
            self.checkpoint = str(self.gdl_model_path.name)

            # database
            self.dataset = file_system_handler.check_file_exists(self.dataset)
            self.dataset_name = self.dataset.name
            self.dataset_extension = file_system_handler.check_file_format(self.dataset)

            # pdb path
            if self.tertiary_structure_method:
                self.pdb_path = self.pdb_path.resolve()
                self.pdb_path.mkdir(parents=True, exist_ok=True)

            #
            if 'distance_based_threshold' in self.edge_construction_functions \
                    or ('sequence_based' in self.edge_construction_functions and self.distance_function):
                if not self.amino_acid_representation:
                    self.amino_acid_representation = 'CA'

                if self.pdb_path:
                    self.pdb_path = file_system_handler.check_file_exists(self.pdb_path)
                else:
                    raise ValueError(
                        f"The parameter 'pdb_path is require."
                    )
            else:
                if self.pdb_path:
                    raise ValueError(
                        f"The edge construction methods '{','.join(self.edge_construction_functions)}' "
                        f"do not require the parameter 'pdb_path'."
                    )
                elif self.tertiary_structure_method:
                    raise ValueError(
                        f"The edge construction methods '{','.join(self.edge_construction_functions)}' "
                        f"do not require the parameter 'tertiary_structure_method'."
                    )
                elif self.amino_acid_representation:
                    raise ValueError(
                        f"The edge construction methods '{','.join(self.edge_construction_functions)}' "
                        f"do not require the parameter 'amino_acid_representation'."
                    )
                if self.validation_mode == 'random_coordinates':
                    raise ValueError(
                        f"The parameter 'validation_mode' cannot be 'random_coordinates'. "
                        f"The edge construction methods '{','.join(self.edge_construction_functions)}' "
                        f"do not use 3D coordinates."
                    )

            # distance_function
            if self.distance_function is None:
                if 'distance_based_threshold' in self.edge_construction_functions:
                    raise ValueError(
                        f"The parameter 'distance_function' is required."
                    )
            else:
                if not {'distance_based_threshold', 'sequence_based'}.intersection(self.edge_construction_functions):
                    raise ValueError(
                        f"The edge construction methods '{','.join(self.edge_construction_functions)}' "
                        f"do not require the parameter 'distance_function'."
                    )

            # distance_threshold
            if self.distance_threshold is None:
                if 'distance_based_threshold' in self.edge_construction_functions:
                    raise ValueError(
                        "The parameter distance_threshold is required'."
                    )
            else:
                if not {'distance_based_threshold'}.intersection(self.edge_construction_functions):
                    raise ValueError(
                        f"The edge construction methods '{','.join(self.edge_construction_functions)}' "
                        f"do not require the parameter 'distance_threshold'."
                    )

            # esm2_model_for_contact_map
            if not self.esm2_model_for_contact_map:
                if 'esm2_contact_map' in self.edge_construction_functions:
                    raise ValueError(
                        f"The parameter esm2_model_for_contact_map is required."
                    )
            else:
                if not {'esm2_contact_map'}.intersection(self.edge_construction_functions):
                    raise ValueError(
                        f"The edge construction methods '{','.join(self.edge_construction_functions)}' "
                        f"do not require the parameter 'esm2_model_for_contact_map'."
                    )

            # probability_threshold
            if not self.probability_threshold:
                if 'esm2_contact_map' in self.edge_construction_functions:
                    raise ValueError(
                        f"The parameter 'probability_threshold' is required'."
                    )
            else:
                if not {'esm2_contact_map'}.intersection(self.edge_construction_functions):
                    raise ValueError(
                        f"The edge construction methods '{','.join(self.edge_construction_functions)}' "
                        f"do not require the parameter 'probability_threshold'."
                    )

            # use_edge_attr
            if self.use_edge_attr:
                if not {'distance_based_threshold', 'esm2_contact_map'}.intersection(
                        self.edge_construction_functions) and 'sequence_based' in self.edge_construction_functions and self.distance_function is None:
                    raise ValueError(
                        f"The parameter 'use_edge_attr' is not require. "
                        f"The edge construction methods '{','.join(self.edge_construction_functions)}' "
                        f"do not generate weight matrices."
                    )

            # validation_mode
            if self.validation_mode and not self.randomness_percentage:
                raise ValueError(
                    f"The parameter 'randomness_percentage' is required."
                )

            if not self.validation_mode and self.randomness_percentage:
                raise ValueError(
                    f"The parameter 'randomness_percentage' is not required."
                )

            # split dataset
            if self.split_method and not self.split_training_fraction:
                raise ValueError(
                    f"The parameter 'split_training_fraction' is required."
                )

            if not self.split_method and self.split_training_fraction:
                raise ValueError(
                    f"The parameter 'split_training_fraction' is not required."
                )

            # applicability domain
            features_collection = FeaturesCollectionLoader()
            ad_methods_collection = ADMethodCollectionLoader()

            none_params = [
                param for param in [
                    'methods_for_ad',
                    'feature_file_for_ad'
                ] if getattr(self, param) is None
            ]

            if len(none_params) == 0:
                self.get_ad = True

                valid_methods = ad_methods_collection.get_method_names()
                for method in self.methods_for_ad:
                    if method not in valid_methods:
                        raise ValueError(
                            f"Invalid method: {method}. Allowed methods: {', '.join(valid_methods)}"
                        )

            elif len(none_params) == 2:
                self.get_ad = False
            else:
                raise ValueError(
                    f"The following parameters must be specified: {', '.join(none_params)}"
                )

            if self.mode in 'training':
                self.feature_types_for_ad = features_collection.get_all_features()
            elif self.mode in ('test', 'inference') and self.get_ad:
                self.methods_for_ad, feature_types_for_ad = ad_methods_collection.get_methods_with_features(
                    methods_for_ad=self.methods_for_ad,
                    features_for_ad=features_collection.get_all_features())
                self.feature_types_for_ad = features_collection.get_features_by_name(feature_types_for_ad)

            # parameters that are only used in training mode
            if self.mode in ('test', 'inference'):
                self.number_of_epochs = None
                self.learning_rate = None
                self.pooling_ratio = None
            if self.mode in ('test', 'training'):
                self.inference_batch_size = None

            return self
        except Exception as e:
            logging.getLogger('workflow_logger').critical(e)
            quit()

