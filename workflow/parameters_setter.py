import logging
from pathlib import Path
from pydantic import BaseModel, FilePath, DirectoryPath, Field, PositiveFloat, PositiveInt, model_validator, \
    field_validator, validator
from typing import Optional, List, Dict
from typing_extensions import Annotated, Literal, Type
import torch
from utils import file_system_handler as file_system_handler


class ParameterSetter(BaseModel):
    mode: Annotated[Optional[str], Field(description='Execution mode')] = None
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
                                                    'esm2_t48']],
                                   Field(description='ESM-2 model to be used')] = None

    tertiary_structure_method: Annotated[Optional[Literal['esmfold']],
                                         Field(description='3D structure prediction method. None indicates to load '
                                                           'existing tertiary structures from PDB files, otherwise, '
                                                           'sequences in input CSV file are predicted using the '
                                                           'specified method')] = None

    pdb_path: Annotated[Optional[DirectoryPath],
                        Field(description='Path where tertiary structures are saved in or loaded from PDB files',
                              exclude=True)] = None

    edge_construction_functions: Annotated[List[Literal['distance_based_threshold', 'esm2_contact_map',
                                                        'peptide_backbone']],
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

    learning_rate: Annotated[Optional[PositiveFloat], Field(description='Learning rate')] = 0.25

    dropout_rate: Annotated[Optional[PositiveFloat], Field(default=0.25, description='Dropout rate')] = 1e-4

    batch_size: Annotated[Optional[PositiveInt], Field(description='Batch size')] = 512

    number_of_epochs: Annotated[Optional[PositiveInt], Field(description='Maximum number of epochs')] = 200

    gdl_model_path: Annotated[Path, Field(description='Path where trained models are saved, or from where a trained '
                                                      'model is loaded for test/inference mode', exclude=True)]

    save_ckpt_per_epoch: Annotated[Optional[bool],
                                   Field(strict=True,
                                         description='True if specified, otherwise, False. '
                                                     'True indicates to save the models per epoch.')] = True

    validation_mode: Annotated[Optional[Literal['coordinates_scrambling', 'embedding_scrambling']],
                               Field(description='Graph construction method for validation of the approach')] = None

    scrambling_percentage: Annotated[Optional[PositiveInt],
                                     Field(description='Percentage of rows to be scrambling')] = None

    output_setting: Annotated[Optional[Dict], Field(description='Output settings', exclude=True)] = None

    seed: Annotated[Optional[PositiveInt],
                    Field(description='Percentage of rows to be scrambling')] = None

    @model_validator(mode='after')
    def check_distance_function(self) -> 'ParameterSetter':
        try:
            self.gdl_model_path = self.gdl_model_path.resolve()
            self.checkpoint = str(self.gdl_model_path.name)

            # database
            self.dataset = file_system_handler.check_file_exists(self.dataset)
            self.dataset_name = self.dataset.name
            self.dataset_extension = file_system_handler.check_file_format(self.dataset)

            # pdb_path
            if self.pdb_path:
                if 'distance_based_threshold' in self.edge_construction_functions \
                        or 'peptide_backbone' in self.edge_construction_functions \
                        and self.distance_function is not None:
                    self.pdb_path = file_system_handler.check_file_exists(self.pdb_path)
                else:
                    self.pdb_path = None
                    self.tertiary_structure_method = None
                    logging.getLogger('workflow_logger').warning(
                        f"Edge construction methods {self.edge_construction_functions} "
                        f"with distance_function={self.distance_function} do not require "
                        f"tertiary structures. Parameters pdb_path and tertiary_structure_method are set to None")

            # distance_function
            if self.distance_function is None:
                if 'distance_based_threshold' in self.edge_construction_functions:
                    raise ValueError('Parameter distance_function is required')
            else:
                if not {'distance_based_threshold', 'peptide_backbone'}.intersection(self.edge_construction_functions):
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

            # use_edge_attr
            if self.use_edge_attr:
                if not {'distance_based_threshold', 'esm2_contact_map'}.intersection(self.edge_construction_functions) \
                        and 'peptide_backbone' in self.edge_construction_functions and self.distance_function is None:
                    self.use_edge_attr = False
                    logging.getLogger('workflow_logger').warning(
                        f"Edge construction methods {self.edge_construction_functions} "
                        f"do not generate weight matrices. Parameter use_edge_attr has been set to False")

            # validation_mode
            if self.validation_mode and not self.scrambling_percentage:
                raise ValueError('Parameter scrambling_percentage is required')

            if not self.validation_mode:
                self.scrambling_percentage = None

            # use_esm2_contact_map
            self.use_esm2_contact_map = True if 'esm2_contact_map' in self.edge_construction_functions else False
            logging.getLogger('workflow_logger').warning(
                f"The parameter use_esm2_contact_map has been set to {self.use_esm2_contact_map}")

            return self
        except Exception as e:
            logging.getLogger('workflow_logger').critical(e)
            quit()
