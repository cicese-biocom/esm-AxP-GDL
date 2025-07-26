import logging
import os
import sys
from datetime import datetime
from pathlib import Path
import random
import numpy as np
import torch
from dotenv import load_dotenv
from pydantic import PositiveInt, PositiveFloat, confloat, create_model
from pydantic.v1 import BaseModel, Field, root_validator, FilePath, DirectoryPath
import pydantic_argparse
from typing import Optional, List, Dict, get_origin, Union, get_args

from pydantic_argparse import ArgumentParser
from torch import hub


from src.applicability_domain.collection import ADMethodCollectionLoader
from src.config.types import (
    ExecutionMode,
    DistanceFunction,
    EdgeConstructionFunction,
    ESM2Representation,
    ESM2ModelForContactMap,
    ValidationMode,
    SplitMethod,
    MethodsForAD,
    ModelingTask,
    GDLArchitecture, TertiaryStructurePredictionMethod, AminoAcidRepresentation
)
from src.feature_extraction.collection import FeaturesCollectionLoader
from src.utils.json import save_json, load_json
from src.utils.path import check_directory_empty, get_output_path_settings, check_file_exists

options_methods_for_ad = ", ".join(f"'{e.value}'" for e in MethodsForAD)
options_edge_construction_functions = ", ".join(f"'{e.value}'" for e in EdgeConstructionFunction)


class CommonArguments(BaseModel):
    dataset: FilePath = Field(
        description="Path to the input dataset in csv format",
    )

    tertiary_structure_method: Optional[TertiaryStructurePredictionMethod] = Field(
        default=None,
        description="3D structure prediction method. None indicates to load existing tertiary "
                    "structures from PDB files , otherwise, sequences in input CSV file are "
                    "predicted using the specified method"
    )

    pdb_path: Optional[DirectoryPath] = Field(
        description="Path where tertiary structures are saved in or loaded from PDB files",
    )

    batch_size:  Optional[PositiveInt] = Field(
        default=512,
        description="Batch size"
    )

    gdl_model_path: Path = Field(
        description="Path where trained models are saved, or from where a trained model is "
                    "loaded for test/inference mode",
    )

    command_line_params: Optional[Path] = Field(
        default=None,
        description="Path to a JSON file with the parameters to be used from the command line. "
                    "Arguments provided directly via the command line take precedence over those "
                    "specified in this file."
    )

    @root_validator(pre=True)
    def check_json_params_arg(cls, values):
        command_line_params = values.get("command_line_params")
        if command_line_params:
            try:
                json_file_path = Path(command_line_params).resolve()
                json_args = load_json(json_file_path)

                argv = {}
                for key, value in json_args.items():
                    if isinstance(value, bool):
                        argv[key] = value
                    elif key not in values or values[key] is None:
                        if isinstance(value, list) and value:
                            argv[key] = ",".join(map(str, value))
                        elif value is not None:
                            argv[key] = value

                sys.argv = _dict_to_argv(sys.argv[0], {**values, **argv})
            except Exception as e:
                raise ValueError(f"Error loading JSON parameters from '{command_line_params}': {e}")

        return values

    @root_validator(skip_on_failure=True)
    def set_device(cls, values):
        values['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.getLogger('workflow_logger').info(f"Device: {values['device']}")
        return values

    @root_validator(skip_on_failure=True)
    def resolve_dataset(cls, values):
        if values.get('dataset'):
            values['dataset'] = check_file_exists(values['dataset'])
        return values

    @root_validator(skip_on_failure=True)
    def resolve_pdb_path(cls, values):
        if values.get('tertiary_structure_method') and values.get('pdb_path'):
            values['pdb_path'] = values['pdb_path'].resolve()
            values['pdb_path'].mkdir(parents=True, exist_ok=True)
        return values

    @root_validator(skip_on_failure=True)
    def validate_log_config_file(cls, values):
        config_file = os.getenv("LOG_CONFIG_FILE")
        if config_file is None:
            raise EnvironmentError("LOG_CONFIG_FILE environment variable is not set in .env file.")

        config_file = Path(config_file).resolve()
        if not config_file.is_file():
            raise FileNotFoundError(f"Logging configuration file not found at: {config_file}")

        return values

    @root_validator(skip_on_failure=True)
    def validate_esm_checkpoints_dir(cls, values):
        esm_checkpoints_env = os.getenv("ESM_CHECKPOINTS_DIR")
        if esm_checkpoints_env is None:
            raise EnvironmentError(
                "The 'ESM_CHECKPOINTS' environment variable is not set in the .env file. It is required for downloading ESM-2 and ESMFold checkpoints.")

        esm_checkpoints_dir = Path(esm_checkpoints_env).resolve()
        if not esm_checkpoints_dir.is_dir():
            raise FileNotFoundError(f"The specified ESM_CHECKPOINTS_DIR directory does not exist: {esm_checkpoints_dir}")

        hub.set_dir(d=str(esm_checkpoints_dir))
        return values

    @root_validator(skip_on_failure=True)
    def validate_esm2_representation_config_file(cls, values):
        config_file = os.getenv("ESM2_REPRESENTATION_CONFIG_FILE")
        if config_file is None:
            raise EnvironmentError("ESM2_REPRESENTATION_CONFIG_FILE environment variable is not set in .env file.")

        config_file = Path(config_file).resolve()
        if not config_file.is_file():
            raise FileNotFoundError(f"ESM2 representations configuration file not found at: {config_file}")
        return values

    @root_validator(skip_on_failure=True)
    def validate_feature_collection_file(cls, values):
        config_file = os.getenv("FEATURE_COLLECTION_FILE")
        if config_file is None:
            raise EnvironmentError("FEATURE_COLLECTION_FILE environment variable is not set in .env file.")

        config_file = Path(config_file).resolve()
        if not config_file.is_file():
            raise FileNotFoundError(f"Feature collection file not found at: {config_file}")
        return values

    @root_validator(skip_on_failure=True)
    def validate_output_setting_file(cls, values):
        config_file = os.getenv("OUTPUT_SETTINGS")
        if config_file is None:
            raise EnvironmentError("OUTPUT_SETTINGS environment variable is not set in .env file.")

        config_file = Path(config_file).resolve()
        if not config_file.is_file():
            raise FileNotFoundError(f"Output setting file not found at: {config_file}")
        return values

    @root_validator(skip_on_failure=True)
    def validate_ad_method_collection_file(cls, values):
        config_file = os.getenv("AD_METHODS_COLLECTION_FILE")
        if config_file is None:
            raise EnvironmentError("AD_METHODS_COLLECTION_FILE environment variable is not set in .env file.")

        config_file = Path(config_file).resolve()
        if not config_file.is_file():
            raise FileNotFoundError(f"Applicability Domain methods collection file not found at: {config_file}")
        return values

    @root_validator(skip_on_failure=True)
    def validate_amino_acid_descriptors_file(cls, values):
        config_file = os.getenv("AMINO_ACID_DESCRIPTORS_FILE")
        if config_file is None:
            raise EnvironmentError("AMINO_ACID_DESCRIPTORS_FILE environment variable is not set in .env file.")

        config_file = Path(config_file).resolve()
        if not config_file.is_file():
            raise FileNotFoundError(f"AMINO_ACID_DESCRIPTORS configuration file not found at: {config_file}")
        return values


class TrainingArguments(CommonArguments):
    esm2_representation: Optional[ESM2Representation] = Field(
        default=ESM2Representation.ESM2_T33,
        description='ESM-2 representation to be used'
    )

    edge_construction_functions: List[EdgeConstructionFunction] = Field(
        description = f"Functions to build edges. Options: {options_edge_construction_functions}",
        unique_items=True
    )

    distance_function: Optional[DistanceFunction] = Field(
        default=None,
        description='Distance function to construct the edges of the distance-based graph'
    )

    distance_threshold: Optional[PositiveFloat] = Field(
        default=None,
        description="Distance threshold to construct the edges of the distance-based graph"
    )

    esm2_model_for_contact_map: Optional[ESM2ModelForContactMap] = Field(
        default=None,
        description='ESM-2 model to be used to obtain ESM-2 contact map'
    )

    probability_threshold: Optional[confloat(gt=0.5, le=1.0)] = Field(
        default=None,
        description="Probability threshold for constructing a graph based on ESM-2 contact maps. "
                    "It takes a value between 0.5 and 1.0."
    )

    amino_acid_representation: Optional[AminoAcidRepresentation] = Field(
        default="CA",
        description='Reference atom into an amino acid to define a relationship (e.g., distance) regarding another amino acid'
    )

    number_of_heads:  Optional[PositiveInt] = Field(
        default=8,
        description="Number of heads"
    )

    hidden_layer_dimension:  Optional[PositiveInt] = Field(
        default=128,
        description="Hidden layer dimension"
    )

    add_self_loops:  Optional[bool] = Field(
        default=False,
        description="True if specified, otherwise, False. True indicates to use auto loops in attention layer"
    )

    use_edge_attr:  Optional[bool] = Field(
        default=False,
        description="True if specified, otherwise, False. True indicates to use edge attributes in graph learning"
    )

    learning_rate: Optional[PositiveFloat] = Field(
        default=1e-4,
        description="Learning rate"
    )

    dropout_rate: Optional[PositiveFloat] = Field(
        default=0.25,
        description="Dropout rate"
    )

    pooling_ratio: Optional[PositiveFloat] = Field(
        default=10,
        description='Pooling ratio'
    )

    number_of_epochs: Optional[PositiveInt] = Field(
        default=200,
        description="Maximum number of epochs"
    )

    save_ckpt_per_epoch: Optional[bool] = Field(
        default=False,
        description="True if specified, otherwise, False. True indicates that the models of every epoch will be saved. "
                    "False indicates that the latest model and the best model regarding the MCC metric will be saved"
    )

    validation_mode: Optional[ValidationMode] = Field(
        default=None,
        description='Criteria to validator that the predictions of the models are not by chance'
    )

    randomness_percentage: Optional[PositiveFloat] = Field(
        default=None,
        description="Percentage of rows to be randomly generated. This parameter and the --validation_mode parameter are used together",
        gt=0.0,
        lt=1.0
    )

    split_method: Optional[SplitMethod] = Field(
        default=None,
        description='Method to split an input dataset in training and validation sets. This parameter is used when an used-defined validation set is not given. To use this parameter, all no-test instances must be marked as training, i.e., value 1 in the input CSV file.'
    )

    split_training_fraction: Optional[PositiveFloat] = Field(
        default=None,
        description="If the --split_method is specified, this parameter represents the percentage of instances to be "
                    "considered as training. The other ones will be allocated in the validation set. It takes a value "
                    "between 0.6 and 0.9."
    )

    gdl_architecture: Optional[GDLArchitecture] = Field(
        default=GDLArchitecture.GATV1,
        description='GDL architecture to use'
    )

    modeling_task: ModelingTask = Field(
        description="Type of modeling task to execute"
    )

    numbers_of_class: Optional[PositiveInt] = Field(
        default=None,
        description="Number of classes to predict (required if modeling_task is 'multiclass')."
    )

    @root_validator(skip_on_failure=True)
    def resolve_gdl_model_path(cls, values):
        values['gdl_model_path'] = values['gdl_model_path'].resolve()
        return values

    @root_validator(skip_on_failure=True)
    def validate_edge_construction_requirements(cls, values):
        funcs = values.get('edge_construction_functions')

        if EdgeConstructionFunction.DISTANCE_BASED_THRESHOLD in funcs or \
                (EdgeConstructionFunction.SEQUENCE_BASED in funcs and values['distance_function']):

            if not values.get('amino_acid_representation'):
                values['amino_acid_representation'] = 'CA'

            if values.get('pdb_path'):
                values['pdb_path'] = check_file_exists(values['pdb_path'])
            else:
                raise ValueError("The parameter 'pdb_path' is required.")
        else:
            if values.get('pdb_path') or values.get('tertiary_structure_method') or values.get(
                    'amino_acid_representation'):
                raise ValueError("The edge construction methods do not require structural parameters.")
            if values.get('validation_mode') == 'random_coordinates':
                raise ValueError("'random_coordinates' validation mode cannot be used without 3D edge construction.")

        return values

    @root_validator(skip_on_failure=True)
    def set_optimizer_params(cls, values):
        values['weight_decay'] = 5e-4
        return (values)

    @root_validator(skip_on_failure=True)
    def set_scheduler_params(cls, values):
        values['step_size'] = 5
        values['gamma'] = 0.9
        return values

    @root_validator(skip_on_failure=True)
    def validate_distance_parameters(cls, values):
        funcs = values['edge_construction_functions']
        if values.get('distance_function') is None:
            if EdgeConstructionFunction.DISTANCE_BASED_THRESHOLD in funcs:
                raise ValueError("The parameter 'distance_function' is required.")
        else:
            if not {EdgeConstructionFunction.DISTANCE_BASED_THRESHOLD,
                    EdgeConstructionFunction.SEQUENCE_BASED}.intersection(funcs):
                raise ValueError("'distance_function' is not required for the selected edge construction methods.")

        if values.get('distance_threshold') is None:
            if EdgeConstructionFunction.DISTANCE_BASED_THRESHOLD in funcs:
                raise ValueError("The parameter 'distance_threshold' is required.")
        else:
            if EdgeConstructionFunction.DISTANCE_BASED_THRESHOLD not in funcs:
                raise ValueError("'distance_threshold' is not required for the selected edge construction methods.")

        return values

    @root_validator(skip_on_failure=True)
    def validate_esm2_contact_map_parameters(cls, values):
        funcs = values['edge_construction_functions']

        if not values.get('esm2_model_for_contact_map') and EdgeConstructionFunction.ESM2_CONTACT_MAP in funcs:
            raise ValueError("The parameter 'esm2_model_for_contact_map' is required.")
        elif values.get('esm2_model_for_contact_map') and EdgeConstructionFunction.ESM2_CONTACT_MAP not in funcs:
            raise ValueError("'esm2_model_for_contact_map' is not required for the selected edge construction methods.")

        if not values.get('probability_threshold') and EdgeConstructionFunction.ESM2_CONTACT_MAP in funcs:
            raise ValueError("The parameter 'probability_threshold' is required.")
        elif values.get('probability_threshold') and EdgeConstructionFunction.ESM2_CONTACT_MAP not in funcs:
            raise ValueError("'probability_threshold' is not required for the selected edge construction methods.")

        return values

    @root_validator(skip_on_failure=True)
    def validate_use_edge_attr(cls, values):
        funcs = values['edge_construction_functions']
        if values.get('use_edge_attr'):
            if not {EdgeConstructionFunction.DISTANCE_BASED_THRESHOLD,
                    EdgeConstructionFunction.ESM2_CONTACT_MAP}.intersection(funcs) \
                    and EdgeConstructionFunction.SEQUENCE_BASED in funcs and values.get('distance_function') is None:
                raise ValueError(
                    "The parameter 'use_edge_attr' is not required for the selected edge construction methods.")
        return values

    @root_validator(skip_on_failure=True)
    def validate_validation_mode(cls, values):
        if values.get('validation_mode') and not values.get('randomness_percentage'):
            raise ValueError("The parameter 'randomness_percentage' is required.")
        if not values.get('validation_mode') and values.get('randomness_percentage'):
            raise ValueError("The parameter 'randomness_percentage' is not required.")
        return values

    @root_validator(skip_on_failure=True)
    def validate_dataset_split(cls, values):
        if values.get('split_method') and not values.get('split_training_fraction'):
            raise ValueError("The parameter 'split_training_fraction' is required.")
        if not values.get('split_method') and values.get('split_training_fraction'):
            raise ValueError("The parameter 'split_training_fraction' is not required.")
        return values

    @root_validator(skip_on_failure=True)
    def load_features_for_ad(cls, values):
        features_collection = FeaturesCollectionLoader()
        values['feature_types_for_ad'] = features_collection.get_all_features()
        return values

    @root_validator(skip_on_failure=True)
    def check_numbers_of_class_required(cls, values):
        modeling_task = values.get("modeling_task")
        n_classes = values.get("numbers_of_class")

        if modeling_task == ModelingTask.MULTICLASS_CLASSIFICATION:
            if n_classes is None:
                raise ValueError("'numbers_of_class' is required when modeling_task is 'multiclass_classification'")
            values["classes"] = list(range(n_classes))

        elif modeling_task == ModelingTask.BINARY_CLASSIFICATION:
            if n_classes is not None:
                raise ValueError("'numbers_of_class' should not be set when modeling_task is 'binary_classification'")
            values["numbers_of_class"] = 2
            values["classes"] = [0, 1]

        elif modeling_task == ModelingTask.REGRESSION:
            if n_classes is not None:
                raise ValueError("'numbers_of_class' should not be set when modeling_task is 'regression'")
            values["numbers_of_class"] = 1

        return values

    @root_validator(skip_on_failure=True)
    def set_mode(cls, values):
        values['execution_mode'] = ExecutionMode.TRAIN
        return values

    @root_validator(skip_on_failure=True)
    def set_output_dir(cls, values):
        base_path = check_directory_empty(values.get('gdl_model_path'))
        base_path.mkdir(parents=True, exist_ok=True)
        values['output_dir'] = get_output_path_settings(base_path, 'training')
        return values


class PredictionArguments(CommonArguments):
    output_path: Path = Field(
        description="The path where the output data will be saved",
    )

    seed: Optional[PositiveInt] = Field(
        default=None,
        description="Seed used during test/inference mode to enable deterministic behavior when possible. "
                    "Operations without deterministic implementations (e.g., scatter_add_cuda_kernel) will "
                    "fall back to non-deterministic versions, issuing a warning (warn_only=True)."
    )

    methods_for_ad: Optional[List[MethodsForAD]] = Field(
        default=None,
        description=f"Methods to build applicability domain model. Options: {options_methods_for_ad}",
        unique_items=True
    )

    feature_file_for_ad: Optional[Path] = Field(
        default=None,
        description="Path of the CSV file of features to build the applicability domain"
    )
    
    prediction_batch_size: Optional[PositiveInt] = Field(
        default=20000,
        description="As the test/inference data are unlimited, this parameters contains the number of instances to "
                    "be processed in a specific chunk (batch) at a time."
    )

    @root_validator(skip_on_failure=True)
    def set_mode(cls, values):
        values['execution_mode'] = ExecutionMode.TEST
        values['mode_path_name'] = "Prediction"
        return values

    @root_validator(skip_on_failure=True)
    def set_seed(cls, values):
        if values.get('seed') is not None:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

            torch.manual_seed(values['seed'])
            random.seed(values['seed'])
            np.random.seed(values['seed'])
            torch.cuda.manual_seed(values['seed'])
            torch.cuda.manual_seed_all(values['seed'])
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            torch.use_deterministic_algorithms(True, warn_only=True)

        return values

    @root_validator(skip_on_failure=True)
    def set_applicability_domain(cls, values):
        values['get_ad'] = False
        features_collection = FeaturesCollectionLoader()
        ad_methods_collection = ADMethodCollectionLoader()

        none_params = [
            param for param in [
                'methods_for_ad',
                'feature_file_for_ad'
            ] if values.get(param) is None
        ]

        if len(none_params) == 0:
            values['get_ad'] = True

            valid_methods = ad_methods_collection.get_method_names()
            for method in values['methods_for_ad']:
                if method not in valid_methods:
                    raise ValueError(
                        f"Invalid method: {method}. Allowed methods: {', '.join(valid_methods)}"
                    )

        elif len(none_params) == 2:
            values['get_ad'] = False
        else:
            raise ValueError(
                f"The following parameters must be specified: {', '.join(none_params)}"
            )

        if values.get('get_ad'):
            values['methods_for_ad'], feature_types_for_ad = ad_methods_collection.get_methods_with_features(
                methods_for_ad=values['methods_for_ad'],
                features_for_ad=features_collection.get_all_features()
            )
            values['feature_types_for_ad'] = features_collection.get_features_by_name(feature_types_for_ad)

        return values

    @root_validator(skip_on_failure=True)
    def set_output_dir(cls, values):
        values.get('output_path').mkdir(parents=True, exist_ok=True)
        current_time = datetime.now().replace(microsecond=0).isoformat().replace(':', '.')
        new_dir = values.get('output_path').joinpath(f"{current_time}-{values.get('mode_path_name')}")
        new_dir.mkdir(parents=True)
        values['output_dir'] = get_output_path_settings(new_dir, values.get("execution_mode"))
        return values

    @root_validator(skip_on_failure=True)
    def add_model_parameters(cls, values):
        checkpoint = torch.load(values['gdl_model_path'])
        values.update(checkpoint.get('parameters', {}))
        return values


class InferenceArguments(PredictionArguments):
    @root_validator(skip_on_failure=True)
    def set_mode(cls, values):
        values['execution_mode'] = ExecutionMode.INFERENCE
        values['mode_path_name'] = "Inference"
        return values


class ExecutionParameters:
    def __init__(self, execution_mode: ExecutionMode):
        self._model = {
            ExecutionMode.TRAIN: TrainingArguments,
            ExecutionMode.TEST: PredictionArguments,
            ExecutionMode.INFERENCE: InferenceArguments
        }[execution_mode]

    def get_parameters(self) -> Union[TrainingArguments, PredictionArguments, InferenceArguments]:
        load_dotenv(dotenv_path='.env')

        parser = pydantic_argparse.ArgumentParser(model=self._model, exit_on_error=True)
        args = parser.parse_typed_args()

        _save_all_parameters(args)
        _save_command_line_parameters(self._model, args)

        return args

        
def _save_all_parameters(args):
    save_json(
        json_file=args.output_dir['workflow_execution_args_file'],
        json_data=args.dict()
    )


def _save_command_line_parameters(model, args):
    user_fields = model.__fields__.keys()
    user_provided = {
        k: v for k, v in args.dict().items()
        if v is not None and k in user_fields
    }

    save_json(
        json_file=args.output_dir['command_line_params'],
        json_data=user_provided
    )


def _dict_to_argv(script_path, params):
    argv = [script_path]
    for key, value in params.items():
        if isinstance(value, bool):
            if value:
                argv.append(f"--{key}")
        elif isinstance(value, (list, set)):
            if value:
                comma_separated = ",".join(map(str, value))
                argv.extend([f"--{key}", comma_separated])
        elif value:
            argv.extend([f"--{key}", str(value)])
    return argv



