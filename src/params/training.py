from pathlib import Path
from typing import Optional, List

from pydantic import PositiveInt, PositiveFloat, confloat
from pydantic.v1 import Field, root_validator

from src.params.common import CommonArguments

from src.config.types import (
    ExecutionMode,
    DistanceFunction,
    EdgeBuildFunction,
    ESM2Representation,
    ESM2ModelForContactMap,
    ValidationMode,
    SplitMethod,
    ModelingTask,
    GDLArchitecture,
)
from src.feature_extraction.collection import FeaturesCollectionLoader
from src.utils.path import check_directory_empty, get_output_path_settings

options_edge_build_functions = ", ".join(f"'{e.value}'" for e in EdgeBuildFunction)


class TrainingArguments(CommonArguments):
    esm2_representation: Optional[ESM2Representation] = Field(
        default=ESM2Representation.ESM2_T33,
        description='ESM-2 representation to be used'
    )

    edge_build_functions: List[EdgeBuildFunction] = Field(
        description=f"Functions to build edges. Options: {options_edge_build_functions}",
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
        description="Probability threshold for constructing a graph based on ESM-2 contact maps"
    )

    number_of_heads: Optional[PositiveInt] = Field(default=8)
    hidden_layer_dimension: Optional[PositiveInt] = Field(default=128)
    add_self_loops: Optional[bool] = Field(default=False)
    use_edge_attr: Optional[bool] = Field(default=False)

    learning_rate: Optional[PositiveFloat] = Field(default=1e-4)
    dropout_rate: Optional[PositiveFloat] = Field(default=0.25)
    pooling_ratio: Optional[PositiveFloat] = Field(default=10)
    number_of_epochs: Optional[PositiveInt] = Field(default=200)

    save_ckpt_per_epoch: Optional[bool] = Field(default=False)

    validation_mode: Optional[ValidationMode] = Field(default=None)
    randomness_percentage: Optional[PositiveFloat] = Field(default=None, gt=0.0, lt=1.0)

    split_method: Optional[SplitMethod] = Field(default=None)
    split_training_fraction: Optional[PositiveFloat] = Field(default=None)

    gdl_architecture: Optional[GDLArchitecture] = Field(default=GDLArchitecture.GATV1)

    modeling_task: ModelingTask
    numbers_of_class: Optional[PositiveInt] = Field(default=None)

    @root_validator(skip_on_failure=True)
    def validate_and_configure(cls, values):
        _configure_execution_mode(values)

        _validate_edge_build_configuration(values)

        _validate_validation_mode_configuration(values)
        _validate_dataset_split_configuration(values)

        _configure_modeling_task(values)
        _configure_optimizer(values)
        _configure_scheduler(values)
        _configure_output_directory(values)

        _load_feature_configuration(values)

        return values


# =========================
# CONFIGURATION
# =========================

def _configure_execution_mode(values):
    values['execution_mode'] = ExecutionMode.TRAIN


def _configure_modeling_task(values):
    task = values.get("modeling_task")
    n = values.get("numbers_of_class")

    if task.name == "MULTICLASS_CLASSIFICATION":
        if n is None:
            raise ValueError("numbers_of_class required")
        values["classes"] = list(range(n))

    elif task.name == "BINARY_CLASSIFICATION":
        if n is not None:
            raise ValueError("numbers_of_class not allowed")
        values["numbers_of_class"] = 2
        values["classes"] = [0, 1]

    elif task.name == "REGRESSION":
        if n is not None:
            raise ValueError("numbers_of_class not allowed")
        values["numbers_of_class"] = 1


def _configure_optimizer(values):
    values['weight_decay'] = 5e-4


def _configure_scheduler(values):
    values['step_size'] = 5
    values['gamma'] = 0.9


def _configure_output_directory(values):
    base = check_directory_empty(values.get('gdl_model_path'))
    base.mkdir(parents=True, exist_ok=True)
    values['output_dir'] = get_output_path_settings(base, ExecutionMode.TRAIN)


def _load_feature_configuration(values):
    loader = FeaturesCollectionLoader()
    values['feature_types_for_ad'] = loader.get_all_features()


def _configure_edge_build_runtime(values, funcs):
    if EdgeBuildFunction.DISTANCE_BASED_THRESHOLD in funcs:
        pdb_path = values.get('pdb_path')
        if pdb_path:
            resolved = Path(pdb_path).resolve()
            resolved.mkdir(parents=True, exist_ok=True)
            values['pdb_path'] = resolved


# =========================
# VALIDATION
# =========================

def _validate_edge_build_configuration(values):
    funcs = values.get('edge_build_functions') or []

    _validate_edge_build_functions_not_empty(funcs)
    _validate_edge_build_functions_compatibility(funcs)

    _validate_edge_build_parameters(funcs, values)
    _validate_edge_attr_usage(funcs, values)

    _configure_edge_build_runtime(values, funcs)


def _validate_edge_build_functions_not_empty(funcs):
    if not funcs:
        raise ValueError("'edge_build_functions' must contain at least one method.")


def _validate_edge_build_functions_compatibility(funcs):
    if EdgeBuildFunction.EMPTY_GRAPH in funcs and len(funcs) > 1:
        raise ValueError("EMPTY_GRAPH cannot be combined with other methods.")


def _validate_edge_build_parameters(funcs, values):
    required_params_by_method = {
        EdgeBuildFunction.DISTANCE_BASED_THRESHOLD: [
            'distance_function',
            'distance_threshold',
            'pdb_path'
        ],
        EdgeBuildFunction.ESM2_CONTACT_MAP: [
            'esm2_model_for_contact_map',
            'probability_threshold'
        ]
    }

    missing_params_by_method = {}
    invalid_params_by_method = {}

    for method, required_params in required_params_by_method.items():
        if method in funcs:
            missing = [
                p for p in required_params
                if values.get(p) is None or values.get(p) == ""
            ]
            if missing:
                missing_params_by_method[method.name] = missing
            else:
                if method == EdgeBuildFunction.DISTANCE_BASED_THRESHOLD:
                    values['amino_acid_representation'] = "CA"

    for method, params in required_params_by_method.items():
        if method not in funcs:
            used = [
                p for p in params
                if values.get(p) is not None
            ]
            if used:
                invalid_params_by_method[method.name] = used

    if missing_params_by_method or invalid_params_by_method:
        error_lines = []

        if missing_params_by_method:
            error_lines.append("Missing required parameters:")
            error_lines.extend(
                f"- {m}: {', '.join(p)}"
                for m, p in missing_params_by_method.items()
            )

        if invalid_params_by_method:
            error_lines.append("Parameters not supported by selected edge_build_functions:")
            error_lines.extend(
                f"- {m}: {', '.join(p)}"
                for m, p in invalid_params_by_method.items()
            )

        raise ValueError("\n".join(error_lines))


def _validate_edge_attr_usage(funcs, values):
    if all(f in {EdgeBuildFunction.SEQUENCE_BASED, EdgeBuildFunction.EMPTY_GRAPH} for f in funcs):
        if values.get('use_edge_attr'):
            raise ValueError("use_edge_attr not allowed.")
        values['use_edge_attr'] = False


def _validate_validation_mode_configuration(values):
    if values.get('validation_mode') and not values.get('randomness_percentage'):
        raise ValueError("randomness_percentage required")
    if not values.get('validation_mode') and values.get('randomness_percentage'):
        raise ValueError("randomness_percentage not required")


def _validate_dataset_split_configuration(values):
    if values.get('split_method') and not values.get('split_training_fraction'):
        raise ValueError("split_training_fraction required")
    if not values.get('split_method') and values.get('split_training_fraction'):
        raise ValueError("split_training_fraction not required")