from pathlib import Path
from typing import Optional, List, Dict, Set

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
from src.utils.path import check_directory_empty, get_output_path_settings, check_file_exists

options_edge_build_functions = ", ".join(f"'{e.value}'" for e in EdgeBuildFunction)


class TrainingArguments(CommonArguments):
    esm2_representation: Optional[ESM2Representation] = Field(
        default=ESM2Representation.ESM2_T33,
        description='ESM-2 representation to be used'
    )

    edge_build_functions: Optional[List[EdgeBuildFunction]] = Field(
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

    number_of_heads: Optional[PositiveInt] = Field(default=8, description="Number of heads")
    hidden_layer_dimension: Optional[PositiveInt] = Field(default=128, description="Hidden layer dimension")

    add_self_loops: Optional[bool] = Field(
        default=False,
        description="True if specified, otherwise, False. True indicates to use auto loops in attention layer"
    )

    use_edge_attr: Optional[bool] = Field(
        default=False,
        description="True if specified, otherwise, False. True indicates to use edge attributes in graph learning"
    )

    learning_rate: Optional[PositiveFloat] = Field(default=1e-4, description="Learning rate")
    dropout_rate: Optional[PositiveFloat] = Field(default=0.25, description="Dropout rate")
    pooling_ratio: Optional[PositiveFloat] = Field(default=10, description='Pooling ratio')
    number_of_epochs: Optional[PositiveInt] = Field(default=200, description="Maximum number of epochs")

    save_ckpt_per_epoch: Optional[bool] = Field(
        default=False,
        description="True if specified, otherwise, False. True indicates that the models of every epoch will be saved. "
                    "False indicates that the latest model and the best model regarding the MCC metric will be saved"
    )

    validation_method: Optional[ValidationMode] = Field(
        default=None,
        description=(
            "Validation strategy to assess whether model predictions are not obtained by chance. Options: "
            "'random_coordinates' (randomizes node geometric coordinates to test structural dependence), "
            "'random_embeddings' (shuffles node features to assess feature importance), "
            "'random_graphs' (uses Erdős-Rényi random graphs as a baseline for graph structure relevance). "
            "If not set, no validation is applied."
        )
    )

    randomness_percentage: Optional[PositiveFloat] = Field(
        default=None,
        description=(
            "Percentage of nodes to be perturbed during validation. For 'random_embeddings', it defines "
            "the fraction of node features to shuffle; for 'random_coordinates', the fraction of node "
            "geometric coordinates to randomize."
        ),
        gt=0.0,
        lt=1.0
    )

    probability_for_edge_creation: Optional[PositiveFloat] = Field(
        default=None,
        description=(
            "Probability of edge creation (p) in the Erdős-Rényi model used in 'random_graphs'. "
            "Controls the density of the generated random graph."
        ),
        gt=0.0,
        lt=1.0
    )

    seed_for_edge_creation: Optional[PositiveInt] = Field(
        default=None,
        description=(
            "Random seed used for reproducible generation of edges in the Erdős-Rényi graph for "
            "'random_graphs' validation."
        )
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
        description='GDL architectures to use'
    )

    modeling_task: ModelingTask = Field(
        description="Type of modeling task to execute"
    )

    numbers_of_class: Optional[PositiveInt] = Field(
        default=None,
        description="Number of classes to predict (required if modeling_task is 'multiclass')."
    )

    @root_validator(skip_on_failure=True)
    def validate_and_configure_training_mode(cls, values):
        _configure_execution_mode(values)

        _validate_edge_build_configuration(values)

        _validate_validation_method_configuration(values)
        _validate_dataset_split_configuration(values)

        _validate_dataset_csv(values)

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





# =========================
# VALIDATION
# =========================

def _validate_edge_build_configuration(values):
    funcs = values.get('edge_build_functions') or []
    validation_method = values.get('validation_method')

    validate_edge_build_funcs_compatibility_with_validation_method(funcs, validation_method)

    _validate_edge_build_functions_not_empty(funcs, validation_method)
    _validate_edge_build_functions_compatibility(funcs)

    _validate_edge_build_parameters(funcs, values)
    _validate_edge_attr_usage(funcs, values)


def _validate_edge_build_functions_not_empty(funcs, validation_method):
    if not funcs and not validation_method:
        raise ValueError("'edge_build_functions' must contain at least one method.")


def _validate_edge_build_functions_compatibility(funcs):
    if EdgeBuildFunction.EMPTY_GRAPH in funcs and len(funcs) > 1:
        raise ValueError("EMPTY_GRAPH cannot be combined with other methods.")


def validate_edge_build_funcs_compatibility_with_validation_method(
    funcs: List["EdgeBuildFunction"],
    validation_method: "ValidationMode"
):
    if validation_method:
        compatibility: Dict["ValidationMode", Set["EdgeBuildFunction"]] = {
            ValidationMode.RANDOM_GRAPHS: frozenset(),

            ValidationMode.RANDOM_COORDINATES: {
                EdgeBuildFunction.DISTANCE_BASED_THRESHOLD
            },

            ValidationMode.RANDOM_EMBEDDINGS: {
                EdgeBuildFunction.DISTANCE_BASED_THRESHOLD,
                EdgeBuildFunction.ESM2_CONTACT_MAP,
                EdgeBuildFunction.SEQUENCE_BASED
            }
        }

        incompatible_funcs = [f for f in funcs if f not in compatibility[validation_method]]

        if incompatible_funcs:
            raise ValueError(
                f"The validation method '{validation_method.value}' is not compatible "
                f"with the following edge build functions: {[f.value for f in incompatible_funcs]}"
            )

def _validate_edge_build_parameters(funcs, values):
    required_params_by_method = {
        EdgeBuildFunction.DISTANCE_BASED_THRESHOLD: [
            'distance_function',
            'distance_threshold',
            'pdb_path',
            'load_tertiary_structure'

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


def _validate_validation_method_configuration(values):
    validation_method = values.get('validation_method')

    # Required params ONLY for methods that truly need them
    required_params_by_method = {
        ValidationMode.RANDOM_EMBEDDINGS: [
            'randomness_percentage',
        ],
        ValidationMode.RANDOM_COORDINATES: [
            'randomness_percentage'
        ],
        # RANDOM_GRAPHS -> no required params
    }

    # Optional params allowed per method
    optional_params_by_method = {
        ValidationMode.RANDOM_GRAPHS: [
            'probability_for_edge_creation',
            'seed_for_edge_creation'
        ]
    }

    all_validation_params = {
        'randomness_percentage',
        'probability_for_edge_creation',
        'seed_for_edge_creation'
    }

    provided_params = [
        p for p in all_validation_params
        if values.get(p) is not None
    ]

    # --- Case 1: No validation mode but params were provided ---
    if validation_method is None:
        if provided_params:
            raise ValueError(
                "Validation parameters were provided but 'validation_method' is not specified:\n"
                + ", ".join(provided_params)
            )
        return values

    # --- Case 2: Validate required params ---
    required_params = required_params_by_method.get(validation_method, [])

    missing = [
        p for p in required_params
        if values.get(p) is None
    ]

    # --- Case 3: Validate unsupported params ---
    allowed_params = set(required_params)

    # Add optional params if any
    allowed_params.update(optional_params_by_method.get(validation_method, []))
    allowed_params.add('validation_method')

    invalid = [
        p for p in provided_params
        if p not in allowed_params
    ]

    # --- Consolidated error ---
    if missing or invalid:
        error_lines = []

        if missing:
            error_lines.append(
                f"Missing required parameters for {validation_method.name}: "
                + ", ".join(missing)
            )

        if invalid:
            error_lines.append(
                f"Parameters not supported by {validation_method.name}: "
                + ", ".join(invalid)
            )

        raise ValueError("\n".join(error_lines))

    # --- Defaults for RANDOM_GRAPHS ---
    if validation_method == ValidationMode.RANDOM_GRAPHS:
        if not values.get('probability_for_edge_creation'):
            values['probability_for_edge_creation'] = 0.5

    return values


def _validate_dataset_split_configuration(values):
    if values.get('split_method') and not values.get('split_training_fraction'):
        raise ValueError("split_training_fraction required")
    if not values.get('split_method') and values.get('split_training_fraction'):
        raise ValueError("split_training_fraction not required")


def _validate_dataset_csv(values):
    """
    Validates that dataset exists and is a CSV file.
    """
    dataset = values.get("dataset")
    dataset_path = check_file_exists(dataset)
    if dataset_path.suffix.lower() != ".csv":
        raise ValueError("Dataset must be a CSV file for training")
    values["dataset"] = dataset_path

    values["dataset_file_type"] = "CSV"
