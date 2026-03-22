from pathlib import Path
from typing import Optional, List
from datetime import datetime
import os
import random
import torch
import numpy as np

from pydantic import PositiveInt
from pydantic.v1 import Field, root_validator

from src.applicability_domain.collection import ADMethodCollectionLoader
from src.params.common import CommonArguments
from src.config.types import ExecutionMode, MethodsForAD
from src.feature_extraction.collection import FeaturesCollectionLoader
from src.utils.path import get_output_path_settings


options_methods_for_ad = ", ".join(f"'{e.value}'" for e in MethodsForAD)


class PredictionArguments(CommonArguments):
    output_path: Path = Field(
        description="The path where the output data will be saved",
    )

    seed: Optional[PositiveInt] = Field(
        default=None,
        description="Seed used during test/inference mode to enable deterministic behavior."
    )

    methods_for_ad: Optional[List[str]] = Field(
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
        description="Number of instances to calculate in a single batch during prediction."
    )

    @root_validator(skip_on_failure=True)
    def validate_and_configure(cls, values):
        _configure_execution_mode(values)

        _validate_applicability_domain_configuration(values)

        _configure_seed(values)
        _configure_applicability_domain(values)
        _configure_output_directory(values)
        _load_model_parameters(values)

        return values


# =========================
# CONFIGURATION
# =========================

def _configure_execution_mode(values):
    values['execution_mode'] = ExecutionMode.TEST
    values['mode_path_name'] = "Prediction"


def _configure_seed(values):
    seed = values.get('seed')

    if seed is not None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        torch.use_deterministic_algorithms(True, warn_only=True)


def _configure_applicability_domain(values):
    values['get_ad'] = False

    methods = values.get('methods_for_ad')
    feature_file = values.get('feature_file_for_ad')

    if methods and feature_file:
        values['get_ad'] = True

        features_collection = FeaturesCollectionLoader()
        ad_methods_collection = ADMethodCollectionLoader()

        values['methods_for_ad'], feature_types_for_ad = ad_methods_collection.get_methods_with_features(
            methods_for_ad=methods,
            features_for_ad=features_collection.get_all_features()
        )

        values['feature_types_for_ad'] = features_collection.get_features_by_name(feature_types_for_ad)


def _configure_output_directory(values):
    output_path: Path = values.get('output_path')
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().replace(microsecond=0).isoformat().replace(":", ".")
    run_dir = output_path / f"{timestamp}-{values.get('mode_path_name')}"
    run_dir.mkdir(parents=True)

    values['output_dir'] = get_output_path_settings(run_dir, values.get("execution_mode"))


def _load_model_parameters(values):
    checkpoint_path: Path = values.get('gdl_model_path')

    if checkpoint_path and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path)
        values.update(checkpoint.get('parameters', {}))


# =========================
# VALIDATION
# =========================

def _validate_applicability_domain_configuration(values):
    methods = values.get('methods_for_ad')
    feature_file = values.get('feature_file_for_ad')

    missing_params = []
    invalid_methods = []

    # --- Validate required combination ---
    if methods is None and feature_file is None:
        return

    if methods is None:
        missing_params.append('methods_for_ad')

    if feature_file is None:
        missing_params.append('feature_file_for_ad')

    # --- Validate method names ---
    if methods:
        valid_methods = ADMethodCollectionLoader().get_method_names()

        invalid_methods = [
            m for m in methods
            if m not in valid_methods
        ]

    # --- Raise consolidated error ---
    if missing_params or invalid_methods:
        error_lines = []

        if missing_params:
            error_lines.append("Missing required parameters for applicability domain:")
            error_lines.append(", ".join(missing_params))

        if invalid_methods:
            error_lines.append("Invalid applicability domain methods:")
            error_lines.append(", ".join(invalid_methods))

        raise ValueError("\n".join(error_lines))