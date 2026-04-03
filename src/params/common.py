import logging
import os
from pathlib import Path
from typing import Optional
import torch
from pydantic.v1 import BaseModel, Field, root_validator, FilePath, DirectoryPath
from dotenv import load_dotenv

from src.config.types import EdgeBuildFunction
from src.utils.json import load_json
from torch import hub


class CommonArguments(BaseModel):
    dataset: Path = Field(
        description="Path to the input dataset (CSV for training/test, CSV or FASTA for inference)"
    )

    load_tertiary_structure: Optional[bool] = Field(
        default=False,
        description="If True, load tertiary structures; otherwise predict tertiary structures with ESMFold"
    )

    pdb_path: Optional[DirectoryPath] = Field(
        description="Path where tertiary structures are saved or loaded from PDB files"
    )

    batch_size: Optional[int] = Field(
        default=512,
        description="Batch size"
    )

    gdl_model_path: Path = Field(
        description="Path to trained models or for loading a trained model in test/inference mode"
    )

    command_line_build_graphs_parameters: Optional[Path] = Field(
        default=None,
        description="Path to a JSON file with command line parameters"
    )

    @root_validator(skip_on_failure=True)
    def validate_and_configure_common_params(cls, values):
        """
        Executes all steps for common arguments in a clear sequence.
        Each step is a helper that may validate or configure a value.
        """

        # Step 1: Load .env variables
        cls._load_env_variables()

        # Step 2: Parse JSON CLI overrides if provided
        cls._load_command_line_json(values)

        # Step 4: Resolve and validate a PDB path if distance-based graph is used
        cls._resolve_pdb_path_if_needed(values)

        # Step 5: Resolve and validate load_tertiary_structure if distance-based graph is used
        cls._resolve_load_tertiary_structure_if_needed(values)

        # Step 6: Validate environment-dependent files
        cls._validate_env_files()

        # Step 7: Configure device
        cls._configure_computational_device(values)

        return values

    # ----------------------
    # Step helpers
    # ----------------------
    @classmethod
    def _load_env_variables(cls):
        """
        Loads .env variables required for execution.
        """
        load_dotenv(dotenv_path=".env")

    @classmethod
    def _load_command_line_json(cls, values):
        """
        Overrides values with parameters from a JSON file if provided.
        """
        json_path = values.get("command_line_build_graphs_parameters")
        if json_path:
            try:
                json_file_path = Path(json_path).resolve()
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

                import sys
                sys.argv = cls._dict_to_argv(sys.argv[0], {**values, **argv})
            except Exception as e:
                raise ValueError(f"Error loading JSON parameters from '{json_path}': {e}")

    @classmethod
    def _resolve_pdb_path_if_needed(cls, values):
        """
        Resolves a PDB path if distance-based graph method is used.
        Creates the directory if it does not exist.
        """

        funcs = values.get('edge_build_functions') or []

        if EdgeBuildFunction.DISTANCE_BASED_THRESHOLD in funcs:
            pdb_path = values.get('pdb_path')
            if pdb_path:
                resolved = Path(pdb_path).resolve()
                resolved.mkdir(parents=True, exist_ok=True)
                values['pdb_path'] = resolved

    @classmethod
    def _resolve_load_tertiary_structure_if_needed(cls, values):
        """
        Validates and resolves the 'load_tertiary_structure' parameter based on the
        selected edge_build_functions.

        - If DISTANCE_BASED_THRESHOLD is not selected, 'load_tertiary_structure'
          must not be provided and will be set to None.
        """

        edge_methods = values.get('edge_build_functions') or []
        load_tertiary_structure = values.get('load_tertiary_structure')

        uses_distance_based = EdgeBuildFunction.DISTANCE_BASED_THRESHOLD in edge_methods

        if not uses_distance_based:
            if load_tertiary_structure is False:
                values['load_tertiary_structure'] = None

        return values

    @classmethod
    def _validate_env_files(cls):
        """
        Validates that environment-dependent configuration files exist.
        """
        required_envs = [
            "LOG_CONFIG_FILE",
            "ESM_CHECKPOINTS_DIR",
            "ESM2_REPRESENTATION_CONFIG_FILE",
            "FEATURE_COLLECTION_FILE",
            "OUTPUT_SETTINGS",
            "AD_METHODS_COLLECTION_FILE",
            "AMINO_ACID_DESCRIPTORS_FILE"
        ]

        for var in required_envs:
            val = os.getenv(var)
            if val is None:
                raise EnvironmentError(f"Environment variable '{var}' is not set in .env file.")
            path = Path(val).resolve()
            if not path.exists():
                raise FileNotFoundError(f"{var} file/directory not found at: {path}")
            # Special handling for ESM_CHECKPOINTS_DIR to set torch hub directory
            if var == "ESM_CHECKPOINTS_DIR":
                hub.set_dir(d=str(path))

    @classmethod
    def _configure_computational_device(cls, values):
        """
        Sets 'device' key to torch.device object based on CUDA availability.
        """
        values["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.getLogger("workflow_logger").info(f"Device configured: {values['device']}")

    @staticmethod
    def _dict_to_argv(script_path, parameters):
        argv = [script_path]
        for key, value in parameters.items():
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