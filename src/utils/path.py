import os
from pathlib import Path
from typing import Dict

from src.config.types import ExecutionMode
from src.utils.json import load_json


def check_directory_exists(base_path: Path) -> Path:
    if not base_path.exists():
        raise NotADirectoryError(f"The directory '{base_path}' does not exist")
    return base_path.resolve()


def check_file_exists(file_path: Path) -> Path:
    if not file_path.exists():
        raise FileNotFoundError(f"The file '{file_path}' does not exist")
    return file_path.resolve()


def check_file_format(file_path: Path) -> Path:
    file_extension = file_path.suffix
    if not file_extension.endswith(('.csv', '.fasta')):
        raise ValueError(f"Unsupported file format. Only csv and fasta files are supported.")
    return file_extension


def check_directory_empty(base_path: Path) -> Path:
    if base_path.exists() and any(base_path.iterdir()):
        raise NotADirectoryError(
            f"The directory '{base_path}' already exists and is not empty. "
            f"Please empty it and try again.")
    return base_path.resolve()


def get_output_path_settings(base_path: Path, execution_mode: ExecutionMode) -> Dict:
    data = load_json(Path(os.getenv("OUTPUT_SETTINGS")).resolve())

    output_path_settings = {}
    for setting in data["output_settings"]:
        if execution_mode.value in setting["modes"]:
            output_path_settings[setting["key"]] = base_path.joinpath(setting["name"])
            output_path_settings[setting["key"]].mkdir(parents=True, exist_ok=True)

            for file in setting.get("files", []):
                if execution_mode.value in file["modes"]:
                    output_path_settings[file["key"]] = output_path_settings[setting["key"]].joinpath(file["name"])
    return output_path_settings
