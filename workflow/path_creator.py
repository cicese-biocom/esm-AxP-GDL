from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict

from injector import inject

from utils import file_system_handler as file_system_handler


class PathCreator:
    @abstractmethod
    def create_path(self, base_path: Path):
        pass


class TrainingModePathCreator(PathCreator):
    def create_path(self, base_path: Path) -> Dict:
        base_path = file_system_handler.check_directory_empty(base_path)
        base_path.mkdir(parents=True, exist_ok=True)         
        return file_system_handler.get_output_path_settings(base_path, 'training')


class TestModePathCreator(PathCreator):
    def create_path(self, base_path: Path) -> Dict:
        base_path = file_system_handler.check_directory_exists(base_path)
        current_time = datetime.now().replace(microsecond=0).isoformat().replace(':', '.')
        new_dir = base_path.joinpath(f"Prediction-{current_time}")
        new_dir.mkdir(parents=True)
        return file_system_handler.get_output_path_settings(new_dir, 'test')


class InferenceModePathCreator(PathCreator):
    def create_path(self, base_path: Path) -> Dict:
        base_path = file_system_handler.check_directory_exists(base_path)
        current_time = datetime.now().replace(microsecond=0).isoformat().replace(':', '.')
        new_dir = base_path.joinpath(f"Inference-{current_time}")
        new_dir.mkdir(parents=True)
        return file_system_handler.get_output_path_settings(new_dir, 'inference')


class PathCreatorContext:
    def __init__(self, path_creator: PathCreator) -> Dict:
        self._path_creator = path_creator

    def create_path(self, base_path: Path) -> Dict:
        return self._path_creator.create_path(base_path)
