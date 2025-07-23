import logging
import logging.config
from pydantic import FilePath
from src.utils.json import load_json
from pathlib import Path


class Logging:
    @staticmethod
    def init(config_file: FilePath, output_dir: FilePath):
        setting_json = Path(config_file).resolve()
        settings = load_json(setting_json)
        settings['handlers']['file']['filename'] = output_dir
        logging.config.dictConfig(settings)