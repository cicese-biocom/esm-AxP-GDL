import logging
import logging.config
from pydantic import FilePath
from src_old.utils import json_parser as json_parser
from pathlib import Path


class Logging:
    @staticmethod
    def init(config_file: FilePath, output_dir: FilePath):
        setting_json = Path(config_file).resolve()
        settings = json_parser.load_json(setting_json)
        settings['handlers']['file']['filename'] = output_dir
        logging.config.dictConfig(settings)