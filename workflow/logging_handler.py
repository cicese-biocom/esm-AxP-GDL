import json
import logging
import logging.config
import datetime
import os
from pydantic import FilePath,DirectoryPath
from utils import json_parser as json_parser
from pathlib import Path


class LoggingHandler:
    @staticmethod
    def initialize_logger(logger_settings_path: FilePath, log_output_path: FilePath):
        setting_json = Path(logger_settings_path).resolve()
        settings = json_parser.load_json(setting_json)
        settings['handlers']['file']['filename'] = log_output_path
        logging.config.dictConfig(settings)


if __name__ == '__main__':
    path = Path('Logs').resolve()
    path.mkdir(parents=True, exist_ok=True)
    workflow_log = path.joinpath('workflow.log')
    LoggingHandler.initialize_logger(logger_settings_path='../settings/logger_setting.json',
                                     log_output_path=workflow_log)
    workflow_logger = logging.getLogger('workflow_logger')

    workflow_logger.debug('This is a debug message')
    workflow_logger.info('And this is an additional information message')
    workflow_logger.warning('Beware! It\'s a warning message!')
    workflow_logger.error('We\'ve a serious problem here')
    workflow_logger.critical('<Sigh>, someone tried dividing by zero... again.')
