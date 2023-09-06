import os
import yaml

class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance.load_config()
        return cls._instance

    def load_config(self):
        path_base = os.getcwd() + os.sep + "config" + os.sep + 'config.yaml'
        with open(path_base, 'r') as config_file:
            self.config_data = yaml.safe_load(config_file)

    def get(self, key, default=None):
        return self.config_data.get(key, default)