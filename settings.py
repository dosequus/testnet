import yaml
from collections import namedtuple

class Configuration:
    _instance = None

    def __new__(cls, config_file='config.yml'):
        if cls._instance is None:
            cls._instance = super(Configuration, cls).__new__(cls)
            cls._instance._load_config(config_file)
        return cls._instance

    def _load_config(self, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        # Convert nested dictionaries into named tuples for easy access
        def dict_to_namedtuple(dictionary):
            return namedtuple('Config', dictionary.keys())(*[dict_to_namedtuple(v) if isinstance(v, dict) else v for v in dictionary.values()])

        self.config = dict_to_namedtuple(config)

    def get_config(self):
        return self.config
