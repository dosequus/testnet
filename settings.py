import yaml
from collections import namedtuple

# Load the configuration from the YAML file
def load_config(config_file='config.yml'):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # Convert nested dictionaries into named tuples for easy access
    def dict_to_namedtuple(dictionary):
        return namedtuple('Config', dictionary.keys())(*[dict_to_namedtuple(v) if isinstance(v, dict) else v for v in dictionary.values()])

    return dict_to_namedtuple(config)
