import copy
import yaml

__all__ = ['get_config', 'Config']

def get_config(path=None):
    """Return config parameters as dictionary.
       
       Args:
         path: path of config yaml file.
           if path is None or '', default config will be returned.

    """
    path = 'default_config.yaml' if not path else path
    config = Config.from_yaml(path)
    return config


class Config(object):
    """ A config utility class."""

    def __init__(self, config_dict=None):
        """ Instantiate Config object from dictionary. """
        for k, v in config_dict.items():
            setattr(self, k, v)

    def __repr__(self):
        return yaml.dump(self.__dict__)

    def to_dict(self):
        return copy.deepcopy(self.__dict__)

    @property
    def keys(self):
        return list(self.__dict__.keys())
    
    @staticmethod
    def from_yaml(path):
        with open(path, 'r') as f:
            config_dict = yaml.load(f)
        return Config(config_dict)
