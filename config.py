import yaml

def get_config(path=None):
    """Return config parameters as dictionary.
       
       Args:
         path: path of config yaml file.
           if path is None or '', default config will be returned.

    """
    path = 'default_config.yaml' if not path else path
    with open(path, 'r') as f:
        config = yaml.load(f)
    return config
