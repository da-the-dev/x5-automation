__all__ = ["config"]


import os
import yaml


with open(os.path.join(__path__), "config.yaml") as stream:
    config = yaml.safe_load(stream)
