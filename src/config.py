__all__ = ["config"]


import os
import yaml


with open(os.path.join(os.getcwd(), "config.yaml"), "r") as stream:
    config = yaml.safe_load(stream)
