__all__ = ["config"]


import os
import yaml


with open(
    (
        "/run/secrets/config.yaml"
        if os.getenv("PROD")
        else os.path.join(os.getcwd(), "config.yaml")
    ),
    "r",
) as stream:
    config = yaml.safe_load(stream)
