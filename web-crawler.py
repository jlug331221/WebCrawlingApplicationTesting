import yaml
import os

config = os.path.join(os.path.dirname(__file__), "web-crawler-config.yaml")

with open(config, "r") as f:
    settings = yaml.load(f)

