import yaml
import os

config = os.path.join(os.path.dirname(__file__), "web-crawler-config.yaml")

sites = []
with open(config, "r") as f:
    sites = yaml.load(f)

print("done")