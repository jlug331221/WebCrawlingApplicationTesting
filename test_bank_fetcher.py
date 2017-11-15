import yaml
import os
import requests


def fetch_bank_for_topic(topic):
    input_bank = os.path.join(os.path.dirname(__file__), "test-bank.yaml")

    banks = []
    with open(input_bank, "r") as f:
        banks = yaml.load(f)

    return banks[topic]