import yaml
import os

input_bank = os.path.join(os.path.dirname(__file__), "test-bank.yaml")

with open(input_bank, "r") as f:
    banks = yaml.load(f)

print(banks)