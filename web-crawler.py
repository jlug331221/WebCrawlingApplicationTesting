import yaml
import os
import requests


config = os.path.join(os.path.dirname(__file__), "web-crawler-config.yaml")

sites = []
with open(config, "r") as f:
    sites = yaml.load(f)

for site in sites:
    for test in site["tests"]:
        s = requests.Session()
        url = test["url"]
        method = test["method"]

        print("*** Test Set:", test["name"], "***")
        for case in test["cases"]:
            print("Test case:", case["name"])

            params = case["params"]
            expectedOutput = case["output"]

            if method.lower() == "post":
                r = s.post(url, data=params)
                if r.status_code != expectedOutput["status_code"]:
                    print("TEST CASE FAILED")
                else:
                    print("TEST CASE PASSED")

