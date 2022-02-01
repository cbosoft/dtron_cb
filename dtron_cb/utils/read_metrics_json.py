from typing import List

import json


def read_metrics_json(path: str) -> List[dict]:
    with open(path) as f:
        lines = f.readlines()

    data = [json.loads(line) for line in lines]

    return data
