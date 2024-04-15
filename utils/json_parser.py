import json
from pathlib import Path
from typing import Dict


def load_json(json_file: Path) -> Dict:
    with open(json_file, 'r') as file:
        return json.load(file)


def save_json(json_file: Path, json_data: Dict) -> None:
    with open(json_file, 'w') as file:
        json.dump(json_data, file, indent=4)
