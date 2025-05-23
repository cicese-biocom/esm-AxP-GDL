import json
from pathlib import Path
from typing import Dict


def load_json(json_file: Path) -> Dict:
    with open(json_file, 'r') as file:
        return json.load(file)


def save_json(json_file: Path, json_data: Dict) -> None:
    def path_converter(obj):
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, set):
            return list(obj)
        raise TypeError(f"Type {type(obj)} not serializable")

    with open(json_file, 'w') as file:
        json.dump(json_data, file, indent=4, default=path_converter)
