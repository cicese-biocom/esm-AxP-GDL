from pathlib import Path
from typing import List, Dict
from utils.json_parser import load_json


class FeaturesCollectionLoader:
    def __init__(
            self
    ):
        json_path = Path.cwd().joinpath("settings", "features_collection.json")
        try:
            self.features_collection = load_json(json_path)['features_collection']
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found at {json_path}")
        except KeyError:
            raise KeyError("Key 'features_collection' not found in the JSON file.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while loading feature groups: {e}")

    def get_all_features(self) -> List[Dict]:
        return [
            {"feature_name": item["feature_name"],
             "feature_id": item.get("feature_id", ""),
             "type": item.get("type")}
            for item in self.features_collection if item["feature_name"]]

    def get_features_by_name(self, feature_names: List[str]) -> List[Dict]:
        return [
            {"feature_name": item["feature_name"],
             "feature_id": item.get("feature_id", ""),
             "type": item.get("type")}
            for item in self.features_collection if item["feature_name"] in feature_names]
