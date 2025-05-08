from pathlib import Path
from typing import List, Dict

from utils.json_parser import load_json


class ADMethodCollectionLoader:
    def __init__(
            self
    ):
        json_path = Path.cwd().joinpath("settings", "ad_methods_collection.json")
        try:
            self.methods_for_ad = load_json(json_path)['ad_methods_collection']
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found at {json_path}")
        except KeyError:
            raise KeyError("Key 'ad_methods_collection' not found in the JSON file.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while loading methods for applicability domain: {e}")

    # features_for_ad: List[Dict]
    def get_methods_with_features(self, methods_for_ad: List[str], features_for_ad: List[Dict]) -> List[Dict]:
        valid_method_ids = {method['method_id'] for method in self.methods_for_ad}

        provided_methods = set(methods_for_ad)
        invalid_methods = provided_methods - valid_method_ids

        if invalid_methods:
            raise ValueError(
                f"The following methods are not valid: {', '.join(invalid_methods)}. "
                f"Please choose valid methods from the available options: {', '.join(valid_method_ids)}."
            )

        ad_methods = []
        all_features = set()
        for method in self.methods_for_ad:
            if method['method_id'] in methods_for_ad:
                intersected_feature_details = [
                    feature for feature in features_for_ad if feature['feature_name'] in method['features']
                ]

                if len(intersected_feature_details) != len(method['features']):
                    raise ValueError(
                        f"The method {method['method_id']} has erroneous features."
                    )

                all_features.update(method['features'])

                ad_methods.append({
                    "method_name": method["method_name"],
                    "features": intersected_feature_details,
                    "method_id": method["method_id"]
                })

        return ad_methods, all_features

    def get_method_names(self) -> List[str]:
        method_names = {method['method_id'] for method in self.methods_for_ad}
        return method_names
