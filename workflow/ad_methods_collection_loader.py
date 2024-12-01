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

    def get_methods_with_features(self, method_for_ad: List[str], features_for_ad: List[Dict]) -> List[Dict]:
        result = []
        unique_column_names = set()
        for method in self.methods_for_ad:
            if method["method_id"] in method_for_ad:
                features_names_for_ad = {feature['feature_name'] for feature in features_for_ad}
                intersected_features = features_names_for_ad.intersection(set(method["features"]))

                intersected_feature_details = [
                    feature for feature in features_for_ad if feature['feature_name'] in intersected_features
                ]

                if not intersected_features:
                    # if there is no intersection in features
                    raise ValueError(
                        f"No matching feature groups found for method '{method['method_name']}' "
                        f"with the specified feature group names: {features_for_ad}"
                    )

                if method.get("apply_per_feature") == "True":
                    # cross join: method x features
                    for feature in intersected_feature_details:
                        column_name = f'{method["method_name"]}_ad({feature["feature_id"]})'
                        if column_name not in unique_column_names:
                            unique_column_names.add(column_name)
                            result.append({
                                "method_name": method["method_name"],
                                "features": [feature],
                                "column_name": column_name
                            })
                else:
                    # add the method with all filtered features

                    if len(intersected_feature_details) == 1:
                        feature = intersected_feature_details[0]
                        if feature['type'] == "feature":
                            raise ValueError(
                                f"The method '{method['method_name']}' cannot be applied with only the "
                                f"'{feature['feature_name']}' feature.")

                    features = '_'.join(feature["feature_id"] for feature in intersected_feature_details)
                    column_name = f'{method["method_name"]}_ad({features})'

                    if column_name not in unique_column_names:
                        unique_column_names.add(column_name)
                        result.append({
                            "method_name": method["method_name"],
                            "features": intersected_feature_details,
                            "column_name": column_name
                        })

        return result


if __name__ == "__main__":
    features_for_ad = [
        {'feature_name': 'graph_centralities', 'feature_id': 'gc_', 'type': 'group'},
        {'feature_name': 'perplexity', 'feature_id': '', 'type': 'feature'},
        {'feature_name': 'amino_acid_descriptors', 'feature_id': 'aad_', 'type': 'group'}
    ]

    method_for_ad = ['percentile_based', 'isolation_forest']

    loader = ADMethodCollectionLoader()

    try:
        result = loader.get_methods_with_features(method_for_ad, features_for_ad)
        print(result)
    except ValueError as e:
        print(e)