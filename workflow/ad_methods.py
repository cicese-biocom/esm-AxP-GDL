from abc import ABC, abstractmethod
import pandas as pd
from sklearn.ensemble import IsolationForest


# factory method


class ADMethod(ABC):
    @abstractmethod
    def getting_ad(
            self,
            features_to_build_domain: pd.DataFrame,
            features_to_eval: pd.DataFrame
    ):
        pass


class PercentileBasedMethod(ADMethod):
    def __init__(self, features_to_build_domain: pd.DataFrame):
        self.model = {}

        for column in features_to_build_domain.columns:
            q1 = features_to_build_domain[column].quantile(q=0.25)
            q3 = features_to_build_domain[column].quantile(q=0.75)
            ric = q3 - q1
            lower_bound = q1 - 1.5 * ric
            upper_bound = q3 + 1.5 * ric

            self.model[column] = [lower_bound, upper_bound]

    def getting_ad(
            self,
            features_to_eval: pd.DataFrame
    ):
        # build domain
        outliers = pd.DataFrame(index=features_to_eval.index, columns=features_to_eval.columns)

        for column in features_to_eval.columns:
            lower_bound = self.model[column][0]
            upper_bound = self.model[column][1]

            outliers[column] = (features_to_eval[column] < lower_bound) | (features_to_eval[column] > upper_bound)

        # eval
        number_of_outliers = outliers.sum(axis=1)
        outlier_score = [f'{x} / {len(features_to_eval.columns)}' for x in number_of_outliers]
        outliter_by_majority_vote = (number_of_outliers > (0.5 * len(features_to_eval.columns))).apply(
            lambda x: -1 if x else 1)

        return outliter_by_majority_vote, outlier_score


class IsolationForestMethod(ADMethod):
    def __init__(self, features_to_build_domain: pd.DataFrame):
        # build domain
        self.model = IsolationForest(random_state=0, n_jobs=-1).fit(features_to_build_domain)

    def getting_ad(
            self,
            features_to_eval: pd.DataFrame
    ):
        # eval
        outliers = self.model.predict(features_to_eval)
        outlier_scores = self.model.decision_function(features_to_eval)

        return outliers, outlier_scores


def getting_ad(method: str, features_to_build_domain: pd.DataFrame):
    methods = {
        "percentile_based": PercentileBasedMethod,
        "isolation_forest": IsolationForestMethod
    }
    return methods[method](features_to_build_domain)


if __name__ == "__main__":
    features_to_build_domain = pd.DataFrame({
        'sequence': [1, 2, 3, 4, 5],
        '1': [10, 12, 15, 10, 14],
        '2': [20, 21, 25, 19, 22],
        '3': [30, 32, 28, 33, 31],
        '4': [40, 38, 35, 37, 39]
    })

    features_to_eval = pd.DataFrame({
        'sequence': [6, 7, 8, 9],
        '1': [11, 9, 999, 15],
        '2': [23, 18, 999, 21],
        '3': [29, 35, 999, 28],
        '4': [41, 36, 34, 40]
    })

    identifiers = features_to_eval.iloc[:, 0]
    features_to_build_domain = features_to_build_domain.iloc[:, 1:]
    features_to_eval = features_to_eval.iloc[:, 1:]

    # percentile_based
    ad_method = "percentile_based"
    percentile_based = getting_ad(ad_method, features_to_build_domain)
    outlier, outlier_score = percentile_based.getting_ad(features_to_eval)
    result = pd.DataFrame({
        'sequence': identifiers,
        ad_method + '_ad_method': ['out' if x == -1 else 'in' for x in outlier],
        ad_method + '_ad_score': outlier_score
    })
    print(result)

    # percentile_based
    ad_method = "isolation_forest"
    isolation_forest = getting_ad(ad_method, features_to_build_domain)
    outlier, outlier_score = isolation_forest.getting_ad(features_to_eval)
    result = pd.DataFrame({
        'sequence': identifiers,
        ad_method + '_ad_method': ['out' if x == -1 else 'in' for x in outlier],
        ad_method + '_ad_score': outlier_score
    })
    print(result)
