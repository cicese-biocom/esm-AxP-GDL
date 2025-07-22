from abc import ABC, abstractmethod
import pandas as pd
from sklearn.ensemble import IsolationForest

# factory method


class ApplicabilityDomainMethod(ABC):
    @abstractmethod
    def eval_model(
            self,
            features_to_build_domain: pd.DataFrame,
            features_to_eval: pd.DataFrame
    ):
        pass


class PercentileBasedMethod(ApplicabilityDomainMethod):
    def __init__(self, features_to_build_domain: pd.DataFrame):
        self.model = {}

        for column in features_to_build_domain.columns:
            q1 = features_to_build_domain[column].quantile(q=0.25)
            q3 = features_to_build_domain[column].quantile(q=0.75)
            ric = q3 - q1
            lower_bound = q1 - 1.5 * ric
            upper_bound = q3 + 1.5 * ric

            self.model[column] = [lower_bound, upper_bound]

    def eval_model(
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
        outlier_score = [f'{out} (out of {len(features_to_eval.columns)})' for out in number_of_outliers]
        outliter_by_majority_vote = (number_of_outliers > (0.5 * len(features_to_eval.columns))).apply(
            lambda out: -1 if out else 1)

        return outliter_by_majority_vote, outlier_score


class IsolationForestMethod(ApplicabilityDomainMethod):
    def __init__(self, features_to_build_domain: pd.DataFrame):
        # build domain
        self.model = IsolationForest(random_state=0, n_jobs=-1).fit(features_to_build_domain)

    def eval_model(
            self,
            features_to_eval: pd.DataFrame
    ):
        # eval
        outliers = self.model.predict(features_to_eval)
        outlier_scores = self.model.decision_function(features_to_eval)

        return outliers, outlier_scores


def build_model(method: str, features_to_build_domain: pd.DataFrame):
    methods = {
        "percentile_based": PercentileBasedMethod,
        "isolation_forest": IsolationForestMethod
    }
    return methods[method](features_to_build_domain)

