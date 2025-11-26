import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# factory method


class ApplicabilityDomainMethod(ABC):
    @abstractmethod
    def eval_model(
            self,
            features_to_eval: pd.DataFrame
    ):
        pass


class PercentileBasedMethod(ApplicabilityDomainMethod):
    def __init__(self, features_to_build_domain: pd.DataFrame):
        self.model = {}

        for column in features_to_build_domain.columns:
            col_data = features_to_build_domain[column].dropna()

            if col_data.empty:
                logging.getLogger('workflow_logger').warning(
                    f"Feature '{column}' was excluded from applicability domain construction because all its values are NaN."
                )
                continue

            q1 = col_data.quantile(q=0.25)
            q3 = col_data.quantile(q=0.75)
            ric = q3 - q1
            lower_bound = q1 - 1.5 * ric
            upper_bound = q3 + 1.5 * ric

            self.model[column] = [lower_bound, upper_bound]

        # If no valid features were found, set the model to None and log a warning
        if not self.model:
            self.model = None
            logging.getLogger('workflow_logger').warning(
                "PercentileBasedMethod could not be created because all input features were empty or contained only NaNs."
            )

    # Evaluation rules:
    # - NaN values are not considered outliers.
    # - NaN values are logged and ignored during domain evaluation.
    # - Majority voting is computed only over valid (non-NaN) features.
    # - The outlier score reports the number of outlier features out of the valid features used.
    def eval_model(
            self,
            features_to_eval: pd.DataFrame
    ):

        logger = logging.getLogger('workflow_logger')

        # If the model was not built, return None and log a warning
        if self.model is None:
            logger.warning(
                "Evaluation skipped in PercentileBasedMethod because the model was not properly initialized."
            )
            return None, None

        # DataFrame to store outlier flags
        outliers = pd.DataFrame(index=features_to_eval.index)

        for column, (lower_bound, upper_bound) in self.model.items():
            col_values = features_to_eval[column]

            # Detect NaN values for this feature
            nan_mask = col_values.isna()
            if nan_mask.any():
                ignored_instances = nan_mask[nan_mask].index.tolist()
                logger.info(
                    f"Feature '{column}' contains NaN values for instances {ignored_instances}. "
                    f"These values will be ignored when evaluating the domain."
                )

            # Outlier = only values outside bounds (NaN is not outlier)
            is_outlier = (col_values < lower_bound) | (col_values > upper_bound)

            # Assign, NaN appears as False (not an outlier)
            outliers[column] = is_outlier

        # Count the number of outlier features per instance
        number_of_outliers = outliers.sum(axis=1)

        # Count usable (non-NaN) features for each instance
        valid_features = (~features_to_eval.isna()).sum(axis=1)

        # Majority rule based only on evaluated (non-NaN) features
        outlier_by_majority_vote = (
                number_of_outliers > (0.5 * valid_features)
        ).apply(lambda out: -1 if out else 1)

        # Human-readable scores
        outlier_score = [
            f"{out} (out of {valid})"
            for out, valid in zip(number_of_outliers, valid_features)
        ]

        return outlier_by_majority_vote, outlier_score


class IsolationForestMethod(ApplicabilityDomainMethod):
    def __init__(self, features_to_build_domain: pd.DataFrame):
        # Remove columns with only NaN values
        clean_data = features_to_build_domain.dropna(axis=1, how='all')

        if clean_data.shape[1] == 0:
            logging.warning(
                "IsolationForest model could not be created: all columns were NaN."
            )
            self.model = None
            self.columns = []
            return

        # Drop rows with any NaN values (IsolationForest cannot handle NaN)
        if not clean_data.empty:
            clean_data = clean_data.dropna(axis=0, how='any')

        if clean_data.empty:
            # Log warning and disable the model
            logging.getLogger('workflow_logger').warning(
                'IsolationForest model could not be created: insufficient clean data after removing NaNs.')
            self.model = None
            self.columns = []
            return

        self.columns = clean_data.columns.tolist()

        # build domain
        try:
            self.model = IsolationForest(random_state=0, n_jobs=-1).fit(clean_data)
        except Exception as e:
            logging.exception("Failed to train IsolationForest model due to: %s", e)
            self.model = None

    def eval_model(
            self,
            features_to_eval: pd.DataFrame
    ):
        if self.model is None:
            return None, None

        # Keep only columns used during training
        eval_data = features_to_eval[self.columns]

        # Identify rows with any NaN values in the required features
        nan_mask = eval_data.isna().any(axis=1)

        # Prepare DataFrame without NaNs for prediction
        clean_eval_data = eval_data[~nan_mask]

        preds = pd.Series(index=features_to_eval.index, dtype=int)
        preds[nan_mask] = -1  # Mark rows with NaNs as out-of-domain

        outlier_scores = pd.Series(index=features_to_eval.index, dtype=float)
        outlier_scores[nan_mask] = np.nan  # Assign NaN score to instances with NaNs

        if not clean_eval_data.empty:
            preds[~nan_mask] = self.model.predict(clean_eval_data)
            outlier_scores[~nan_mask] = self.model.decision_function(clean_eval_data)

        return preds, outlier_scores


def build_model(method: str, features_to_build_domain: pd.DataFrame):
    methods = {
        "percentile_based": PercentileBasedMethod,
        "isolation_forest": IsolationForestMethod
    }
    return methods[method](features_to_build_domain)


