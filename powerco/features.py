"""Snapshot features with explicit temporal eligibility assumptions."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

CATEGORICAL = ["channel_sales", "origin_up", "has_gas"]
NUMERIC = [
    "cons_12m",
    "cons_gas_12m",
    "cons_last_month",
    "nb_prod_act",
    "num_years_antig",
    "pow_max",
]
RAW_FEATURES = CATEGORICAL + NUMERIC
ENGINEERED = "recent_electricity_share"


class SnapshotFeatures(TransformerMixin, BaseEstimator):
    """Stateless selection and ratio; no date or forecast values are accepted.

    Eligibility is conditional on supplied historical fields being available at
    the unspecified customer snapshot. This is not a verified temporal guarantee.
    """

    def fit(self, X, y=None):
        self._check_columns(X)
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        return self

    def transform(self, X):
        self._check_columns(X)
        out = X[RAW_FEATURES].copy()
        for col in CATEGORICAL:
            out[col] = out[col].fillna("MISSING").astype(str)
        for col in NUMERIC:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        denominator = out["cons_12m"].where(out["cons_12m"] > 0)
        out[ENGINEERED] = (out["cons_last_month"] / denominator).replace([np.inf, -np.inf], np.nan)
        return out

    @staticmethod
    def _check_columns(X):
        missing = set(RAW_FEATURES).difference(X.columns)
        if missing:
            raise ValueError(f"Missing required prediction features: {sorted(missing)}")


def signed_log(values):
    """Reduce scale of valid extreme consumption without deleting customers."""
    return np.sign(values) * np.log1p(np.abs(values))
