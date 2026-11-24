"""Fold-local preprocessing and estimator composition."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from powerco.features import CATEGORICAL, ENGINEERED, NUMERIC, SnapshotFeatures, signed_log


def pipeline(model, linear=False):
    numeric = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if linear:
        numeric += [
            ("log", FunctionTransformer(signed_log, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    pre = ColumnTransformer(
        [
            ("num", Pipeline(numeric), NUMERIC + [ENGINEERED]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )
    return Pipeline([("features", SnapshotFeatures()), ("preprocess", pre), ("model", model)])
