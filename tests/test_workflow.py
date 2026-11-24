"""Focused leakage/feature contract checks; no test-set model optimisation."""

import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from powerco.config import DATA_DIR
from powerco.features import ENGINEERED, RAW_FEATURES, SnapshotFeatures
from powerco.preprocessing import pipeline


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_csv(DATA_DIR / "client_data.csv").iloc[:300].copy()

    def test_excluded_columns_cannot_affect_features(self):
        a = SnapshotFeatures().fit_transform(self.raw)
        changed = self.raw.copy()
        for col in changed.columns.difference(RAW_FEATURES):
            changed[col] = "UNAVAILABLE FUTURE VALUE"
        pd.testing.assert_frame_equal(a, SnapshotFeatures().fit_transform(changed))

    def test_undefined_ratio_becomes_missing_not_infinity(self):
        data = self.raw.iloc[:2].copy()
        data["cons_12m"] = [0, 100]
        data["cons_last_month"] = [100, 10]
        result = SnapshotFeatures().fit_transform(data)
        self.assertTrue(pd.isna(result[ENGINEERED].iloc[0]))
        self.assertAlmostEqual(result[ENGINEERED].iloc[1], 0.1)

    def test_fold_transformer_does_not_learn_validation_categories(self):
        train, validation = self.raw.iloc[:200], self.raw.iloc[200:].copy()
        validation["channel_sales"] = "UNSEEN_VALIDATION_CHANNEL"
        model = pipeline(LogisticRegression(max_iter=3000, random_state=42), linear=True)
        fitted = clone(model).fit(train, train.churn)
        encoder = fitted["preprocess"].named_transformers_["cat"]
        self.assertNotIn("UNSEEN_VALIDATION_CHANNEL", encoder.categories_[0])
        self.assertTrue(np.isfinite(fitted.predict_proba(validation)).all())
        self.assertNotIn("UNSEEN_VALIDATION_CHANNEL", encoder.categories_[0])


if __name__ == "__main__":
    unittest.main()
