"""Predeclared candidates and RF search; no test-set-driven configuration."""

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV

from powerco.config import SEED
from powerco.preprocessing import pipeline

SCORES = {
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
}


def candidates() -> dict:
    """Return fresh estimator instances for every execution."""
    forest = {"class_weight": "balanced", "random_state": SEED, "n_jobs": 1}
    boosting = {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 15, "random_state": SEED}
    return {
        "Dummy": pipeline(DummyClassifier(strategy="prior")),
        "LogisticRegression": pipeline(
            LogisticRegression(class_weight="balanced", max_iter=3000, random_state=SEED),
            linear=True,
        ),
        "RandomForest": pipeline(
            RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5, **forest)
        ),
        "HistGradientBoosting": pipeline(HistGradientBoostingClassifier(**boosting)),
        "BalancedHistGradientBoosting": pipeline(
            HistGradientBoostingClassifier(class_weight="balanced", **boosting)
        ),
        "SimplerRandomForest": pipeline(
            RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=20, **forest)
        ),
    }


def forest_search(cv: list, jobs: int) -> RandomizedSearchCV:
    return RandomizedSearchCV(
        pipeline(RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=1)),
        {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [6, 10, 14, None],
            "model__min_samples_leaf": [2, 5, 10],
            "model__max_features": ["sqrt", 0.7],
        },
        n_iter=12,
        scoring="f1",
        cv=cv,
        random_state=SEED,
        n_jobs=jobs,
        refit=True,
        error_score="raise",
        verbose=1,
    )
