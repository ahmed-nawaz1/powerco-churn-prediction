"""Fixed-fold diagnostic explanations, never causal feature attribution."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance

from powerco.config import SEED
from powerco.features import CATEGORICAL, RAW_FEATURES
from powerco.plots import savefig


def explain(selected, X, y, cv, threshold, out, jobs):
    # Fixed first fold: explanations are diagnostic, not an independent performance test.
    tr, va = cv[0]
    model = clone(selected).fit(X.iloc[tr], y.iloc[tr])
    raw = X.iloc[va]
    yp = y.iloc[va]
    importance = permutation_importance(
        model,
        raw[RAW_FEATURES],
        yp,
        scoring="average_precision",
        n_repeats=5,
        random_state=SEED,
        n_jobs=jobs,
    )
    frame = pd.DataFrame(
        {
            "feature": RAW_FEATURES,
            "mean_ap_decrease": importance.importances_mean,
            "std_over_permutations": importance.importances_std,
        }
    ).sort_values("mean_ap_decrease", ascending=False)
    frame.to_csv(out / "permutation_importance.csv", index=False)
    frame.sort_values("mean_ap_decrease").plot.barh(
        x="feature",
        y="mean_ap_decrease",
        xerr="std_over_permutations",
        legend=False,
        figsize=(8, 5),
        color="#287891",
    )
    plt.xlabel("Decrease in validation average precision")
    plt.title("Fixed fold 1: raw-feature permutation importance")
    savefig(out / "figures" / "09_importance.png")
    prob = model.predict_proba(raw)[:, 1]
    pred = prob >= threshold
    errors = np.where(
        pred, np.where(yp.to_numpy() == 1, "TP", "FP"), np.where(yp.to_numpy() == 1, "FN", "TN")
    )
    rows = []
    # One observed example of each available outcome; no invented examples.
    for outcome in ["TP", "FP", "FN", "TN"]:
        candidates = np.flatnonzero(errors == outcome)
        if not len(candidates):
            continue
        i = int(candidates[np.argmax(prob[candidates])])
        example = raw.iloc[[i]].copy()
        for col in RAW_FEATURES:
            changed = example.copy()
            replacement = (
                X.iloc[tr][col].mode().iloc[0] if col in CATEGORICAL else X.iloc[tr][col].median()
            )
            changed[col] = replacement
            alternative = float(model.predict_proba(changed)[0, 1])
            rows.append(
                dict(
                    row_index=int(example.index[0]),
                    outcome=outcome,
                    feature=col,
                    observed_value=str(example[col].iloc[0]),
                    replacement=str(replacement),
                    original_score=float(prob[i]),
                    replacement_score=alternative,
                    score_change=alternative - float(prob[i]),
                )
            )
    pd.DataFrame(rows).to_csv(out / "local_prediction_sensitivity.csv", index=False)
