"""Classification metrics and service-segment diagnostics."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def metrics(y, prob, threshold=0.5):
    pred = (np.asarray(prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(
        accuracy=accuracy_score(y, pred),
        precision=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred, zero_division=0),
        f1=f1_score(y, pred, zero_division=0),
        roc_auc=roc_auc_score(y, prob) if len(np.unique(y)) == 2 else None,
        average_precision=average_precision_score(y, prob) if np.sum(y) else None,
        brier=brier_score_loss(y, prob),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
        n=len(y),
        positives=int(np.sum(y)),
        threshold=float(threshold),
    )


def segments(raw, y, prob, threshold, out):
    data = raw.copy()
    data["tenure_band"] = pd.cut(
        data.num_years_antig, [-np.inf, 3, 6, np.inf], labels=["<=3 years", "4-6 years", ">6 years"]
    )
    rows = []
    for col in ["has_gas", "channel_sales", "tenure_band"]:
        for value, idx in data.groupby(col, observed=True).groups.items():
            pos = data.index.get_indexer(idx)
            row = metrics(np.asarray(y)[pos], np.asarray(prob)[pos], threshold)
            row.update(segment=col, value=str(value), small_positive_sample=row["positives"] < 30)
            row["false_positive_rate"] = row["fp"] / max(1, row["fp"] + row["tn"])
            rows.append(row)
    pd.DataFrame(rows).to_csv(out / "segment_oof_metrics.csv", index=False)
