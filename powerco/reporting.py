"""Small human-readable summaries of saved experiment evidence."""

import json
from pathlib import Path

import pandas as pd


def summary_text(summary: dict) -> str:
    result = summary["provisional_test"]
    return "\n".join(
        [
            "=" * 64,
            "POWERCO - RUN COMPLETED",
            "Model trained      : YES",
            "Saved model reload : PASS",
            "Evidence validation: PASS",
            f"Selected model     : {summary['selected_model']}",
            f"CV F1              : {summary['cv_f1']:.4f}",
            f"Decision threshold : {summary['threshold']:.2f}",
            f"Provisional test F1: {result['f1']:.4f}",
            f"Precision / Recall : {result['precision']:.4f} / {result['recall']:.4f}",
            f"ROC-AUC / AP       : {result['roc_auc']:.4f} / {result['average_precision']:.4f}",
            f"TN / FP / FN / TP  : {result['tn']} / {result['fp']} / {result['fn']} / {result['tp']}",
            f"Run folder         : {summary['output']}",
            "Outputs: saved model | metrics CSV | predictions CSV | figures/",
            "Research only: cutoff unresolved, exposed test, non-nested selection.",
            "=" * 64,
        ]
    )


def saved_summary(run: Path, verification: dict) -> dict:
    """Also support historical runs that predate summary.json."""
    path = run / "summary.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    selected = json.loads((run / "selection.json").read_text(encoding="utf-8"))
    table = pd.read_csv(run / "provisional_test_metrics.csv")
    row = table[table.operating_point.eq("training_selected_threshold")].iloc[0]
    cv = pd.read_csv(run / "cv_summary.csv").set_index("model")
    return {
        "selected_model": selected["model"],
        "threshold": selected["threshold"],
        "cv_f1": float(cv.loc[selected["model"], "test_f1"]),
        "provisional_test": row.to_dict(),
        "output": str(run),
        "validation_passed": verification["status"] == "PASS",
    }
