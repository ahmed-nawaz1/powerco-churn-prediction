"""Read-only evidence validation, including immutable relocated historical runs."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from powerco.config import DATA_DIR, LEGACY_INPUT_PATHS, N_SPLITS, ROOT, project_path
from powerco.evaluation import metrics
from powerco.io import require, sha, write_json


def normalise(path: str) -> str:
    return path.replace("\\", "/")


def validate(run: str | Path) -> dict:
    """Verify hashes and recompute metrics; no fitting or parameter selection."""
    out = project_path(run)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    require("completed_utc" in manifest, "Experiment manifest is incomplete")
    version = manifest.get("schema_version", 1)
    checksums = json.loads((out / "checksums.json").read_text(encoding="utf-8"))
    for name, expected in checksums.items():
        path = (out / normalise(name)).resolve()
        require(path.is_relative_to(out), f"Unsafe checksum path: {name}")
        require(path.is_file() and sha(path) == expected, f"Changed result: {name}")
    for name, expected in manifest["input_sha256"].items():
        relocated = (
            LEGACY_INPUT_PATHS.get(normalise(name), normalise(name))
            if version == 1
            else normalise(name)
        )
        path = ROOT / relocated
        require(
            path.is_file() and sha(path) == expected, f"Changed input: {name} (now {relocated})"
        )
    archived_sources = []
    for name, expected in manifest["source_sha256"].items():
        name = normalise(name)
        current = ROOT / name
        if current.is_file() and sha(current) == expected:
            continue
        snapshot = out / "source" / (Path(name).name if version == 1 else name)
        require(
            snapshot.is_file() and sha(snapshot) == expected, f"Executed source unavailable: {name}"
        )
        archived_sources.append(name)
    source_evidence = "ARCHIVED_EXECUTED_SOURCE" if archived_sources else "CURRENT_SOURCE"
    clients = pd.read_csv(DATA_DIR / "client_data.csv")
    split = pd.read_csv(out / "split_assignments.csv")
    require(split.id.is_unique and split.row_index.is_unique, "Duplicate split IDs or row indices")
    require(set(split.id) == set(clients.id), "Split/customer ID mismatch")
    training = split[split.partition.eq("train")]
    testing = split[split.partition.eq("legacy_provisional_test")]
    require(len(training) + len(testing) == len(clients), "Partition row count mismatch")
    require(not set(training.id) & set(testing.id), "Train/test IDs overlap")
    require(set(training.validation_fold) == set(range(N_SPLITS)), "Five validation folds missing")
    require(testing.validation_fold.eq(-1).all(), "Test rows found inside CV folds")
    folds = pd.read_csv(out / "fold_metrics.csv")
    require(folds.groupby("model").size().eq(N_SPLITS).all(), "Incomplete per-model CV evidence")
    require(not folds.duplicated(["model", "fold"]).any(), "Duplicate model/fold result")
    require(
        len(pd.read_csv(out / "search_results.csv")) == 12, "RF search candidate count mismatch"
    )
    lock = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    require(lock["selected_before_new_test"] is True, "Selection was not frozen before test")
    predictions = pd.read_csv(out / "provisional_test_predictions.csv")
    require(
        predictions.row_index.is_unique and set(predictions.row_index) == set(testing.row_index),
        "Saved predictions/test split mismatch",
    )
    raw = clients.loc[predictions.row_index]
    require(
        np.array_equal(raw.churn.to_numpy(), predictions.churn.to_numpy()),
        "Saved labels differ from input",
    )
    require(
        np.array_equal(raw.id.to_numpy(), predictions.id.to_numpy()), "Saved IDs differ from input"
    )
    require(
        np.isfinite(predictions.probability).all() and predictions.probability.between(0, 1).all(),
        "Invalid probability output",
    )
    recalculated = metrics(predictions.churn, predictions.probability, lock["threshold"])
    table = pd.read_csv(out / "provisional_test_metrics.csv")
    selected = table[
        (table.model == lock["model"]) & table.operating_point.eq("training_selected_threshold")
    ]
    require(len(selected) == 1, "Missing or duplicate selected-model test result")
    for key, value in recalculated.items():
        require(
            np.isclose(selected.iloc[0][key], value, rtol=1e-12, atol=1e-12),
            f"Incorrect metric: {key}",
        )
    expected_labels = (predictions.probability >= lock["threshold"]).astype(int)
    require(
        np.array_equal(predictions.prediction, expected_labels), "Incorrect saved decision labels"
    )
    model = joblib.load(out / "selected_model.joblib")
    require(
        model["threshold"] == lock["threshold"] and model["model"] == lock["model"],
        "Model/selection mismatch",
    )
    require(
        np.allclose(
            model["pipeline"].predict_proba(raw)[:, 1],
            predictions.probability,
            rtol=1e-12,
            atol=1e-12,
        ),
        "Reloaded model predictions differ",
    )
    require(len(list((out / "figures").glob("*.png"))) == 10, "Ten expected figures not found")
    require((out / "git-log.txt").is_file(), "Git availability/history record missing")
    return {
        "status": "PASS",
        "checksummed_result_files": len(checksums),
        "split_rows": len(split),
        "models": folds.model.nunique(),
        "selected": lock["model"],
        "test_f1_recomputed": recalculated["f1"],
        "test_status": "PROVISIONAL",
        "source_evidence": source_evidence,
        "archived_sources": archived_sources,
        "checks": [
            "input/executed-source/output hashes",
            "unique disjoint split and five folds",
            "12 search candidates",
            "source labels and saved decisions",
            "saved metrics recomputed",
            "model reload/prediction consistency",
            "ten figures",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    result = validate(args.run)
    if args.save:
        destination = project_path(args.run) / "validation.json"
        require(not destination.exists(), f"Validation evidence already exists: {destination}")
        write_json(destination, result)
    print(json.dumps(result, indent=2))
