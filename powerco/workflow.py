"""Run the seven assessment stages with fixed seeds and auditable evidence."""

import importlib.metadata
import logging
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)

from powerco.config import ASSESSMENT_DIR, AUDIT_DIR, DATA_DIR, LATEST_RUN, N_SPLITS, ROOT, SEED
from powerco.data import load_datasets, structure
from powerco.eda import eda
from powerco.evaluation import metrics, segments
from powerco.explain import explain
from powerco.features import ENGINEERED, RAW_FEATURES, SnapshotFeatures
from powerco.io import configure_logging, require, sha, utc_now, write_json
from powerco.models import SCORES, candidates, forest_search
from powerco.plots import evaluation_figures

LOGGER = logging.getLogger("powerco")
MUTABLE_FILES = {
    "checksums.json",
    "run.log",
    "status.json",
    "summary.json",
    "SUMMARY.txt",
    "validation.json",
}


def snapshot_sources(out: Path, jobs: int) -> dict:
    """Snapshot executed source while preserving its package-relative paths."""
    sources = [
        ROOT / name
        for name in [
            "main.py",
            "run.ps1",
            "pyproject.toml",
            "requirements.txt",
            "requirements-report.txt",
        ]
    ]
    for directory in ["powerco", "tools", "tests"]:
        sources.extend(sorted((ROOT / directory).glob("*.py")))
    for source in sources:
        destination = out / "source" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    inputs = [
        DATA_DIR / "client_data.csv",
        DATA_DIR / "price_data.csv",
        ASSESSMENT_DIR / "cps7008-assessment-1.pdf",
        AUDIT_DIR / "PROTOCOL.md",
        ROOT / "legacy/main.py",
    ]
    manifest = {
        "schema_version": 2,
        "started_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "seed": SEED,
        "command": sys.argv,
        "jobs": jobs,
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in [
                "numpy",
                "pandas",
                "scikit-learn",
                "scipy",
                "matplotlib",
                "joblib",
                "threadpoolctl",
            ]
        },
        "input_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in inputs},
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
        "test_status": "LEGACY TEST ALREADY INSPECTED; ALL NEW TEST RESULTS PROVISIONAL",
        "temporal_status": "UNVERIFIED SNAPSHOT; PRICING/DATE/FORECAST/FINANCIAL FIELDS EXCLUDED",
        "excluded_mutable_files": sorted(MUTABLE_FILES),
    }
    write_json(out / "manifest.json", manifest)
    frozen = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    )
    (out / "environment-freeze.txt").write_text(frozen.stdout, encoding="utf-8")
    try:
        git = subprocess.run(
            ["git", "log", "--date=iso-strict", "--format=fuller", "--stat"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        history = (
            git.stdout
            if git.returncode == 0
            else (
                "No Git repository was present. Earlier commits cannot be recovered here.\n"
                + git.stderr
            )
        )
    except FileNotFoundError:
        history = "Git executable unavailable. No Git history has been fabricated.\n"
    (out / "git-log.txt").write_text(history, encoding="utf-8")
    return manifest


def make_split(clients, out):
    labels = clients.churn
    train, test = train_test_split(
        np.arange(len(clients)), test_size=0.2, stratify=labels, random_state=SEED
    )
    require(not set(train) & set(test), "Train/test rows overlap")
    features = clients.drop(columns="churn")
    training, target = features.iloc[train], labels.iloc[train]
    folds = list(
        StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(training, target)
    )
    assignment = clients[["id"]].copy()
    assignment["partition"] = "train"
    assignment.loc[test, "partition"] = "legacy_provisional_test"
    assignment["row_index"] = assignment.index
    assignment["validation_fold"] = -1
    for fold, (_, validation) in enumerate(folds):
        assignment.loc[training.iloc[validation].index, "validation_fold"] = fold
    assignment.to_csv(out / "split_assignments.csv", index=False)
    return features, labels, train, test, training, target, folds


def develop_models(training, target, folds, out, jobs):
    models = candidates()
    search = forest_search(folds, jobs)
    LOGGER.info("RF search: 12 candidates x 5 folds = 60 fits | workers=%d", jobs)
    search.fit(training, target)
    pd.DataFrame(search.cv_results_).to_csv(out / "search_results.csv", index=False)
    write_json(
        out / "search_best.json", {"params": search.best_params_, "mean_f1": search.best_score_}
    )
    LOGGER.info("RF search DONE | best CV F1=%.4f | %s", search.best_score_, search.best_params_)
    models["TunedRandomForest"] = clone(search.best_estimator_)
    rows = []
    for position, (name, estimator) in enumerate(models.items(), start=1):
        LOGGER.info("CV %d/%d | %s | training five folds...", position, len(models), name)
        scores = cross_validate(
            estimator,
            training,
            target,
            cv=folds,
            scoring=SCORES,
            return_train_score=True,
            n_jobs=jobs,
            error_score="raise",
        )
        for fold in range(N_SPLITS):
            row = {"model": name, "fold": fold, "fit_time": float(scores["fit_time"][fold])}
            row.update(
                {
                    key: float(values[fold])
                    for key, values in scores.items()
                    if key.startswith(("test_", "train_"))
                }
            )
            rows.append(row)
        pd.DataFrame(rows).to_csv(out / "fold_metrics.csv", index=False)
        LOGGER.info(
            "CV DONE | %-30s train F1=%.4f | validation F1=%.4f",
            name,
            scores["train_f1"].mean(),
            scores["test_f1"].mean(),
        )
    scores = pd.DataFrame(rows)
    means = scores.groupby("model", sort=False).mean(numeric_only=True).drop(columns="fold")
    means["validation_f1_std"] = scores.groupby("model").test_f1.std()
    means["train_validation_f1_gap"] = means.train_f1 - means.test_f1
    means.to_csv(out / "cv_summary.csv")
    return models, means


def freeze_selection(models, means, training, target, folds, out, jobs):
    winner = means.test_f1.idxmax()
    probabilities = cross_val_predict(
        models[winner], training, target, cv=folds, method="predict_proba", n_jobs=jobs
    )[:, 1]
    curve = pd.DataFrame(
        [
            metrics(target, probabilities, value)
            for value in np.round(np.arange(0.05, 0.951, 0.01), 2)
        ]
    )
    curve["distance_to_default"] = abs(curve.threshold - 0.5)
    chosen = curve.sort_values(
        ["f1", "distance_to_default", "threshold"], ascending=[False, True, False]
    ).iloc[0]
    threshold = float(chosen.threshold)
    curve.to_csv(out / "threshold_oof.csv", index=False)
    pd.DataFrame(
        {
            "row_index": training.index,
            "y": target.to_numpy(),
            "probability": probabilities,
            "prediction": (probabilities >= threshold).astype(int),
        }
    ).to_csv(out / "oof_predictions.csv", index=False)
    write_json(
        out / "selection.json",
        {
            "model": winner,
            "threshold": threshold,
            "selected_before_new_test": True,
            "selection": "training mean CV F1 at 0.5; training pooled OOF F1 threshold",
            "cv_selection_bias": "Search/model/threshold selection not nested; selected OOF estimates optimistic",
            "params": models[winner].get_params(deep=False),
            "frozen_utc": utc_now(),
        },
    )
    LOGGER.info(
        "SELECTION FROZEN | %s | threshold=%.2f | test not used for selection", winner, threshold
    )
    return winner, threshold, curve, probabilities


def evaluate_locked_models(
    models, training, target, test_features, test_target, clients, test, winner, threshold, out
):
    rows, probabilities, selected = [], {}, None
    for position, (name, estimator) in enumerate(models.items(), start=1):
        LOGGER.info(
            "Final fit %d/%d | %s | training %d customers...",
            position,
            len(models),
            name,
            len(training),
        )
        fitted = clone(estimator).fit(training, target)
        scores = fitted.predict_proba(test_features)[:, 1]
        probabilities[name] = scores
        rows.append({"model": name, "operating_point": "default", **metrics(test_target, scores)})
        LOGGER.info("TRAINED | %-30s | provisional test F1=%.4f", name, rows[-1]["f1"])
        if name == winner:
            selected = fitted
    scores = probabilities[winner]
    final_metrics = {
        "model": winner,
        "operating_point": "training_selected_threshold",
        **metrics(test_target, scores, threshold),
    }
    rows.append(final_metrics)
    pd.DataFrame(rows).to_csv(out / "provisional_test_metrics.csv", index=False)
    predictions = clients.iloc[test][["id"] + RAW_FEATURES + ["churn"]].copy()
    predictions["row_index"] = predictions.index
    predictions["probability"] = scores
    predictions["prediction"] = (scores >= threshold).astype(int)
    predictions["outcome"] = np.where(
        predictions.prediction.eq(1),
        np.where(predictions.churn.eq(1), "TP", "FP"),
        np.where(predictions.churn.eq(1), "FN", "TN"),
    )
    predictions.to_csv(out / "provisional_test_predictions.csv", index=False)
    errors = predictions[predictions.outcome.isin(["FP", "FN"])]
    pd.concat(
        [
            errors[errors.outcome.eq(kind)]
            .sort_values("probability", ascending=kind == "FN")
            .head(5)
            for kind in ["FP", "FN"]
        ]
    ).to_csv(out / "misclassified_examples.csv", index=False)
    joblib.dump(
        {
            "pipeline": selected,
            "threshold": threshold,
            "model": winner,
            "eligibility": "Conditional snapshot; not approved for deployment",
        },
        out / "selected_model.joblib",
    )
    loaded = joblib.load(out / "selected_model.joblib")
    require(
        np.array_equal(loaded["pipeline"].predict_proba(test_features)[:, 1], scores),
        "Reloaded model predictions differ",
    )
    return probabilities, final_metrics


def run(output: Path, jobs: int = 3) -> Path:
    """Validate a full run before publishing it as the latest successful model."""
    require(jobs >= 1, "--jobs must be at least 1")
    output.mkdir(parents=True, exist_ok=False)
    (output / "figures").mkdir()
    configure_logging(output / "run.log")
    started = time.perf_counter()
    state = {
        "status": "RUNNING",
        "started_utc": utc_now(),
        "stage": 0,
        "model_trained": False,
        "validation_passed": False,
    }

    def stage(number, message):
        state.update(stage=number, message=message, updated_utc=utc_now())
        write_json(output / "status.json", state)
        LOGGER.info("[%d/7] %s", number, message)

    try:
        LOGGER.info("PowerCo experiment | seed=%d | output=%s", SEED, output)
        LOGGER.info("Research limitation: legacy test exposed; prediction cutoff unresolved")
        manifest = snapshot_sources(output, jobs)
        stage(1, "Problem framing, input validation and fixed train/test split")
        clients, prices = load_datasets()
        structure(clients, prices, output)
        features, labels, train, test, training, target, folds = make_split(clients, output)
        LOGGER.info(
            "Data OK | customers=%d | prices=%d | train=%d | provisional test=%d",
            len(clients),
            len(prices),
            len(train),
            len(test),
        )
        stage(2, "Training-data EDA and pricing investigation (EDA only)")
        eda(clients.iloc[train], prices, output)
        stage(3, "Feature policy and fold-local preprocessing")
        pd.DataFrame(
            [
                {
                    "feature": column,
                    "use": (
                        "target"
                        if column == "churn"
                        else "conditional historical snapshot feature"
                        if column in RAW_FEATURES
                        else "join/split key only"
                        if column == "id"
                        else "excluded: as-of timing unverified"
                    ),
                    "timing_verified": False,
                }
                for column in clients.columns
            ]
        ).to_csv(output / "feature_eligibility.csv", index=False)
        stage(4, "Model training, five-fold CV, RF search and training-only threshold selection")
        models, means = develop_models(training, target, folds, output, jobs)
        winner, threshold, curve, oof = freeze_selection(
            models, means, training, target, folds, output, jobs
        )
        prepared = clone(models[winner][:-1]).fit_transform(training, target)
        before = SnapshotFeatures().fit_transform(training)
        require(np.isfinite(prepared).all(), "Nonfinite encoded training values")
        write_json(
            output / "preprocessing_checks.json",
            {
                "raw_rows": len(training),
                "selected_raw_features": len(RAW_FEATURES),
                "with_engineering": before.shape[1],
                "encoded_shape": list(prepared.shape),
                "engineered_missing_before_imputation": int(before[ENGINEERED].isna().sum()),
                "missing_after": int(np.isnan(prepared).sum()),
                "all_finite": True,
                "policy": "Median imputation and OHE/LR scaling fit within folds; extremes retained",
            },
        )
        stage(5, "Locked final fits and one provisional test evaluation phase")
        probabilities, final_metrics = evaluate_locked_models(
            models,
            training,
            target,
            features.iloc[test],
            labels.iloc[test],
            clients,
            test,
            winner,
            threshold,
            output,
        )
        state["model_trained"] = True
        stage(6, "Error/segment diagnostics and fixed-fold interpretability")
        segments(training, target.to_numpy(), oof, threshold, output)
        explain(models[winner], training, target, folds, threshold, output, jobs)
        evaluation_figures(
            output, means, curve, threshold, winner, labels.iloc[test], probabilities
        )
        stage(7, "Evidence validation and research-result summary")
        manifest.update(
            completed_utc=utc_now(),
            selected=winner,
            validation={
                "disjoint_split": True,
                "unique_keys": True,
                "finite_preprocessing": True,
                "model_reload_exact": True,
                "selection_before_test": True,
            },
        )
        write_json(output / "manifest.json", manifest)
        write_json(
            output / "checksums.json",
            {
                file.relative_to(output).as_posix(): sha(file)
                for file in sorted(output.rglob("*"))
                if file.is_file() and file.name not in MUTABLE_FILES
            },
        )
        from powerco.validate import validate

        verification = validate(output)
        write_json(output / "validation.json", verification)
        summary = {
            "status": "COMPLETED",
            "model_trained": True,
            "validation_passed": True,
            "selected_model": winner,
            "cv_f1": float(means.loc[winner, "test_f1"]),
            "threshold": threshold,
            "provisional_test": final_metrics,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "output": output.relative_to(ROOT).as_posix(),
            "limitations": [
                manifest["test_status"],
                manifest["temporal_status"],
                "CV/model/threshold selection is not nested",
            ],
        }
        write_json(output / "summary.json", summary)
        from powerco.reporting import summary_text

        (output / "SUMMARY.txt").write_text(summary_text(summary), encoding="utf-8")
        state.update(status="COMPLETED", validation_passed=True, completed_utc=utc_now())
        write_json(output / "status.json", state)
        write_json(
            LATEST_RUN, {"run": output.relative_to(ROOT).as_posix(), "completed_utc": utc_now()}
        )
        LOGGER.info("VALIDATION PASS | saved model reload and output consistency verified")
        print("\n" + summary_text(summary), flush=True)
        return output
    except BaseException as error:
        state.update(
            status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED",
            error=str(error),
            updated_utc=utc_now(),
        )
        write_json(output / "status.json", state)
        LOGGER.exception("Run did not complete. Evidence retained in %s", output)
        raise

