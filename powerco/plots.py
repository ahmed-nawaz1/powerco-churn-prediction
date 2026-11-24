"""Noninteractive plot export for evidence and reports."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def evaluation_figures(out, means, curve, threshold, winner, y_test, probabilities):
    """Export diagnostics without changing any model or threshold selection."""
    figures = out / "figures"
    plt.figure(figsize=(7, 4))
    means[["train_f1", "test_f1"]].plot.bar(ax=plt.gca())
    plt.ylabel("F1 at 0.5")
    plt.xticks(rotation=30, ha="right")
    plt.title("Training/CV gap - selected/tuned CV estimates are optimistic")
    savefig(figures / "05_overfitting.png")
    plt.figure(figsize=(7, 4))
    for column in ["precision", "recall", "f1"]:
        plt.plot(curve.threshold, curve[column], label=column)
    plt.axvline(threshold, color="black", ls="--", label=f"Frozen threshold {threshold:.2f}")
    plt.xlabel("Threshold")
    plt.ylabel("Pooled OOF metric")
    plt.legend()
    plt.title("Training threshold selection (selection-biased evidence)")
    savefig(figures / "06_threshold.png")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, probability in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, probability)
        precision, recall, _ = precision_recall_curve(y_test, probability)
        axes[0].plot(fpr, tpr, label=name)
        axes[1].plot(recall, precision, label=name)
    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate")
    axes[1].axhline(y_test.mean(), color="black", ls="--")
    axes[1].set(xlabel="Recall", ylabel="Precision")
    axes[1].legend(fontsize=6)
    figure.suptitle("Legacy test ROC and PR curves - PROVISIONAL")
    savefig(figures / "07_test_curves.png")
    probability = probabilities[winner]
    matrix = confusion_matrix(y_test, probability >= threshold, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            plt.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > matrix.max() / 2 else "black",
            )
    plt.xticks([0, 1], ["Retain", "Churn"])
    plt.yticks([0, 1], ["Retain", "Churn"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{winner}: provisional test, threshold {threshold:.2f}")
    savefig(figures / "08_confusion.png")
    calibration = pd.DataFrame({"score": probability, "y": y_test.to_numpy()})
    calibration["bin"] = pd.cut(calibration.score, np.linspace(0, 1, 11), include_lowest=True)
    calibration.groupby("bin", observed=True).agg(
        n=("y", "size"), observed=("y", "mean"), mean_score=("score", "mean")
    ).to_csv(out / "calibration_diagnostic.csv")
    service = pd.read_csv(out / "segment_oof_metrics.csv")
    service[service.segment.eq("has_gas")].plot.bar(
        x="value", y=["recall", "false_positive_rate"], figsize=(6, 4)
    )
    plt.ylabel("Rate")
    plt.title("OOF service-segment diagnostics (not protected-group fairness)")
    savefig(figures / "10_segments.png")
