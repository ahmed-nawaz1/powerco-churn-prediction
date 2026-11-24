"""Training-partition EDA; historical prices remain descriptive only."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from powerco.io import require
from powerco.plots import savefig


def eda(train, prices, out):
    """Relationships use the legacy training rows only; prices are descriptive."""
    figs = out / "figures"
    train.describe(include="all").T.to_csv(out / "training_summary.csv")
    numeric = train.select_dtypes(include=np.number).drop(columns="churn")
    q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
    iqr = q3 - q1
    outliers = pd.DataFrame(
        {
            "iqr_low": q1 - 1.5 * iqr,
            "iqr_high": q3 + 1.5 * iqr,
            "outside_iqr_count": ((numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)).sum(),
            "negative_count": (numeric < 0).sum(),
            "zero_count": (numeric == 0).sum(),
        }
    )
    outliers.to_csv(out / "outlier_diagnostics.csv")
    corr = train.select_dtypes(include=np.number).corr(method="spearman")
    corr.to_csv(out / "spearman_correlations.csv")
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    plt.xticks(range(len(corr)), corr.columns, rotation=90, fontsize=6)
    plt.yticks(range(len(corr)), corr.columns, fontsize=6)
    plt.colorbar(label="Spearman correlation")
    plt.title("Training data correlations (association only)")
    savefig(figs / "02_correlations.png")
    train.churn.value_counts().sort_index().plot.bar(color=["#287891", "#df8646"], rot=0)
    plt.xticks([0, 1], ["Retained", "Churned"])
    plt.ylabel("Customers")
    plt.title("Training class imbalance")
    savefig(figs / "01_churn.png")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, col in zip(axes, ["cons_12m", "num_years_antig", "pow_max"]):
        for target in [0, 1]:
            v = train.loc[train.churn.eq(target), col]
            if col == "cons_12m":
                v = np.sign(v) * np.log1p(np.abs(v))
            ax.hist(v, bins=30, density=True, alpha=0.5, label=f"churn={target}")
        ax.set_title(col + (" (signed log1p)" if col == "cons_12m" else ""))
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    savefig(figs / "03_distributions.png")
    price_cols = [x for x in prices if x.startswith("price_") and x != "price_date"]
    aggregate = prices.groupby("id")[price_cols].agg(["mean", "std", "min", "max"])
    aggregate.columns = ["eda_" + "_".join(x) for x in aggregate.columns]
    merged = train.merge(aggregate, on="id", how="left", validate="one_to_one")
    require(len(merged) == len(train), "Exploratory price merge changed customer count")
    merged[["id", "churn"] + list(aggregate.columns)].to_csv(
        out / "pricing_exploratory_merge.csv", index=False
    )
    # Rank-based bins avoid unstable ties; duplicates are collapsed, never force fabricated bins.
    mean_price = "eda_price_off_peak_var_mean"
    merged["price_bin"] = pd.qcut(merged[mean_price], 4, duplicates="drop")
    relationship = merged.groupby("price_bin", observed=True).churn.agg(["size", "sum", "mean"])
    relationship.to_csv(out / "pricing_relationship.csv")
    relationship["mean"].plot.bar(rot=25, color="#287891", figsize=(8, 4))
    plt.ylabel("Observed churn proportion")
    plt.xlabel("Mean off-peak variable price quartile")
    plt.title("Training pricing association - EDA ONLY, cutoff unknown")
    savefig(figs / "04_pricing_eda.png")
    daily = prices[prices.id.isin(train.id)].groupby("price_date")[price_cols].mean()
    daily.to_csv(out / "price_history.csv")
    return relationship
