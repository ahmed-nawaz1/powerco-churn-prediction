"""Input schema and structural quality checks; no silent duplicate removal."""

import pandas as pd

from powerco.config import DATA_DIR
from powerco.io import require, write_json


def load_datasets():
    """Load original CSVs; validate fields before expensive training starts."""
    from powerco.features import RAW_FEATURES

    clients = pd.read_csv(DATA_DIR / "client_data.csv")
    prices = pd.read_csv(DATA_DIR / "price_data.csv")
    required_clients = set(
        RAW_FEATURES + ["id", "churn", "date_activ", "date_end", "date_modif_prod", "date_renewal"]
    )
    required_prices = {
        "id",
        "price_date",
        "price_off_peak_var",
        "price_peak_var",
        "price_mid_peak_var",
        "price_off_peak_fix",
        "price_peak_fix",
        "price_mid_peak_fix",
    }
    require(
        required_clients.issubset(clients.columns),
        f"Missing client columns: {sorted(required_clients - set(clients.columns))}",
    )
    require(
        required_prices.issubset(prices.columns),
        f"Missing pricing columns: {sorted(required_prices - set(prices.columns))}",
    )
    require(not clients["id"].isna().any(), "Customer IDs must not be missing")
    return clients, prices


def structure(c, p, out):
    info = {}
    for name, data, keys in [("clients", c, ["id"]), ("prices", p, ["id", "price_date"])]:
        info[name] = dict(
            rows=len(data),
            columns=len(data.columns),
            unique_ids=data.id.nunique(),
            row_duplicates=int(data.duplicated().sum()),
            key_duplicates=int(data.duplicated(keys).sum()),
            missing=data.isna().sum().to_dict(),
            dtypes=data.dtypes.astype(str).to_dict(),
            literal_missing={
                col: int(data[col].eq("MISSING").sum())
                for col in data.select_dtypes(include=["object", "str"])
            },
        )
        require(
            data.duplicated(keys).sum() == 0,
            f"{name}: duplicate keys need review; nothing silently deleted",
        )
    require(
        set(c.churn.unique()) == {0, 1},
        "churn must contain both classes 0 and 1 without missing labels",
    )
    dates = {}
    for col in ["date_activ", "date_end", "date_modif_prod", "date_renewal"]:
        d = pd.to_datetime(c[col], errors="coerce")
        dates[col] = dict(min=str(d.min()), max=str(d.max()), invalid=int(d.isna().sum()))
    d = pd.to_datetime(p.price_date, errors="coerce")
    dates["price_date"] = dict(min=str(d.min()), max=str(d.max()), invalid=int(d.isna().sum()))
    info["date_ranges"] = dates
    info["cutoff"] = "NOT ESTABLISHED; data ranges are not prediction time"
    info["target_counts"] = c.churn.value_counts().sort_index().to_dict()
    info["price_only_ids"] = len(set(p.id) - set(c.id))
    info["clients_without_price"] = len(set(c.id) - set(p.id))
    write_json(out / "data_quality.json", info)
    return info
