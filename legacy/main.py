
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# ==========================================
# STEP 1: CONFIGURATION
# ==========================================

DATASET_PATH = "client_data.csv"
PRICE_DATASET_PATH = "price_data.csv"

TARGET_COLUMN = "churn"
ID_COLUMN = "id"

DATE_COLUMNS = [
    "date_activ",
    "date_end",
    "date_modif_prod",
    "date_renewal",
]

ONEHOT_COLUMNS = [
    "channel_sales",
    "origin_up",
]

PRICE_COLUMNS = [
    "price_off_peak_var",
    "price_peak_var",
    "price_mid_peak_var",
    "price_off_peak_fix",
    "price_peak_fix",
    "price_mid_peak_fix",
]


# ==========================================
# STEP 2: LOAD DATASETS
# ==========================================

df = pd.read_csv(DATASET_PATH)
price_df = pd.read_csv(PRICE_DATASET_PATH)

print("\n========== DATASET ==========")
print("Total Customers:", len(df))
print("Customer Data Shape:", df.shape)
print("Price Data Shape:", price_df.shape)


# ==========================================
# STEP 3: INSPECT PRICING DATA
# ==========================================

print("\n========== PRICE DATA ==========")

print("Unique Customers:", price_df["id"].nunique())
print("Missing Values:", price_df.isna().sum().sum())

price_dates = pd.to_datetime(
    price_df["price_date"],
    errors="coerce",
)

print("Earliest Price Date:", price_dates.min())
print("Latest Price Date:", price_dates.max())


# ==========================================
# STEP 4: PRICING FEATURE ENGINEERING
# ==========================================

# Calculate average pricing per customer.
# These features are exploratory until the
# prediction cutoff is confirmed.

price_features = (
    price_df.groupby("id")[PRICE_COLUMNS]
    .mean()
    .add_prefix("avg_")
    .reset_index()
)

print("\n========== PRICE FEATURES ==========")
print("Shape:", price_features.shape)
print(
    "Duplicate IDs:",
    price_features["id"].duplicated().sum(),
)


# ==========================================
# STEP 5: MERGE DATASETS
# ==========================================

merged_df = df.merge(
    price_features,
    on="id",
    how="left",
    validate="one_to_one",
)

print("\n========== MERGE VALIDATION ==========")
print("Original Customers:", len(df))
print("Merged Customers:", len(merged_df))
print(
    "Customers Without Pricing:",
    merged_df["avg_price_off_peak_var"].isna().sum(),
)
print("Merged Shape:", merged_df.shape)

assert len(merged_df) == len(df)
assert merged_df["id"].duplicated().sum() == 0

# IMPORTANT:
# Do not use the merged pricing features for
# training until their timing is verified.


# ==========================================
# STEP 6: DATE VERIFICATION
# ==========================================

print("\n========== DATE VERIFICATION ==========")

for col in DATE_COLUMNS:
    dates = pd.to_datetime(df[col], errors="coerce")

    print(f"\n{col}")
    print("Earliest:", dates.min())
    print("Latest:", dates.max())
    print("Missing:", dates.isna().sum())


# ==========================================
# STEP 7: FEATURES AND TARGET
# ==========================================

X = df.drop(
    columns=[TARGET_COLUMN, ID_COLUMN]
)

y = df[TARGET_COLUMN]


# ==========================================
# STEP 8: TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

X_train = X_train.copy()
X_test = X_test.copy()

print("\n========== TRAIN / TEST SPLIT ==========")
print("Training Customers:", len(X_train))
print("Testing Customers:", len(X_test))


# ==========================================
# STEP 9: REMOVE RAW DATE COLUMNS
# ==========================================

# No arbitrary prediction date is assumed.

X_train = X_train.drop(columns=DATE_COLUMNS)
X_test = X_test.drop(columns=DATE_COLUMNS)


# ==========================================
# STEP 10: BINARY ENCODING
# ==========================================

def encode_binary_features(data):
    """Convert has_gas into numerical values."""

    data = data.copy()

    data["has_gas"] = data["has_gas"].map({
        "t": 1,
        "f": 0,
    })

    return data


X_train = encode_binary_features(X_train)
X_test = encode_binary_features(X_test)


# ==========================================
# STEP 11: ONE-HOT ENCODING
# ==========================================

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False,
)

# Fit only on training data
encoder.fit(X_train[ONEHOT_COLUMNS])


def apply_onehot_encoding(data, fitted_encoder):
    """Apply fitted categorical encoder."""

    data = data.copy()

    encoded_array = fitted_encoder.transform(
        data[ONEHOT_COLUMNS]
    )

    encoded_df = pd.DataFrame(
        encoded_array,
        columns=fitted_encoder.get_feature_names_out(),
        index=data.index,
    )

    data = data.drop(columns=ONEHOT_COLUMNS)

    return pd.concat(
        [data, encoded_df],
        axis=1,
    )


X_train = apply_onehot_encoding(X_train, encoder)
X_test = apply_onehot_encoding(X_test, encoder)


# ==========================================
# STEP 12: DATA QUALITY VALIDATION
# ==========================================

print("\n========== FINAL VALIDATION ==========")

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

print(
    "Train Missing Values:",
    X_train.isna().sum().sum(),
)
print(
    "Test Missing Values:",
    X_test.isna().sum().sum(),
)

assert X_train.columns.equals(X_test.columns)
assert X_train.isna().sum().sum() == 0
assert X_test.isna().sum().sum() == 0

assert np.isfinite(
    X_train.to_numpy(dtype=float)
).all()

assert np.isfinite(
    X_test.to_numpy(dtype=float)
).all()

assert len(X_train) == len(y_train)
assert len(X_test) == len(y_test)

print("Preprocessing Validation: PASSED")

print("\nTarget Distribution:")
print(y_train.value_counts(normalize=True).round(4))

print("\nConstant Features:")
constant_cols = X_train.columns[
    X_train.nunique(dropna=False) <= 1
].tolist()

print(constant_cols)


# ==========================================
# STEP 13: DEFINE THREE ML MODELS
# ==========================================

models = {

    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        ),
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),

    "Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=15,
        random_state=42,
    ),
}


# ==========================================
# STEP 14: TRAIN MODELS
# ==========================================

print("\n========== MODEL TRAINING ==========")

for name, model in models.items():

    print(f"\nTraining: {name}")

    model.fit(X_train, y_train)

    print("Status: COMPLETE")


# ==========================================
# STEP 15: MODEL EVALUATION
# ==========================================

def evaluate_model(name, model, features, target):
    """Evaluate a trained classification model."""

    predictions = model.predict(features)

    probabilities = model.predict_proba(
        features
    )[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(
            target, predictions
        ),
        "Precision": precision_score(
            target, predictions, zero_division=0
        ),
        "Recall": recall_score(
            target, predictions, zero_division=0
        ),
        "F1": f1_score(
            target, predictions, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(
            target, probabilities
        ),
        "PR_AUC": average_precision_score(
            target, probabilities
        ),
    }

    print(f"\n========== {name.upper()} ==========")

    for metric, value in metrics.items():
        if metric != "Model":
            print(f"{metric}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(target, predictions))

    return metrics


# ==========================================
# STEP 16: COMPARE MODELS
# ==========================================

results = []

for name, model in models.items():

    result = evaluate_model(
        name,
        model,
        X_test,
        y_test,
    )

    results.append(result)


results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")

print(
    results_df.round(4).to_string(index=False)
)

print("\n========== IMPORTANT NOTES ==========")
print("Results are provisional baselines.")
print("Cross-validation and tuning are pending.")
print("Prediction cutoff is not yet verified.")
print("Pricing features are not yet used in training.")
print("No models or results have been saved.")


# ==========================================
# STAGE 4: CROSS-VALIDATION
# ==========================================

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

# Use original, unencoded training features.
# X and y were created before preprocessing.
X_cv = X.loc[X_train.index].copy()
y_cv = y.loc[X_train.index].copy()

X_cv = X_cv.drop(columns=DATE_COLUMNS)

def encode_gas(data):
    return data.replace({"t": 1, "f": 0})

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            ONEHOT_COLUMNS,
        ),
        (
            "gas",
            FunctionTransformer(encode_gas),
            ["has_gas"],
        ),
    ],
    remainder="passthrough",
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

print("\n========== CROSS-VALIDATION ==========")

for name, model in models.items():

    # Rebuild preprocessing inside each CV fold.
    pipeline = make_pipeline(
        preprocessor,
        model,
    )

    scores = cross_validate(
        pipeline,
        X_cv,
        y_cv,
        cv=cv,
        scoring={
            "f1": "f1",
            "roc_auc": "roc_auc",
            "average_precision": "average_precision",
        },
        n_jobs=-1,
    )

    print(f"\n{name}")
    print("Mean F1:", round(scores["test_f1"].mean(), 4))
    print("Mean ROC-AUC:", round(scores["test_roc_auc"].mean(), 4))
    print(
        "Mean PR-AUC:",
        round(scores["test_average_precision"].mean(), 4),
    )

    # ==========================================
# STAGE 4: HYPERPARAMETER OPTIMISATION
# ==========================================

from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import clone

# Build a fresh pipeline to prevent
# preprocessing leakage during CV.
rf_pipeline = make_pipeline(
    clone(preprocessor),
    RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    ),
)

param_distributions = {
    "randomforestclassifier__n_estimators": [100, 200, 300],
    "randomforestclassifier__max_depth": [6, 10, 14, None],
    "randomforestclassifier__min_samples_leaf": [2, 5, 10],
    "randomforestclassifier__max_features": ["sqrt", 0.7],
}

search = RandomizedSearchCV(
    estimator=rf_pipeline,
    param_distributions=param_distributions,
    n_iter=12,
    scoring="f1",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    refit=True,
)

print("\n========== RANDOM FOREST TUNING ==========")

search.fit(X_cv, y_cv)

print("Best CV F1:", round(search.best_score_, 4))
print("Best Parameters:")
print(search.best_params_)

print("\n========== TUNED MODEL TEST ==========")

# Evaluate only after choosing parameters using CV.
X_test_raw = X.loc[X_test.index].copy()
X_test_raw = X_test_raw.drop(columns=DATE_COLUMNS)

tuned_rf = search.best_estimator_

evaluate_model(
    "TUNED RANDOM FOREST",
    tuned_rf,
    X_test_raw,
    y_test,
)

# ==========================================
# STAGE 4: OVERFITTING ANALYSIS
# ==========================================

from sklearn.model_selection import cross_validate

print("\n========== OVERFITTING ANALYSIS ==========")

overfit_scores = cross_validate(
    tuned_rf,
    X_cv,
    y_cv,
    cv=cv,
    scoring={
        "f1": "f1",
        "roc_auc": "roc_auc",
    },
    return_train_score=True,
    n_jobs=-1,
)

for metric in ["f1", "roc_auc"]:

    train_score = overfit_scores[
        f"train_{metric}"
    ].mean()

    validation_score = overfit_scores[
        f"test_{metric}"
    ].mean()

    print(f"\n{metric.upper()}")
    print("Training:", round(train_score, 4))
    print("Validation:", round(validation_score, 4))
    print("Gap:", round(train_score - validation_score, 4))

    # ==========================================
# STAGE 4: OVERFITTING REDUCTION EXPERIMENT
# ==========================================

from sklearn.base import clone

simpler_rf = make_pipeline(
    clone(preprocessor),
    RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    ),
)

simpler_scores = cross_validate(
    simpler_rf,
    X_cv,
    y_cv,
    cv=cv,
    scoring={
        "f1": "f1",
        "roc_auc": "roc_auc",
    },
    return_train_score=True,
    n_jobs=-1,
)

print("\n========== SIMPLER RANDOM FOREST ==========")

for metric in ["f1", "roc_auc"]:

    train_score = simpler_scores[
        f"train_{metric}"
    ].mean()

    validation_score = simpler_scores[
        f"test_{metric}"
    ].mean()

    print(f"\n{metric.upper()}")
    print("Training:", round(train_score, 4))
    print("Validation:", round(validation_score, 4))
    print("Gap:", round(train_score - validation_score, 4))