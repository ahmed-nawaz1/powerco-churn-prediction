# PowerCo Customer Churn Prediction

This is the CPS7008 Assessment 1 machine-learning project for predicting
PowerCo customer churn.

## Run the model

Open PowerShell in this folder and run one command:

```powershell
.\run.ps1
```

This runs all seven assessment stages and saves a new result folder in
`artifacts/runs/`. The result folder contains the selected model, metrics,
figures, predictions, error analysis, validation evidence and `SUMMARY.txt`.

You can also run it directly:

```powershell
.\.venv\Scripts\python.exe main.py
```

The project never overwrites an earlier run. Every run receives a timestamped
folder.

## Installation

The project uses Python 3.13 and the pinned libraries in `requirements.txt`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\run.ps1
```

## Simple project map

```text
main.py                 The only file you need to run
run.ps1                 One-command PowerShell launcher
powerco/                Active ML code used by main.py
data/raw/               Supplied customer and pricing CSV files
artifacts/runs/         Results created after each model run
docs/assessment/        Teacher brief and supplied assessment documents
docs/audit/             Method and limitation records
reports/                Separate assignment report drafts/PDFs
tests/                  Automatic code checks
tools/build_report.py   Report builder for the later report stage
legacy/                 Preserved old code; not used by main.py
```

You only need to understand `main.py`, `powerco/workflow.py` and the simple
file guide inside `powerco/README.md`. The smaller files inside `powerco/`
keep individual assessment stages readable and do not change the output
contract.

## Seven stages saved by each run

1. Load and validate the customer and pricing data.
2. Produce training-only EDA and pricing evidence.
3. Apply leakage-safe preprocessing and feature engineering.
4. Train Logistic Regression, Random Forest and Gradient Boosting models.
5. Compare models with five-fold cross-validation and tune Random Forest.
6. Evaluate metrics, errors, segments and explanations.
7. Save figures, metrics, predictions, validation files and the selected model.

## Important limitations

The prediction cutoff is not verified, so historical pricing remains EDA-only.
The supplied legacy test set was previously inspected, so its metrics are
provisional. Do not tune the model using those test results.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
