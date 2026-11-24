# PowerCo Model Code

This folder contains the active implementation. Start from the project root
with `.\run.ps1`; normally you do not need to open every file.

## Simple flow

```text
main.py
  -> workflow.py
  -> data.py
  -> eda.py
  -> features.py + preprocessing.py
  -> models.py
  -> evaluation.py + explain.py + plots.py
  -> validate.py + reporting.py
```

## File guide

| File | Simple purpose |
|---|---|
| `config.py` | Stores paths, random seed and fixed experiment settings. |
| `data.py` | Loads both CSV files and checks their structure. |
| `eda.py` | Creates training-data and pricing exploratory evidence. |
| `features.py` | Selects safe inputs and creates the consumption ratio. |
| `preprocessing.py` | Handles missing values, categories and scaling inside CV. |
| `models.py` | Defines the baseline, alternative models and RF search. |
| `workflow.py` | Runs the seven assessment stages in order. |
| `evaluation.py` | Calculates metrics, errors and customer-segment results. |
| `explain.py` | Creates feature-importance and local sensitivity evidence. |
| `plots.py` | Saves the ten required charts. |
| `validate.py` | Checks hashes, saved metrics, splits and model reload. |
| `reporting.py` | Writes the readable `SUMMARY.txt` output. |
| `io.py` | Shared logging, JSON, hashing and safety helpers. |

## What the run saves

Every successful run creates `artifacts/runs/<run_id>/` with the model, metrics,
figures, predictions, error analysis, segment analysis, source snapshot,
validation files and a human-readable summary. The report PDF is separate.

## The only command you need

```powershell
.\run.ps1
```

