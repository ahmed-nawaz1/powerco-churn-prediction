# Project structure and execution contract

```text
main.py                  # Simple official entry point; runs the full experiment
run.ps1                  # One command, using the local .venv without activation
powerco/
  config.py              # Paths, fixed seed and CV settings
  data.py                # Load CSVs and check schema, keys and data quality
  features.py            # Stateless historical feature policy and derived ratio
  preprocessing.py       # Fold-local imputation, encoding and LR scaling
  models.py              # Fresh candidates, scoring rules and fixed RF search
  workflow.py            # Seven-stage orchestration; source/data provenance
  evaluation.py          # Metrics, error definitions and service-segment checks
  explain.py             # Fixed-fold global and local diagnostic explanations
  plots.py               # Saved charts; no interactive display requirement
  validate.py            # Hashes, split checks, metric recalculation, model reload
  reporting.py           # Concise console/file result summaries
data/raw/                # Unchanged supplied CSVs
docs/assessment/         # Original brief, dictionary, cover sheet and AI guidance
docs/audit/              # Contemporaneous requirement and methodology records
artifacts/runs/<run_id>/ # Versioned model, metrics, predictions, figures and log
reports/                 # Separately generated report drafts and PDFs
tools/                   # Report builder
tests/                   # Feature, entrypoint and integrity tests
legacy/                  # Original script and previous implementation preserved
```

## One run

`main.py` directly calls the seven-stage workflow. It performs data validation,
training-only EDA, fold-local preprocessing, fixed CV/search/threshold selection,
locked final fits, provisional test evaluation, diagnostic explanations and
evidence validation. It uses the existing protocol and feature policy; the
workflow does not tune from test results. Model selection remains non-nested and
the test remains exposed.

Each run creates a new directory and refuses overwrites. Progress goes to the
terminal and `run.log`; `status.json` records RUNNING, COMPLETED, FAILED or
INTERRUPTED. Only a validated complete run updates `artifacts/latest_run.json`.
`summary.json` and `SUMMARY.txt` make model/threshold/metrics easy to inspect.

## Integrity and reproducibility

Source snapshots preserve paths. Inputs, source and immutable outputs are hashed.
Dynamic status/log/summary/validation files are explicitly excluded from the
experiment checksum set to avoid circular checksums; archive hashes include them.
Historical schema-1 input paths map to their new locations without changing data.
If live source has changed, the validator verifies the recorded executed snapshot
and prints ARCHIVED_EXECUTED_SOURCE rather than claiming current-source identity.

This is a professional research project structure, not a production serving system.
The cutoff and point-in-time provenance remain unresolved; pricing is EDA-only.
Student authorship/AI declaration and genuine Git history limitations remain in the
assessment audit. Paths in historical reports/audits are contemporaneous references;
use this document and README for the current layout.
