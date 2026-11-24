# CPS7008 PowerCo customer churn - research artefact

This preserves the original `main.py` and adds a separate reproducible workflow.
Read `audit/INITIAL_AUDIT.md` and `audit/PROTOCOL.md` first.
The report is an AI-assisted working draft for student review, not a signed
declaration of independent authorship or a deployment approval.

## Environment and reproduction (PowerShell)

The verified existing environment is CPython 3.13.12 on Windows. Direct versions
are pinned in requirements.txt; the run's environment-freeze.txt captures all
installed packages. Fresh dependency installation is not independently verified.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m powerco.workflow --output runs/my_new_run --jobs 3
.\.venv\Scripts\python.exe -m powerco.validate runs/my_new_run --save
```

Both supplied CSVs must be at the project root. Paths are resolved from the
package rather than the shell's current directory. Each run requires a new output
directory and refuses overwrite. Seeds and CV indices are fixed; elapsed times
and timestamps vary. Hardware/library versions can affect floating point results.
Run time depends on machine capacity; 12 RF search candidates require 60 fits.

Do not import `main.py`: it executes training and repeated test evaluations at
module scope. It remains the student's original historical implementation.

## Design and evidence

The legacy 80/20 seed-42 stratified split is reconstructed. All new model selection
and threshold search use the training partition, with five fold-local pipelines.
Model families are Logistic Regression, Random Forest and histogram gradient
boosting, plus a dummy control, balanced-HGB ablation and simpler-RF ablation.
The fixed RF search and threshold protocol are in audit/PROTOCOL.md.

The selected conditional research feature set includes sales channel, original
campaign, gas-service indicator, previous annual electricity/gas consumption,
last-month electricity consumption, active products, supplied tenure and subscribed
power. A recent/annual electricity share is engineered without assuming a date.
Prices, raw dates, forecasts and financial fields are excluded from modelling
pending their timestamp provenance. This is a precaution, not evidence that each
excluded feature leaks. Even retained fields require verified snapshot timing.

The run folder contains CSV evidence, 10 figures, fitted pipeline, source snapshot,
data/source checksums, complete RF search candidates, fold metrics, OOF and
provisional test predictions, feature eligibility, segment diagnostics, fixed-fold
permutation importance and observed-example local sensitivity. Source code is the
authoritative description of transformations and exact model settings.

## Limitations and manual work

- Legacy test scores have already been seen. New test figures and metrics are
  provisional. A new random split does not make previously exposed customers new.
- Search/model/OOF threshold selection are not nested; reported CV and selected
  OOF scores are optimistically selected. No confidence interval for generalisation
  is claimed. Use genuinely new, temporally verified customers for confirmation.
- Pricing relationship investigation remains EDA-only. Confirm a cutoff before
  assessing incremental predictive value; do not permanently discard pricing.
- No verified contact budget, error costs, treatment effects or monetary savings.
  F1 is a provisional research objective; 0.5 is a comparison rule, not a business
  optimum. Scores from class-weighted models need calibration validation.
- Segment checks are not demographic fairness certification. No protected-group
  attributes are supplied. Hashed IDs are pseudonymous, not guaranteed anonymous.
- No historical Git repository exists. git-log.txt truthfully records this; no
  historical commits have been invented. Actual future commits can be recorded.
- Student must check the AI permission applicable to this module, read sources,
  review every interpretation, add personal reflection and a truthful declaration.
  The brief says approximately 2400 report words and a 3000-word maximum;
  confirm what tables, references and appendices count toward the limit.

## Report and packaging

`tools/build_report.py` uses the Python standard library and ReportLab to turn a
validated run into the separate working-draft PDF and editable Markdown. Install
`requirements-report.txt` in the report environment, then:

```powershell
python tools/build_report.py --run runs/my_new_run --output output/report_review
```

The build refuses an existing output directory. Reporting does not retrain models
or select thresholds. `tools/package_artefact.py` validates the run and creates a
software ZIP separately from the PDF:

```powershell
.\.venv\Scripts\python.exe tools/package_artefact.py --run runs/my_new_run --output output/my_software.zip
```

Create the output parent directory first. The ZIP excludes the report PDF,
virtual environments and caches.
CSV data are included for reproduction within the assessment; share only through
the authorised university submission channel. Load joblib files only from trusted
sources and with the recorded environment.
