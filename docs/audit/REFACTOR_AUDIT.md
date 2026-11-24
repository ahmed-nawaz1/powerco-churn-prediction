# Structure and entry-point refactor - 17 November 2026

Scope: reorganise the existing project, add one clear command, show training and
failure status, and preserve assessment evidence. This is an engineering refactor,
not a new attempt to select better models using the exposed test set.

## Preservation and location changes

| Previous location | Current location |
|---|---|
| original main.py | legacy/main.py (unchanged SHA256) |
| first AI-assisted package/tools/tests/README | legacy/previous_pipeline/ |
| client_data.csv, price_data.csv | data/raw/ (unchanged bytes) |
| assessment PDF and three DOCX files | docs/assessment/ |
| audit/ | docs/audit/ |
| runs/ | artifacts/runs/ |
| earlier output/ | artifacts/archive/output_v1/ |
| earlier tmp/ | artifacts/archive/qa_v1/ |

No original data, source, report, experiment results or Git evidence were deleted.
Historical audit/report paths are retained as contemporaneous records. The current
README and docs/ARCHITECTURE.md document the new paths. Old manifests/checksums
have not been rewritten to pretend the historical run used refactored code.

## Delivered engineering changes

- Root main.py is the official, import-safe entry point; run.ps1 selects .venv.
- Data, features, preprocessing, models, evaluation, explanation, plots, validation,
  inference and CLI responsibilities have separate modules.
- Seven visible stages, per-model training/CV messages, persisted log and atomic
  run state. Failed/interrupted runs keep evidence and never replace latest success.
- One-command full run, a separate no-training status check and saved-model CSV
  prediction. Every new run/output refuses accidental overwrite.
- Machine-readable summary and a human-readable SUMMARY.txt accompany each new run.
- Runtime integrity checks use explicit exceptions rather than removable asserts.
- Schema-1 relocated inputs are mapped explicitly; executed source snapshots are
  validated and identified as archived whenever current source differs.
- Fixed seed, folds, feature policy, RF candidates, model/threshold selection and
  provisional-test interpretation remain as declared in PROTOCOL.md.
- Ruff 0.17.3 formatting/import/static checks apply to active source; legacy copies
  and generated artefacts are excluded. Existing runtime dependencies were retained.
- Eight focused tests cover leakage-sensitive feature handling, import safety,
  default CLI dispatch, invalid workers, honest failure state and output tampering.

## Evidence locations

New timestamped runs include run.log, status.json, summary.json, SUMMARY.txt,
validation.json, source snapshots and the full prior experiment evidence schema.
Use the latest successful-run pointer in artifacts/latest_run.json.
Full refactor verification results are in REFACTOR_VERIFICATION.json after execution.
Software archive integrity and extracted-directory smoke checks are recorded next
to the current ZIP under artifacts/submissions/.

## Assessment qualifications preserved

The cutoff and point-in-time availability remain unresolved, so pricing is EDA-only.
Test results are provisional; non-nested selection remains optimistically selected.
Historical Git commits cannot be fabricated. Report/student reflection/AI
declaration and final academic judgement remain for the student. This refactor
does not change these qualifications or claim production readiness.
