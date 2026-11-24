# Initial audit - 17 November 2026

Source: all 14 pages of cps7008-assessment-1.pdf, original main.py,
AI Guidance.docx, cover sheet, Data Description.docx and both actual CSVs.
DONE means the stated evidence exists; code presence is not execution evidence.

| Requirement | Initial status | Evidence / missing work |
|---|---|---|
| Stage 1: supervised binary task, target | PARTIAL | main.py uses churn; narrative and prediction time absent |
| Stage 1: feature justification, constraints and errors | PARTIAL | dates/prices excluded; other timestamp risks unexplained |
| Stage 2: structure and quality | PARTIAL | console checks; literal MISSING not counted |
| Stage 2: imbalance, distributions, outliers, correlations | PARTIAL | target print and one PNG only; no verified saved provenance |
| Stage 2: pricing/customer relationships | PARTIAL | aggregate merge implemented; no relationship outputs |
| Stage 2: leakage and limitations | PARTIAL | cutoff caveat present, no feature-by-feature eligibility record |
| Stage 3: missing, duplicates and outliers | PARTIAL | asserts no NaNs; no robust missing/outlier handling |
| Stage 3: encoding and scaling | PARTIAL | OHE present; LR scales; no saved before/after evidence |
| Stage 3: justified merge and engineering | PARTIAL | one-to-one exploratory mean-price merge only |
| Stage 3: imbalance and fold-local preprocessing | PARTIAL | balanced LR/RF and fold-local CV pipelines present; HGB unweighted |
| Stage 4: LR plus two alternatives | PARTIAL | all implementations found; no persisted executed metrics |
| Stage 4: cross-validation and optimisation | PARTIAL | five folds, 12 RF candidates; console-only results |
| Stage 4: overfitting control | PARTIAL | train/CV and simpler RF comparisons implemented |
| Stage 5: required metrics and confusion matrices | PARTIAL | console evaluation; test already inspected |
| Stage 5: misclassified examples and segments | BLOCKED | not implemented |
| Stage 6: feature/prediction explanations | BLOCKED | not implemented |
| Stage 6: fairness, privacy, causation | BLOCKED | no documented evaluation |
| Stage 7: justified model and business recommendations | BLOCKED | no saved evidence or narrative |
| Reproducible artefact and git log | PARTIAL | no README, lock, persisted model, log or Git repository |
| Structured report and separate PDF | BLOCKED | absent |
| Cover reflection and authorship | BLOCKED | student action required |

Original main.py SHA256:
fe7978ac3a73bb82c4f496910f19fe566205b89ea6141a8c88a4cdc3a3c5a0fb

The script has no main guard. Importing/running it trains models and evaluates
the legacy test set multiple times. Do not import it as a library.
Earlier user-supplied scores are recorded separately, not treated as verified
current results. No earlier results log or fitted models were present.

## Confirmed structural checks

Client shape 14606 x 26; prices 193002 x 8. No full-row duplicates,
duplicate client IDs, duplicate price (id, date) keys or actual NaNs found.
There are 3725 channel_sales and 64 origin_up literal MISSING tokens.
All client IDs have prices; 1490 price IDs have no client record.
Prices span 2015-01-01 through 2015-12-01; date_modif_prod includes
2016-01-29. These ranges do not establish a prediction cutoff.
Data Description mentions activity_new, which is absent, and describes meter
rental as 2 months despite its 12m column name; do not assume the units.

## Clarifications supplied by student in this chat

Student states the previously supplied AI guidance was superseded and instructs
AI-assisted implementation/drafting. Written module-specific permission was not
provided. Final student review and truthful declaration remain required.
Student confirms no verified cutoff, contact budget or error costs; agrees to
EDA-only prices, provisional F1 threshold and unknown financial benefits.

## Required versus optional

Required: seven stages, Python artefact, reproducibility, actual experiment
evidence, critical report, separate PDF, genuine available Git log.
Optional: SHAP specifically (other justified explanations are acceptable),
deployment, dashboards, monetary simulations and additional algorithms.
These optional enhancements will not be presented as assessment requirements.
