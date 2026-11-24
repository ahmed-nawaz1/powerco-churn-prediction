# Final requirement-by-requirement audit

17 November 2026. Current executed run: `artifacts/runs/20261117_234639_854637`.
The pre-refactor audit is preserved in COMPLIANCE_AUDIT_before_refactor.md.
Refactor equivalence and one-command checks are in REFACTOR_VERIFICATION.json.
DONE = implemented, executed and checked for this conditional research experiment.
PARTIAL = evidence exists but a material clarification or student review remains.
BLOCKED = missing external evidence cannot truthfully be manufactured.
This is an implementation audit, not an assurance of a particular university mark.

| Brief requirement (pp.6-9) | Status | Traceable evidence and remaining qualification |
|---|---|---|
| 1: Supervised binary task | DONE | Report section 1; churn 1/0 and supplied three-month horizon |
| 1: Target and justified inputs | PARTIAL | feature_eligibility.csv; SnapshotFeatures; retained historical snapshot semantics remain unverified |
| 1: Objectives, constraints and error implications | DONE | Report sections 1,5,7; DECISIONS_AND_LIMITATIONS; unknown costs/budget explicitly retained |
| 2: Structure, distributions and data quality | DONE | data_quality.json; training_summary.csv; Figures 1-3 and training histograms |
| 2: Class imbalance and missingness | DONE | 1419/14606 churn; true nulls vs literal MISSING distinguished; Figure 1 |
| 2: Outliers and correlations | DONE | outlier_diagnostics.csv; spearman_correlations.csv; 02_correlations.png; valid extremes retained with explanation |
| 2: Customer/pricing/churn relationships | DONE | training-only distributions, price quartiles and history, one-to-one exploratory pricing merge; no causal claim |
| 2: Leakage and limitations | DONE | PROTOCOL; date ranges; report limitations; financial/forecast/dates/prices excluded from modelling |
| 3: Missing values, duplicates and outliers | DONE | explicit key validation; numeric imputation in folds; MISSING category; outlier diagnostic/retention policy |
| 3: Encode categoricals and scale where appropriate | DONE | pipeline; unseen-category test; LR signed log/StandardScaler; trees unscaled |
| 3: Combine datasets and justify engineered features | DONE | price means/std/min/max for EDA; valid one-to-one customer merge; positive-denominator electricity share |
| 3: Establish incremental modelling value of prices | BLOCKED | cutoff unavailable; pricing remains an active future modelling candidate, not permanently excluded |
| 3: Imbalance and fold-local preprocessing | DONE | balanced LR/RF and HGB ablation; all learned preprocessing inside CV estimators; no cross-fold oversampling |
| 3: Before/after evidence | DONE | preprocessing_checks.json: 9 raw / 10 engineered / 22 encoded; 97 undefined training ratios imputed; finite output |
| 4: Logistic Regression baseline | DONE | fold_metrics.csv and cv_summary.csv; fitted/evaluated pipeline |
| 4: At least two alternatives | DONE | RF and HGB; dummy, simpler RF and balanced HGB comparisons also executed |
| 4: Validation and cross-validation | DONE | disjoint legacy split; split_assignments.csv; five-fold evidence per model; fixed seeds |
| 4: Hyperparameter optimisation | DONE | 12 RF configurations x 5 folds; full search_results.csv and selected parameters |
| 4: Address overfitting | DONE | train/validation metrics; simpler RF comparison; Figure 4; no claim that gap is eliminated |
| Additional requested threshold selection | DONE | training OOF grid; frozen selection.json before new test phase; threshold 0.51 |
| 5: Precision, recall, F1 and ROC-AUC | DONE | CV fold scores and provisional_test_metrics.csv; AP and Brier also saved |
| 5: Confusion matrices and interpretation | DONE | complete TN/FP/FN/TP per model; selected matrix Figure 5; report error workload |
| 5: False positives/negatives | DONE | misclassified_examples.csv with actual source rows; report section 5; no fictional customer explanations |
| 5: Relevant customer segments | DONE | segment_oof_metrics.csv; gas/channel/tenure denominators and small-positive-sample flags; Table 3 |
| 5: Fresh independent test evidence | BLOCKED | legacy test already inspected; all new results explicitly provisional; genuinely new labelled data unavailable |
| 6: Important features and predictions | DONE | fixed-fold raw-input permutation AP decreases, 5 repeats; observed-example local prediction sensitivity |
| 6: Appropriate interpretability method | DONE | permutation + local sensitivity; limitations stated; SHAP itself optional |
| 6: Bias, fairness, privacy and transparency | DONE | actual segment diagnostics; no protected attributes/fairness certification; restricted access and human review discussed |
| 6: Predictive association versus causation | DONE | report sections 2,6,7; sensitivity and price relationships expressly noncausal |
| 7: Select and justify suitable model | DONE | locked training-CV F1 choice; comparison with LR and ranking-objective trade-off explained |
| 7: Findings and associated factors | DONE | observed price-bin rates, customer distributions, permutation rankings, error/segment evidence |
| 7: Practical retention recommendations | DONE | conditional human-reviewed pilot and randomised treatment evaluation; no invented savings |
| 7: Benefits, risks, limitations and monitoring | DONE | report section 7; decision register; delayed outcome metrics and governance |
| Reproducible Python artefact | DONE | original source retained; package/main guard; README; pins; freeze; model; source/input/result checksums |
| Model reload and result consistency | DONE | validation.json PASS; 64 hashed evidence files; current-source model reload; saved metrics recomputed; 8 tests passed |
| Professional structure and one-command execution | DONE | root main.py and run.ps1; modular powerco package; run/status/predict verified; Ruff checks passed |
| Genuine development evidence / git log | PARTIAL | original legacy/main.py, prior workflow and source snapshots preserved; identical refactor predictions checked; git-log.txt records absent repository |
| Historical Git commits | BLOCKED | no .git existed; student must locate authentic prior history elsewhere, if any; no backdated commits created |
| Required report sections with tables/figures | DONE | report working draft: 2335 body words, sections 1-9, 6 figures and 3 tables; source evidence map |
| References | PARTIAL | four primary web sources checked 17 Nov 2026; supplied brief listed; student must read/verify and review APA formatting |
| Separate PDF report and software ZIP | DONE | reports/draft_20261117_refactored PDF; artifacts/submissions/powerco_software_refactored.zip; report PDF excluded from ZIP |
| Personal reflection, AI declaration, authorship | BLOCKED | student-specific statements intentionally unfilled; supplied cover sheet unchanged |
| Final report and submission approval | PARTIAL | working draft requires student critical review; cutoff and word-count scope need clarification |

## Executed results (new conservative conditional feature set)

Tuned RF mean CV F1 = 0.2375558415. Selected threshold = 0.51.
At that threshold, provisional test precision = 0.1651376147, recall =
0.4436619718, F1 = 0.2406876791, ROC-AUC = 0.6548581672 and AP =
0.2035848703. Confusion matrix: TN 2001, FP 637, FN 158, TP 126.
CV train F1 = 0.4239157059; validation F1 = 0.2375558415.
These are saved-code results, not replacements for the unverified earlier
broader-feature scores in legacy_reported_results.json.

## Required manual tasks before submission

1. Confirm the module's current AI permission; keep this chat and accurately
   declare AI-assisted implementation, analysis support and report drafting.
2. Read the draft against every figure/table and rewrite or correct it to
   demonstrate your own academic understanding; provide final authorship review.
3. Complete the original cover sheet: name/ID as applicable, personal reflection,
   AI use tools/purposes/sections/example prompts, and declarations.
4. Ask the convenor/data owner for prediction cutoff, label observation dates and
   point-in-time feature provenance; document continued uncertainty if unanswered.
5. Confirm approximately 2400 words / 3000 maximum and how references, tables,
   captions and appendices are counted. Body prose is 2335 words; the rest is extra.
6. Locate genuine prior Git history if it exists elsewhere. Absence is disclosed;
   an initial commit today would not reconstruct past incremental development.
7. Do not present this legacy test as independent final validation. New prospective
   evidence would require genuinely new customers with verified timestamps.
8. Upload the reviewed report PDF and software ZIP separately via Moodle by the
   deadline shown in the brief. No upload has been performed by this workflow.

## Optional enhancements, not falsely marked as missing mandatory work

Nested CV, SHAP, an interactive dashboard, production deployment, subgroup
confidence intervals and monetary cost simulations. Before additional experiments,
predeclare their purpose and keep the exposed legacy test out of optimisation.
