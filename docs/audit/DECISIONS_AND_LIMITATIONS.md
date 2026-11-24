# Decision and limitation register

| ID | Decision or open issue | Reason and evidence | Status / owner |
|---|---|---|---|
| D01 | Predict churn=1 within supplied following-three-month label | Brief pp.4-6; supervised binary classification | DONE, conditional on snapshot |
| D02 | Reuse legacy stratified split, seed 42 | Preserve comparison provenance, never pretend the inspected test is fresh | DONE |
| D03 | Pricing EDA only | Dates are 2015, but no label observation/prediction date supplied | BLOCKED modelling, convenor/data owner |
| D04 | Raw contract dates excluded | Future registered end dates may be known but actual point-in-time versions are absent | DONE conservative exclusion |
| D05 | Forecast and financial fields excluded from primary modelling | Forecast generation and margin accounting timestamps unverified | DONE, reassess after clarification |
| D06 | Supplied historical/current-contract fields conditionally retained | Intended snapshot semantics are more plausible, still unverified | PARTIAL temporal validity |
| D07 | Literal MISSING kept as category | Prevent silent conversion to an arbitrary valid channel; missingness can be operationally informative | DONE; investigate process bias |
| D08 | Numeric median imputation inside folds | Undefined zero-denominator share requires a stable training-only fallback | DONE |
| D09 | Retain outliers, log/scale LR | Extreme usage may represent valid large SMEs; IQR flags are diagnostics, not deletion rules | DONE |
| D10 | Balanced LR/RF and explicit HGB weighting comparison | Class imbalance; avoid synthetic samples crossing folds | DONE when executed evidence exists |
| D11 | F1 selection without monetary optimum | Error costs, budget and treatment response unknown | DONE provisional research; business context BLOCKED |
| D12 | Five-fold search/selection not nested | Finite scope; selection bias is documented rather than concealed | PARTIAL generalisation evidence |
| D13 | Permutation + local sensitivity rather than SHAP | Model-agnostic diagnostics; raw feature permutation recomputes the derived share | DONE when executed; no causal claim |
| D14 | Service/channel/tenure segment diagnostics | Relevant observed groups, but no protected attributes or demographic conclusions | PARTIAL fairness assurance |
| D15 | No fabricated git history | No .git existed during audit | BLOCKED historical provenance; student locates any genuine prior logs |
| D16 | AI-assisted draft only | Student states superseding permission exists; exact university wording not supplied | Student checks permission and final authorship |
| D17 | Separate PDF and ZIP | Brief p.9 requires separate uploads | DONE after validation/packaging |

## Error meaning and business constraints

A false positive is a retained customer incorrectly flagged. Possible consequences
are contact workload, unnecessary discounts and annoyance. A false negative is a
customer who leaves without being flagged; a retention opportunity may be missed.
Neither relative cost nor the preventable portion of churn can be inferred from
the supplied data. The loss from a false negative is not automatically larger.
Offer effect, profit impact and causal response require a controlled campaign.

## Ethics and monitoring design

Service groups and hashed campaign/channel categories may proxy unequal treatment.
Compare recall and false-positive rates with denominators, flag groups with fewer
than 30 positive examples as unstable, and investigate rather than forcing a
parity rule without business/governance context. Protected-group fairness cannot
be established from absent attributes. Missing-category differences may reflect
data collection processes rather than customer behaviour.

Store pseudonymous IDs and prediction outputs with restricted access, retention
periods and a documented purpose. Do not publish row-level examples. Explain that
scores are conditional model estimates, provide human review, and avoid using
churn scores to deny essential services or impose punitive prices.

Before any pilot: verify cutoff, validate on genuinely new time-stamped customers,
measure calibration, agree costs/budget and authorise data use. During a pilot:
monitor missing/unknown category rates, feature drift, score distribution, contact
volume, complaints and service-group differences. After the three-month label
matures, assess precision/recall/F1, AP, ROC-AUC and calibration. Set alert limits
from an approved baseline; no arbitrary production thresholds are asserted here.
Compare randomised offer/control groups to estimate incremental retention and
net costs. Review model drift and retrain only through versioned validation.
