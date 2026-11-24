# Locked research protocol

This protocol is written before executing the new workflow. It does not remove
contamination from previously inspected test results or full-data exploration.

1. Preserve original main.py and data; hash inputs and archive original code.
2. Reconstruct the legacy stratified 80/20 seed-42 split by original row order.
   Store indices, customer IDs, hashes and five stratified training folds.
3. Primary eligible fields: three supplied categorical fields and six supplied
   historical/contract fields listed in powerco/features.py. Treat snapshot
   availability as an unresolved assumption, not verified temporal safety.
   Exclude raw dates, all 2015 prices, forecasts and financial/margin fields
   from primary models because their as-of availability cannot be established.
   Forecasts might have existed prospectively; exclusion is precautionary,
   not proof of leakage. Reassess all eligibility after clarification.
4. Engineer a recent/annual electricity share using positive annual denominator;
   undefined ratios become missing. Retain MISSING as explicit categorical state.
   Median imputation and unknown-safe OHE are fitted within each training fold.
   LR uses signed-log numerical values and scaling; trees retain numerical scale.
   Do not delete valid extreme customers based only on their magnitude.
5. Fit dummy baseline, balanced LR, balanced RF, original unweighted HGB,
   balanced HGB, tuned RF and simpler RF on the same five folds.
   RF search: the original 12 random candidates, seed 42, F1 objective.
   This is not nested CV: tuning and model comparison estimates are optimistic.
6. Select the candidate by mean training-fold validation F1 at threshold 0.5
   (tie: simpler model by declaration order). Select an F1 threshold from its
   training OOF probabilities on a fixed 0.05..0.95 grid at step 0.01.
   This pooled OOF threshold estimate is also selection-biased; it is a research
   operating point, not a cost-optimal business decision or fresh validation.
7. Freeze selected name, parameters and threshold to selection.json BEFORE
   reading any new test prediction. Evaluate all locked candidates at 0.5 and
   the selected model at its frozen threshold in one final test phase.
   Never tune from these test outputs. Legacy test results remain provisional.
8. Segment/error analysis and local/global explanations are diagnostics, not
   further model selection. Use OOF predictions for segments and untouched
   fold 1 for permutation importance and local perturbations. Diagnostics cannot
   establish protected-group fairness or causal effects. Permutations may be
   unrealistic for correlated features.
9. Persist raw metrics, fold results, search configuration, predictions, figures,
   source/data hashes, dependencies and fitted selected pipeline. Validate saved
   predictions against reload and audit every brief requirement.
10. Produce a clearly labelled report working draft and separate software ZIP.
    No invented savings, references, historical commits, reflections or approval.
    Student supplies final academic judgement, reflection and declaration.
