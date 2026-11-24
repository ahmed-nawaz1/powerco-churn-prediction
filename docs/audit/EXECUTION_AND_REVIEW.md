# Execution and review record

Actual modelling run: `runs/validated_20261117`, Python 3.13.12 and the
recorded existing environment; no environment installation was required.
All seven candidate configurations completed five-fold evaluation, the fixed
12-candidate RF search completed, and the selected model was frozen before the
single new provisional test-evaluation phase. The earlier exposed test remains
provisional. No further optimisation was performed after reading these outputs.

`powerco.validate` passed 44 checksummed result/source files, actual input hashes,
14,606 split assignments, seven models, the twelve search candidates and the
selected test metrics recomputed from saved predictions. Reloaded pipeline scores
matched saved test scores within numerical precision. No missing/infinite encoded
values remained. All ten figures were visually inspected.

Three targeted unit checks passed: excluded-field invariance, undefined-ratio
handling and fold-isolated unseen-category behaviour. This verifies specific
failure modes, not complete future-data validity or economic value.

The dummy control generated expected undefined-precision warnings because it
predicts no positive examples; precision is reported as zero. This is a known
metric convention, not a failed model fit.

PDF review used the bundled ReportLab 4.4.9 environment and Poppler renderer.
All seven pages of output/report_final/powerco_report_WORKING_DRAFT.pdf were
rendered and visually inspected. Orphan headings and a separated table caption
were corrected before the final render. No clipped tables, overlaps or unreadable
main text were observed. Body prose is 2,335 words; all PDF-extractable text is
2,981 words, including references/tables/headers/footers. Text inside rasterised
charts is not included in that extraction count. The university's counting rules
still require confirmation. Earlier render versions are retained as local review
iterations and are not packaged as the final report.

Poppler emitted missing-display-font notices for Symbol and ArialUnicode while
rendering. The actual rendered output uses legible Helvetica and chart text;
all final pages were inspected rather than treating the notices as proof of a
visual defect or suppressing the check.

Web sources actually opened and checked on 17 November 2026:

- https://jmlr.org/papers/v12/pedregosa11a.html
- https://scikit-learn.org/stable/common_pitfalls.html
- https://scikit-learn.org/stable/modules/classification_threshold.html
- https://scikit-learn.org/stable/modules/permutation_importance.html

Fresh package installation, prospective validation, verified point-in-time data,
personal reflection and student final authorship have not been represented as
completed checks. Archive validation is recorded separately alongside the ZIP.
