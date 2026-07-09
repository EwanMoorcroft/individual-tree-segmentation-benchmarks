# SegmentAnyTree public-safe examples

This directory contains no point coordinates, model files or private paths.

Files beginning with `provisional_released_checkpoint_` record the earlier
32-plot released-checkpoint diagnostic. The final export did not preserve
point correspondence, so the coordinate-rematched metrics and workbook are not
accepted accuracy results. They are retained to make the failed route
traceable.

`pilot_metrics.csv` and `pilot_status.json` retain the earlier development
pilot investigation. Those values are also provisional.

Files ending in `_example` contain fabricated values that illustrate table
schemas. They are not benchmark results.

The accepted full-training SAT checkpoint is
`sat_for_quicktune_to49_20260706_140730`.

- `sat_plot_failure_modes_*.csv` contains per-plot validation and held-out
  test metrics for the accepted checkpoint.
- `sat_site_failure_modes_*.csv` aggregates those metrics by split and site.
- `sat_unmatched_prediction_audit_*.csv` and
  `sat_unmatched_reference_audit_*.csv` explain false-positive and
  false-negative failure modes without exposing point coordinates.
- `sat_training_vs_validation_domain_audit_*.csv` records public-safe split and
  domain-distribution counts.
- `sat_validation_postprocess_*` records a validation-only post-processing
  sweep. It is an ablation, not the headline held-out test result.

The headline accepted SAT result is the unfiltered aligned point-wise held-out
test mean F1 `0.4825`. The validation-selected size filter improved validation
slightly but did not replace the accepted unfiltered test score.
