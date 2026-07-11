# SegmentAnyTree public-safe examples

This directory contains no point coordinates, model files or private paths.

Files beginning with `provisional_released_checkpoint_` record the earlier
32-plot released-checkpoint diagnostic. The final export did not preserve
point correspondence, so the coordinate-rematched metrics and workbook are not
aligned accuracy results. They are retained to make the failed route
traceable.

`pilot_metrics.csv` and `pilot_status.json` retain the earlier development
pilot investigation. Those values are also provisional.

Files ending in `_example` contain fabricated values that illustrate table
schemas. They are not benchmark results.

The completed historical full-training SAT checkpoint is
`sat_for_quicktune_to49_20260706_140730`. Its canonical public evidence is:

- `sat_final_test_aligned_summary_*.csv`, the retained 11-plot aggregate; and
- `sat_final_test_aligned_provenance_*.json`, the result identity, arithmetic,
  evidence locations and retained-provenance gaps.

- `sat_plot_failure_modes_*.csv` contains a pre-final per-plot diagnostic
  snapshot for the same checkpoint, but not the final evaluation ID.
- `sat_site_failure_modes_*.csv` aggregates that diagnostic snapshot by split
  and site.
- `sat_unmatched_prediction_audit_*.csv` and
  `sat_unmatched_reference_audit_*.csv` explain false-positive and
  false-negative failure modes without exposing point coordinates.
- `sat_training_vs_validation_domain_audit_*.csv` records public-safe split and
  domain-distribution counts.
- `sat_validation_postprocess_*` records a validation-only post-processing
  sweep. It is an ablation, not the headline held-out test result.

The historical unfiltered aligned held-out result has mean plot F1 `0.4825` and
micro F1 `0.4692` (TP=202, FP=336, FN=121). The older diagnostic snapshot has
mean plot F1 `0.4798`; it is retained for failure-mode interpretation and must
not be substituted for the final aggregate. The validation-selected size
filter improved validation slightly but did not replace that historical result.

`for_instance_result_registry.csv` is the status authority for current and
historical result roles. The aligned pretrained and replacement fine-tuned
target rows are complete. Their authoritative public aggregates are in
`sat_completed_target_results_20260711.csv`; the matching provenance record is
`sat_completed_target_provenance_20260711.json`. The exact CULS, NIBIO, RMIT,
SCION and TUWIEN breakdown for both targets is in
`sat_completed_target_site_results_20260711.csv`.
