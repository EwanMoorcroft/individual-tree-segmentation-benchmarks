# Public Outputs

This directory contains small, public-safe reporting artifacts derived from
committed result tables. It must not contain raw point clouds, predictions,
checkpoints, containers, logs or machine-specific paths.

## FOR-instance Method Tracker

[`sat_treex_benchmark_metrics/for_instance_method_benchmark_tracker.xlsx`](sat_treex_benchmark_metrics/for_instance_method_benchmark_tracker.xlsx)
summarises the completed SegmentAnyTree, TreeX and TreeLearn results. Its canonical
sources are:

- `methods/segmentanytree/examples/sat_final_test_aligned_summary_*.csv`;
- `methods/segmentanytree/examples/sat_final_test_aligned_provenance_*.json`;
- `methods/segmentanytree/examples/sat_completed_target_results_20260711.csv`;
- `methods/segmentanytree/examples/sat_completed_target_site_results_20260711.csv`;
- `methods/segmentanytree/examples/sat_completed_target_provenance_20260711.json`;
- `methods/treex/examples/treex_split_summary.csv`; and
- `methods/treex/examples/treex_site_summary.csv`;
- `methods/treelearn/examples/treelearn_completed_development_results_20260712.csv`;
- `methods/treelearn/examples/treelearn_completed_development_site_results_20260712.csv`; and
- `methods/treelearn/examples/treelearn_finetune_validation_results_20260712.csv`;
- `methods/treelearn/examples/treelearn_finetuned_test_results_20260713.csv`;
- `methods/treelearn/examples/treelearn_finetuned_test_site_results_20260713.csv`; and
- `methods/treelearn/examples/treelearn_finetuned_test_provenance_20260713.json`.

The workbook distinguishes mean plot metrics from count-aggregated micro
metrics. It includes completed SegmentAnyTree, TreeX and leakage-controlled
TreeLearn fine-tuned test rows for CULS, NIBIO, RMIT, SCION and TUWIEN. The
overlap-affected TreeLearn published-checkpoint development diagnostic remains
in a separate comparable group.

[`sat_treex_benchmark_metrics/for_instance_prediction_retention_registry.csv`](sat_treex_benchmark_metrics/for_instance_prediction_retention_registry.csv)
records the off-Git prediction sets required for future metrics. A completed
accuracy row is not publication-ready unless this registry says its prediction
set remains retained.
