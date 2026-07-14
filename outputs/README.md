# Public Outputs

This directory contains small, public-safe reporting artifacts derived from
committed result tables. It must not contain raw point clouds, predictions,
checkpoints, containers, logs or machine-specific paths.

## FOR-instance Method Tracker

[`sat_treex_benchmark_metrics/for_instance_method_benchmark_tracker.xlsx`](sat_treex_benchmark_metrics/for_instance_method_benchmark_tracker.xlsx)
contains only the completed primary held-out-test results for SegmentAnyTree,
TreeX and TreeLearn. Every result row uses the supplied 11-plot test split, 323
reference instances and the same point-aligned evaluator. Its canonical
sources are:

- `methods/segmentanytree/examples/sat_final_test_aligned_summary_*.csv`;
- `methods/segmentanytree/examples/sat_final_test_aligned_provenance_*.json`;
- `methods/segmentanytree/examples/sat_completed_target_results_20260711.csv`;
- `methods/segmentanytree/examples/sat_completed_target_plot_results_20260711.csv`;
- `methods/segmentanytree/examples/sat_completed_target_site_results_20260711.csv`;
- `methods/segmentanytree/examples/sat_completed_target_provenance_20260711.json`;
- `methods/treex/examples/treex_combined_dev_test_summary.csv`;
- `methods/treex/examples/treex_prediction_retention_manifest.json`;
- `methods/treex/examples/treex_split_summary.csv`;
- `methods/treex/examples/treex_site_summary.csv`;
- `methods/treelearn/examples/treelearn_pretrained_test_plot_results_20260714.csv`;
- `methods/treelearn/examples/treelearn_pretrained_test_results_20260714.csv`;
- `methods/treelearn/examples/treelearn_pretrained_test_site_results_20260714.csv`;
- `methods/treelearn/examples/treelearn_pretrained_test_provenance_20260714.json`;
- `methods/treelearn/examples/treelearn_finetuned_test_plot_results_20260713.csv`;
- `methods/treelearn/examples/treelearn_finetuned_test_results_20260713.csv`;
- `methods/treelearn/examples/treelearn_finetuned_test_site_results_20260713.csv`; and
- `methods/treelearn/examples/treelearn_finetuned_test_provenance_20260713.json`.

The workbook distinguishes mean plot metrics from count-aggregated micro
metrics. It includes two completed SegmentAnyTree rows, the unsupervised
parameterised TreeX row and two completed TreeLearn rows for CULS, NIBIO,
RMIT, SCION and TUWIEN. TreeX has no fine-tuning stage. The clean TreeLearn
published-pretrained row used a documented execution-only zero-cluster
recovery for one plot; it changed no model weights or settings and is recorded
in the corresponding provenance file.

[`sat_treex_benchmark_metrics/for_instance_method_development_diagnostics.csv`](sat_treex_benchmark_metrics/for_instance_method_development_diagnostics.csv)
preserves the three TreeLearn development-only diagnostics that were formerly
mixed into the headline table. Their 21-plot and 5-plot scopes are not directly
comparable with the primary 11-plot test results.

[`sat_treex_benchmark_metrics/for_instance_prediction_retention_registry.csv`](sat_treex_benchmark_metrics/for_instance_prediction_retention_registry.csv)
records the off-Git prediction sets required for future metrics. A completed
accuracy row is not publication-ready unless this registry says its prediction
set remains retained. Headline rows include the exact prediction and metric
roots, retention-manifest path and hash, and public evidence path.
