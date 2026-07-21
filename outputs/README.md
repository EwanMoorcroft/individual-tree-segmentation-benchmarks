# Public Outputs

This directory contains small, public-safe reporting artifacts derived from
committed result tables. It must not contain raw point clouds, predictions,
checkpoints, containers, logs or machine-specific paths.

## FOR-instance Method Tracker

[`for_instance_benchmark_metrics/for_instance_method_benchmark_results.csv`](for_instance_benchmark_metrics/for_instance_method_benchmark_results.csv)
is the canonical headline table. It contains completed held-out-test rows for
SegmentAnyTree, TreeX, TreeLearn and TLS2trees. Every row uses the supplied
11-plot test split, 323 reference instances, point-aligned predictions, IoU
`>= 0.5` and maximum-cardinality one-to-one matching. Protocol and evaluation
mask columns make the TLS2trees class-3-ignore scoring domain explicit. Its
canonical sources include:

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
- `methods/treelearn/examples/treelearn_finetuned_test_site_results_20260713.csv`;
- `methods/treelearn/examples/treelearn_finetuned_test_provenance_20260713.json`;
- `methods/tls2trees/examples/tls2trees_development_tuned_test_plot_results.csv`;
- `methods/tls2trees/examples/tls2trees_development_tuned_test_site_results.csv`;
- `methods/tls2trees/examples/tls2trees_development_tuned_test_results.csv`;
- `methods/tls2trees/examples/tls2trees_development_tuned_test_provenance.json`;
- `methods/tls2trees/examples/tls2trees_development_tuned_prediction_retention_manifest.json`;
- `methods/tls2trees/examples/tls2trees_published_default_test_plot_results.csv`;
- `methods/tls2trees/examples/tls2trees_published_default_test_site_results.csv`;
- `methods/tls2trees/examples/tls2trees_published_default_test_results.csv`;
- `methods/tls2trees/examples/tls2trees_published_default_test_provenance.json`;
- `methods/tls2trees/examples/tls2trees_published_default_prediction_retention_manifest.json`;
- `methods/tls2trees/examples/tls2trees_published_default_leaf_off_test_plot_diagnostic.csv`;
- `methods/tls2trees/examples/tls2trees_published_default_leaf_off_test_site_diagnostic.csv`; and
- `methods/tls2trees/examples/tls2trees_published_default_leaf_off_test_diagnostic.csv`.

The
[`for_instance_benchmark_metrics/for_instance_method_benchmark_tracker.xlsx`](for_instance_benchmark_metrics/for_instance_method_benchmark_tracker.xlsx)
mirrors the canonical tables. It keeps protocol groups visible so TLS2trees is
not silently ranked against methods using a different evaluation mask. TreeX
has no fine-tuning stage. The clean TreeLearn published-pretrained row used a
documented execution-only zero-cluster recovery for one plot; it changed no
model weights or settings and is recorded in the corresponding provenance
file.

[`for_instance_benchmark_metrics/for_instance_method_development_diagnostics.csv`](for_instance_benchmark_metrics/for_instance_method_development_diagnostics.csv)
preserves the three TreeLearn development-only diagnostics and the TLS2trees
leaf-off target diagnostic. The TLS2trees development leaf-attachment screen
is published separately as per-plot, candidate and provenance evidence under
`methods/tls2trees/examples/` in
`tls2trees_development_leaf_screen_plot_results.csv`,
`tls2trees_development_leaf_screen_candidate_results.csv` and
`tls2trees_development_leaf_screen_provenance.json`.

[`for_instance_benchmark_metrics/for_instance_prediction_retention_registry.csv`](for_instance_benchmark_metrics/for_instance_prediction_retention_registry.csv)
records the off-Git prediction sets required for future metrics. A completed
accuracy row is not publication-ready unless this registry says its prediction
set remains retained. Headline rows include the exact prediction and metric
roots, retention-manifest path and hash, and public evidence path. Each
completed TLS2trees variant has one retained prediction set and one neutral
public evidence path.
The `variant` column uses the same canonical method-variant slugs as the
headline table; `retention_profile` distinguishes development diagnostics,
checkpoint sweeps and completed held-out prediction sets.
