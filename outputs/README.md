# Public Outputs

This directory contains small, public-safe reporting artifacts derived from
committed result tables. It must not contain raw point clouds, predictions,
checkpoints, containers, logs or machine-specific paths.

## FOR-instance Method Tracker

[`sat_treex_benchmark_metrics/for_instance_method_benchmark_tracker.xlsx`](sat_treex_benchmark_metrics/for_instance_method_benchmark_tracker.xlsx)
summarises the completed SegmentAnyTree and TreeX results. Its canonical
sources are:

- `methods/segmentanytree/examples/sat_final_test_aligned_summary_*.csv`;
- `methods/segmentanytree/examples/sat_final_test_aligned_provenance_*.json`;
- `methods/treex/examples/treex_split_summary.csv`; and
- `methods/treex/examples/treex_site_summary.csv`.

The workbook distinguishes mean plot metrics from count-aggregated micro
metrics. SegmentAnyTree site-level rows are labelled as a pre-final diagnostic
snapshot because the final per-plot table has not been transferred locally.
