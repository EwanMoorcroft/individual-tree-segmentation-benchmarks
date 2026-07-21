# Public Outputs

This directory contains small, public-safe reporting artifacts derived from
committed result tables. It must not contain raw point clouds, predictions,
checkpoints, containers, logs or machine-specific paths.

Status date: 21 July 2026.

## FOR-instance Canonical Tables

[`for_instance_benchmark_metrics/for_instance_method_benchmark_results.csv`](for_instance_benchmark_metrics/for_instance_method_benchmark_results.csv)
is the canonical accepted held-out metric table. It contains seven completed
rows for SegmentAnyTree, TreeX, TreeLearn and TLS2trees. Every row uses the
supplied 11-plot test split, 323 reference instances, point-aligned
predictions, IoU `>= 0.5` and maximum-cardinality one-to-one matching. The five
shared-mask rows form one comparison group: two primary results and three
baselines. The development-tuned and published-default TLS2trees rows use a
class-3-ignore domain and form a separate within-method comparison; they must
not be included in the shared ranking.

[`for_instance_benchmark_metrics/benchmark_result_registry.csv`](for_instance_benchmark_metrics/benchmark_result_registry.csv)
is the canonical governance layer. It preserves every accepted, diagnostic,
historical, rejected, operational and candidate identity while adding
controlled result/completion roles, ranking eligibility/exclusion, learning
regime, dataset exposure, prediction material, scoring mask and leaf
attachment. Human-readable descriptions and historical IDs remain intact.

The held-out metric table's canonical evidence sources include:

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
- `methods/tls2trees/examples/tls2trees_development_tuned_leaf_off_test_plot_diagnostic.csv`;
- `methods/tls2trees/examples/tls2trees_development_tuned_leaf_off_test_site_diagnostic.csv`;
- `methods/tls2trees/examples/tls2trees_development_tuned_leaf_off_test_diagnostic.csv`;
- `methods/tls2trees/examples/tls2trees_published_default_test_plot_results.csv`;
- `methods/tls2trees/examples/tls2trees_published_default_test_site_results.csv`;
- `methods/tls2trees/examples/tls2trees_published_default_test_results.csv`;
- `methods/tls2trees/examples/tls2trees_published_default_test_provenance.json`;
- `methods/tls2trees/examples/tls2trees_published_default_prediction_retention_manifest.json`;
- `methods/tls2trees/examples/tls2trees_published_default_leaf_off_test_plot_diagnostic.csv`;
- `methods/tls2trees/examples/tls2trees_published_default_leaf_off_test_site_diagnostic.csv`; and
- `methods/tls2trees/examples/tls2trees_published_default_leaf_off_test_diagnostic.csv`.

TreeX has no fine-tuning stage. The clean TreeLearn published-pretrained row
used a documented execution-only zero-cluster recovery for one plot; it
changed no model weights or settings and is recorded in the corresponding
provenance file.

[`for_instance_benchmark_metrics/for_instance_method_development_diagnostics.csv`](for_instance_benchmark_metrics/for_instance_method_development_diagnostics.csv)
preserves the TreeX reference-labelled-mask diagnostic, three TreeLearn
development-only diagnostics and both TLS2trees leaf-off target diagnostics.
The TLS2trees development leaf-attachment screen is published separately as
per-plot, candidate and provenance evidence under
`methods/tls2trees/examples/` in
`tls2trees_development_leaf_screen_plot_results.csv`,
`tls2trees_development_leaf_screen_candidate_results.csv` and
`tls2trees_development_leaf_screen_provenance.json`.

[`for_instance_benchmark_metrics/for_instance_prediction_retention_registry.csv`](for_instance_benchmark_metrics/for_instance_prediction_retention_registry.csv)
records the off-Git prediction sets required for future metrics. A completed
accuracy row is not publication-ready unless this registry says its prediction
set remains retained. Accepted held-out rows include the exact prediction and
metric roots, retention-manifest path and hash, and public evidence path. Each
completed TLS2trees variant has one retained prediction set and one neutral
public evidence path.
The `variant` column uses the same canonical method-variant slugs as the
accepted held-out metric table; `retention_profile` distinguishes development
diagnostics, checkpoint sweeps and completed held-out prediction sets.

The governance support tables are:

- [`test_exposure_ledger.csv`](for_instance_benchmark_metrics/test_exposure_ledger.csv),
  which distinguishes a test job executing, metrics being viewed, predictions
  being visualised and any later configuration change;
- [`method_development_budget.csv`](for_instance_benchmark_metrics/method_development_budget.csv),
  which records evidence-backed tuning/training counts and uses `unknown` for
  unavailable compute time;
- [`method_environment_provenance.csv`](for_instance_benchmark_metrics/method_environment_provenance.csv),
  which separates method-specific upstream, container/environment and
  checkpoint evidence; and
- [`diagnostic_metric_availability.csv`](for_instance_benchmark_metrics/diagnostic_metric_availability.csv),
  which states whether multiple-IoU, bootstrap and error-decomposition
  diagnostics are supported, unavailable or not recorded for each result.

Build derived site, plot-distribution and bootstrap-CI diagnostic summaries
with:

```bash
python scripts/reporting/build_for_instance_governance_outputs.py
```

Use the same command with `--check` to fail when any tracked generated CSV is
missing or stale.

The generated files are:

- [`for_instance_method_site_results.csv`](for_instance_benchmark_metrics/for_instance_method_site_results.csv),
  a consolidated 35-row site table for the seven accepted held-out rows;
- [`for_instance_plot_distribution_diagnostics.csv`](for_instance_benchmark_metrics/for_instance_plot_distribution_diagnostics.csv),
  which reports median and interquartile plot F1, zero-F1 plot counts and a
  five-site macro summary separately from the canonical point estimates; and
- [`for_instance_bootstrap_confidence_intervals.csv`](for_instance_benchmark_metrics/for_instance_bootstrap_confidence_intervals.csv),
  which reports ordinary and site-stratified plot-bootstrap intervals with
  seed `20260721`, `10000` iterations and explicit diagnostic/selection flags.

Missing evidence is never inferred from a result value or a resource request.
The command reads the canonical evidence-led tables; it does not invent or
rewrite exposure, budget or environment facts.

## Generated Workbook

[`for_instance_benchmark_metrics/for_instance_method_benchmark_tracker.xlsx`](for_instance_benchmark_metrics/for_instance_method_benchmark_tracker.xlsx)
is a generated review artefact, not an independent data source. Build it from
the canonical CSV tables with:

```bash
python scripts/reporting/build_for_instance_workbook.py
```

Use `python scripts/reporting/build_for_instance_workbook.py --check` for a
byte-exact stale-artifact gate.

The builder is deterministic and does not require Microsoft Excel or
LibreOffice. Automated tests reconcile identities, table ranges and aggregates
against the CSV sources. Separate sheets show the five-row ranked leaderboard,
the two differently scoped TLS2trees rows, the legacy-compatible all-held-out
view, result governance, test exposure, development budgets, environment
provenance and diagnostic availability. Protocol and ranking fields remain
visible so different masks are not silently combined; visual layout and
public-safety review remain required before release.
