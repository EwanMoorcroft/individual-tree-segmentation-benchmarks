# TreeX Public Results

`treex_combined_dev_test_summary.csv` is the authoritative public result
table. It contains one row for each of the 32 exact-path FOR-instance plots:
21 development plots and 11 held-out test plots.

The remaining CSV files are reproducible views of that table:

- `treex_dev_full_summary.csv` and `treex_test_full_summary.csv` contain the
  source split tables;
- `treex_split_summary.csv` and `treex_site_summary.csv` contain aggregate
  statistics; and
- `treex_best_plots_by_strict_f1.csv` and
  `treex_worst_plots_by_strict_f1.csv` contain ranked diagnostic subsets.

The primary reported metric is strict F1. Labelled-mask F1 is retained as a
secondary diagnostic because it excludes predicted points outside the labelled
reference-tree mask.

Machine-specific plot lists, missing-path diagnostics, prediction paths,
per-plot intermediate files and pilot manifests are intentionally excluded.
Full `.las` and `.npz` predictions belong under the Git-ignored
`local_outputs/treex_predictions/` directory.
