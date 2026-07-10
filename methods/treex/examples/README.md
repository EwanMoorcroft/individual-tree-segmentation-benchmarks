# TreeX Public Results

`treex_combined_dev_test_summary.csv` is the authoritative public result
table. It contains one row for each of the 32 exact-path FOR-instance plots:
21 development plots and 11 held-out test plots.

The remaining CSV files are reproducible views of that table:

- `treex_dev_full_summary.csv` and `treex_test_full_summary.csv` contain the
  source split tables;
- `treex_run_metadata.csv` contains the immutable plot IDs, point counts,
  instance counts and measured runtimes used by the local rebuild;
- `treex_split_summary.csv` and `treex_site_summary.csv` contain aggregate
  counts, mean plot metrics and count-derived micro metrics; and
- `treex_best_plots_by_strict_f1.csv` and
  `treex_worst_plots_by_strict_f1.csv` contain ranked harmonised diagnostic
  subsets. The established filenames retain `strict` to preserve links; here
  it means strict one-to-one matching.

The primary metric is harmonised union-mask F1 with maximum-cardinality
one-to-one matching at IoU `>= 0.5`. Reference-labelled-mask F1 is retained as
a secondary diagnostic because it excludes prediction support on reference
background points.

Rebuild every table and plot from the retained ignored NPZ files with:

```bash
python methods/treex/scripts/rebuild_treex_public_results.py
```

Machine-specific plot lists, missing-path diagnostics, prediction paths,
per-plot intermediate files and pilot manifests are intentionally excluded.
Full `.las` and `.npz` predictions belong under the Git-ignored
`local_outputs/treex_predictions/` directory.
