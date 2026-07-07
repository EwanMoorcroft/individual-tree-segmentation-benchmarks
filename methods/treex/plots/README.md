# TreeX Plot Artifacts

This directory contains small public-safe TreeX summary plots regenerated from
the committed development and test summary CSVs:

- `treex_labelled_mask_f1_by_plot.png`
- `treex_predicted_vs_reference_counts.png`
- `treex_runtime_vs_strict_f1.png`
- `treex_strict_f1_by_plot.png`

Do not place raw predictions, point clouds, logs or private-path screenshots in
this directory.

Regenerate the plots and derived CSVs with:

```bash
python methods/treex/scripts/create_treex_final_summaries.py \
  --dev-csv methods/treex/examples/treex_dev_full_summary.csv \
  --test-csv methods/treex/examples/treex_test_full_summary.csv \
  --output-dir methods/treex/examples \
  --plot-dir methods/treex/plots
```
