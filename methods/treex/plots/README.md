# TreeX Plot Artifacts

This directory contains small public-safe TreeX summary plots regenerated from
the committed development and test summary CSVs:

- `treex_labelled_mask_f1_by_plot.png`
- `treex_predicted_vs_reference_counts.png`
- `treex_runtime_vs_strict_f1.png`
- `treex_strict_f1_by_plot.png`

The two established `strict` filenames are retained for stable links. Their
metric is the harmonised union-mask result with strict one-to-one matching.

Do not place raw predictions, point clouds, logs or private-path screenshots in
this directory.

Regenerate all source tables, derived CSVs and plots from retained predictions
with:

```bash
python methods/treex/scripts/rebuild_treex_public_results.py
```
