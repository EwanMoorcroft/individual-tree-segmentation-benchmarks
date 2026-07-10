# FOR-instance TreeX Benchmark

## Status

This method run is complete on the locally available FOR-instance exact-path
subset on Barkla.

- Method: `TreeXAlgorithm` from `pointtree`
- Preset: `TreeXPresetULS`
- Mode: unsupervised parameterised run, not training
- Dataset type: ULS
- Environment: `~/fastscratch/venvs/treex`
- Barkla repo root: `~/scratch/tree-seg-benchmark`

Upstream source: <https://github.com/ai4trees/pointtree>. Method citation:
<https://doi.org/10.48550/arXiv.2509.03633>.

The exact installed `pointtree` package version was not captured in the
retained Barkla metadata. This is an explicit reproducibility limitation, not
an inferred version. The package source is external and is not vendored here.

## Exact-path split rule

The workflow uses only metadata paths that exist locally.

- No basename fallback across collections
- No `NIBIO2` to `NIBIO` remapping
- No synthetic split expansion

Final local counts:

- Development: 21 plots
- Test: 11 plots
- Existing exact paths: 32 plots
- Missing exact paths: 50 metadata rows

The Barkla result lists use these CSV schemas:

- `plot_id,input_las,metadata_path,mapping_rule`
- `plot_id,input_las,metadata_path,folder,split,mapping_rule`
- `metadata_path,folder,split,expected_las`

## Runtime profile

The completed run used the profile
`all_points_real_intensity_no_intensity_filter`.

```python
params = dict(TreeXPresetULS())
params["invalid_tree_id"] = -1
params["num_workers"] = 8
params["visualization_folder"] = None
params["random_seed"] = 0
params["stem_search_min_cluster_intensity"] = None
```

The intensity threshold is disabled because an earlier pilot accidentally used
classification values as intensity and produced zero detections.

## Reproduction workflow

Create exact-path plot lists from the supplied metadata:

```bash
python methods/treex/scripts/make_treex_for_instance_exact_split_lists.py \
  --dataset-root "$HOME/data/datasets/for_instance/FORinstance_dataset" \
  --metadata-csv "$HOME/data/datasets/for_instance/FORinstance_dataset/data_split_metadata.csv" \
  --existing-output results/treex_for_instance/treex_for_instance_existing_exact_paths.csv \
  --missing-output results/treex_for_instance/treex_for_instance_missing_exact_paths.csv \
  --dev-output results/treex_for_instance/treex_for_instance_dev_plots.csv \
  --test-output results/treex_for_instance/treex_for_instance_test_plots.csv
```

Create the evaluation lists before submitting the arrays:

```bash
python methods/treex/scripts/make_treex_eval_list.py \
  --input-csv results/treex_for_instance/treex_for_instance_dev_plots.csv \
  --output-csv results/treex_for_instance/treex_eval_summary_list.csv

python methods/treex/scripts/make_treex_test_eval_list.py \
  --input-csv results/treex_for_instance/treex_for_instance_test_plots.csv \
  --output-csv results/treex_for_instance/treex_test_eval_summary_list.csv
```

The checked-in Slurm files default to task `0` as a safety pilot. Override the
array ranges for the recorded 21 development and 11 test plots:

```bash
DEV_RUN_JOB=$(sbatch --parsable --array=0-20%4 \
  methods/treex/slurm/run_treex_for_instance_dev_array.sbatch)
DEV_EVAL_JOB=$(sbatch --parsable --dependency=afterok:"$DEV_RUN_JOB" \
  --array=0-20%4 \
  methods/treex/slurm/evaluate_treex_for_instance_array.sbatch)

TEST_RUN_JOB=$(sbatch --parsable --array=0-10%4 \
  methods/treex/slurm/run_treex_for_instance_test_array.sbatch)
TEST_EVAL_JOB=$(sbatch --parsable --dependency=afterok:"$TEST_RUN_JOB" \
  --array=0-10%4 \
  methods/treex/slurm/evaluate_treex_for_instance_test_array.sbatch)
```

After all evaluation tasks complete, collect the per-plot JSON records and
regenerate the aggregate tables and plots:

```bash
python methods/treex/scripts/create_treex_split_summary.py \
  --plot-list results/treex_for_instance/treex_for_instance_dev_plots.csv \
  --results-root results/treex_for_instance \
  --output-csv results/treex_for_instance/treex_dev_full_summary.csv \
  --split dev

python methods/treex/scripts/create_treex_split_summary.py \
  --plot-list results/treex_for_instance/treex_for_instance_test_plots.csv \
  --results-root results/treex_for_instance \
  --output-csv results/treex_for_instance/treex_test_full_summary.csv \
  --split test

python methods/treex/scripts/create_treex_final_summaries.py
```

The ignored local backup contains the point-aligned NPZ files needed to
recompute the public metrics. Rebuild the committed tables and plots directly
from those retained arrays with:

```bash
python methods/treex/scripts/rebuild_treex_public_results.py
```

## Evaluation protocol

Reference tree points are defined by:

- semantic classes `{4, 5, 6}`
- `treeID > 0`

Two views are reported from separate IoU matrices:

1. Harmonised union-mask protocol: the union of valid reference-tree points
   and points assigned a non-invalid TreeX instance.
2. Reference-labelled-mask diagnostic: valid reference-tree points only.

The harmonised union-mask result is primary. The labelled-mask value is not a
cross-method headline metric because it excludes predicted support on
reference background points.

Both views use maximum-cardinality one-to-one matching at IoU `>= 0.5`.

## Aggregate results

### Development

- Plots: 21
- Reference trees: 807
- Predicted trees: 1420
- Harmonised TP / FP / FN: 322 / 1098 / 485
- Harmonised mean plot F1: 0.314321
- Harmonised median plot F1: 0.301887
- Harmonised micro F1: 0.289178
- Labelled-mask TP / FP / FN: 350 / 523 / 457
- Labelled-mask mean plot F1: 0.458660
- Labelled-mask micro F1: 0.416667
- Mean plot-level matched IoU, harmonised: 0.730934
- Mean runtime: 57.326995 s/plot
- Total runtime: 1203.866887 s

### Test

- Plots: 11
- Reference trees: 323
- Predicted trees: 653
- Harmonised TP / FP / FN: 177 / 476 / 146
- Harmonised mean plot F1: 0.383108
- Harmonised median plot F1: 0.384615
- Harmonised micro F1: 0.362705
- Labelled-mask TP / FP / FN: 186 / 228 / 137
- Labelled-mask mean plot F1: 0.522187
- Labelled-mask micro F1: 0.504749
- Mean plot-level matched IoU, harmonised: 0.803764
- Mean runtime: 47.226986 s/plot
- Total runtime: 519.496842 s

Headline dissertation numbers:

- Test harmonised mean plot F1: `0.3831`
- Test harmonised micro F1: `0.3627`
- Test labelled-mask mean plot F1, diagnostic only: `0.5222`

The authoritative per-plot table is
[`../examples/treex_combined_dev_test_summary.csv`](../examples/treex_combined_dev_test_summary.csv).
It contains 32 unique plot rows and reconciles exactly with the split and site
aggregate tables.

## Interpretation

TreeX is not a high-performing method overall on this FOR-instance subset. It
often matches some trees with high IoU when a match succeeds, but it also
misses many reference trees and produces many extra instances. The gap between
the reference-labelled-mask diagnostic and harmonised union-mask F1 is
material and must remain explicit in reporting.

## Local prediction backup

All prediction outputs should stay outside Git. Use
[`scripts/rsync_treex_predictions_from_barkla.sh`](../scripts/rsync_treex_predictions_from_barkla.sh)
to copy them into `local_outputs/treex_predictions/` on the Mac.

Use
[`scripts/rsync_treex_public_results_from_barkla.sh`](../scripts/rsync_treex_public_results_from_barkla.sh)
to refresh the committed public-safe CSV and plot artifacts from Barkla.
Regenerate the plot files from the committed source CSVs before publication so
their rendering does not depend on Barkla's Matplotlib backend.

The local backup audit on 6 July 2026 found 32 final `.npz` files and 32 final
`.las` files, one pair per evaluated plot. Pilot output and machine-specific
path manifests are excluded from the public artifacts.

## Barkla jobs

| Stage | Job ID | Outcome |
| --- | ---: | --- |
| TreeX pilot v2 | `9653979` | Completed |
| Development evaluation array | `9654326` | Completed |
| Held-out test inference array | `9654351` | Completed |
| Held-out test evaluation array | `9654371` | Completed |
