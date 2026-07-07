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

The exact installed `pointtree` package version is awaiting capture from the
completed Barkla environment. The package source is external and is not
vendored here.

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

## Evaluation protocol

Reference tree points are defined by:

- semantic classes `{4, 5, 6}`
- `treeID > 0`

Two protocols are reported:

1. Labelled-mask protocol
2. Strict whole-prediction protocol

Use strict F1 as the cautious headline result.

Matching is greedy one-to-one with IoU threshold `0.5`.

## Aggregate results

### Development

- Plots: 21
- Reference trees: 807
- Predicted trees: 1420
- True positives: 350
- False positives, labelled-mask: 523
- False positives, strict: 1070
- False negatives: 457
- Mean labelled-mask F1: 0.458660
- Median labelled-mask F1: 0.453333
- Mean strict F1: 0.340853
- Median strict F1: 0.342342
- Mean matched IoU: 0.761298
- Median matched IoU: 0.773742
- Mean runtime: 57.326995 s/plot
- Total runtime: 1203.866887 s

### Test

- Plots: 11
- Reference trees: 323
- Predicted trees: 653
- True positives: 186
- False positives, labelled-mask: 228
- False positives, strict: 467
- False negatives: 137
- Mean labelled-mask F1: 0.522187
- Median labelled-mask F1: 0.550000
- Mean strict F1: 0.402175
- Median strict F1: 0.412371
- Mean matched IoU: 0.829067
- Median matched IoU: 0.822880
- Mean runtime: 47.226986 s/plot
- Total runtime: 519.496842 s

Headline dissertation numbers:

- Test strict F1: `0.402`
- Test labelled-mask F1: `0.522`

The authoritative per-plot table is
[`../examples/treex_combined_dev_test_summary.csv`](../examples/treex_combined_dev_test_summary.csv).
It contains 32 unique plot rows and reconciles exactly with the split and site
aggregate tables.

## Interpretation

TreeX is not a high-performing method overall on this FOR-instance subset. It
often matches some trees with high IoU when a match succeeds, but it also
misses many reference trees and produces many extra instances. The gap between
labelled-mask and strict F1 is material and should be preserved in reporting.

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
