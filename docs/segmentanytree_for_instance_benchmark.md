# SegmentAnyTree On FOR-instance

## Status

This is the next full labelled accuracy benchmark. The workflow is implemented,
but no SegmentAnyTree accuracy values are available until prediction,
normalisation and evaluation jobs complete successfully.

FOR-instance supports instance precision, recall, F1 and point-set IoU because
its annotated LAS files contain plot-wise `treeID` values. Evaluation includes
semantic classes `4` (stem), `5` (live branches) and `6` (woody branches), and
ignores classes `0`, `1`, `2` and `3` plus `treeID = 0`.

The pilot is `CULS/plot_1_annotated.las`. The full workflow selects all 32 LAS
files deterministically and preserves labels read from
`data_split_metadata.csv`.

## Upstream Interface Check

SegmentAnyTree is an external dependency and is not vendored here:

```bash
cd ~/scratch/tree-seg-benchmark
mkdir -p external
git clone https://github.com/SmartForest-no/SegmentAnyTree.git external/SegmentAnyTree
git -C external/SegmentAnyTree rev-parse HEAD
sed -n '1,240p' external/SegmentAnyTree/README.md
sed -n '1,240p' external/SegmentAnyTree/run_inference.sh
```

The upstream `main` branch documents:

```text
bash run_inference.sh <path_to_input_dir> <path_to_output_dir> <clean_output_dir>
```

The inspected script also contains installation-specific paths. Consequently,
`method.command_template` remains `null` in
`configs/for_instance_segmentanytree_benchmark.yml`. Inspect the installed
Barkla checkout, its dependencies, checkpoint paths, device requirements and
output schema before setting an argument-list template. Do not copy the
documented example into the active template without that check.

The wrapper never invokes a shell and records the expanded argument list. It
fails before segmentation when the checkout or template is missing.

## Environment Preflight

```bash
module purge
module load miniforge3/25.3.0-python3.12.10
source ~/fastscratch/venvs/treebench/bin/activate
cd ~/scratch/tree-seg-benchmark
python --version
python -m pytest
mkdir -p logs/segmentanytree_for_instance
```

Install SegmentAnyTree dependencies and checkpoints according to the inspected
checkout. Do not start a source build or large checkpoint download without
first checking Barkla storage, supported devices and upstream instructions.

The supplied prediction jobs default to the `nodes` partition and conservative
CPU resources. This is suitable for preflight and dry-run work only. If the
installed SegmentAnyTree version requires a GPU, use a verified Barkla GPU
partition and resource request when submitting prediction jobs; do not add a
guessed GPU directive to the scripts.

## Run Sequence

### 1. Inspect SegmentAnyTree

Run the upstream interface check above, record its commit, then set
`method.command_template` in the benchmark config. Also set or record the
observed prediction format and instance field. Keep `treeID` out of method
inputs.

### 2. Inventory FOR-instance

The log directory must exist before `sbatch` opens its output files:

```bash
mkdir -p logs/segmentanytree_for_instance
sbatch scripts/slurm/inspect_for_instance_inventory.sbatch
```

Review:

```bash
column -s, -t < results/metadata/segmentanytree_for_instance/inventory.csv | less -S
```

Confirm that all 32 files are present and that split labels were resolved.

### 3. Dry-run The Pilot

The pilot script is a dry-run by default:

```bash
sbatch scripts/slurm/run_segmentanytree_for_instance_pilot.sbatch
```

It must print the exact command, pilot input and output directory without
running SegmentAnyTree.

### 4. Run The Pilot Prediction

Only after the dry-run command, dependencies, checkpoint and device request
have been verified:

```bash
sbatch --export=ALL,SEGMENTANYTREE_EXECUTE=1 \
  scripts/slurm/run_segmentanytree_for_instance_pilot.sbatch
```

If a GPU is required, add only the partition and generic-resource values
confirmed for Barkla:

```bash
sbatch --partition=<verified_gpu_partition> --gres=<verified_gpu_request> \
  --export=ALL,SEGMENTANYTREE_EXECUTE=1 \
  scripts/slurm/run_segmentanytree_for_instance_pilot.sbatch
```

Inspect the pilot output and run metadata before continuing. A zero method
return code with no output files is treated as failure.

### 5. Normalise The Pilot

Declare the format observed in the pilot. For an output directory containing
one point cloud per tree:

```bash
python scripts/methods/normalise_segmentanytree_predictions.py \
  --input data/predictions/segmentanytree/for_instance/CULS/plot_1_annotated/final_results \
  --output-dir data/interim/segmentanytree/for_instance/CULS/plot_1_annotated/normalised_predictions \
  --format per_tree_directory
```

For one labelled LAS, LAZ or PLY, use
`--format labelled_point_cloud --instance-field <verified_field>`. The adapter
will not guess an instance field or undocumented format.

### 6. Evaluate The Pilot

Read the pilot split from the inventory, then run:

```bash
python scripts/evaluation/instance_iou_f1.py \
  --plot-name plot_1_annotated \
  --collection CULS \
  --split <verified_split> \
  --relative-path CULS/plot_1_annotated.las \
  --predicted-instance-dir data/interim/segmentanytree/for_instance/CULS/plot_1_annotated/normalised_predictions \
  --reference-labelled-point-cloud ~/data/datasets/for_instance/FORinstance_dataset/CULS/plot_1_annotated.las \
  --reference-label-field treeID \
  --reference-semantic-field classification \
  --reference-classes 4 5 6 \
  --ignored-reference-classes 0 1 2 3 \
  --ignore-reference-labels 0 \
  --iou-threshold 0.5 \
  --coordinate-tolerance 0.02 \
  --run-metadata-json results/metadata/segmentanytree_for_instance/runs/CULS/plot_1_annotated_run.json \
  --output-json results/metadata/segmentanytree_for_instance/evaluations/CULS/plot_1_annotated.json
```

### 7. Inspect Pilot Metrics

Review the metadata, metrics, matched pairs and unmatched instance tables.
Confirm class filtering, coordinate alignment, matching threshold and
tolerance before fixing the final workflow.

### 8. Run The Full Array

Do not train or tune on the test split. Do not submit the full array until the
pilot output schema and evaluation settings are fixed.

CPU submission example:

```bash
PRED_JOB=$(sbatch --parsable \
  --export=ALL,SEGMENTANYTREE_EXECUTE=1 \
  scripts/slurm/run_segmentanytree_for_instance_array.sbatch)
NORM_JOB=$(sbatch --parsable --dependency=afterok:${PRED_JOB} \
  --export=ALL,SEGMENTANYTREE_OUTPUT_FORMAT=per_tree_directory \
  scripts/slurm/normalise_segmentanytree_for_instance_array.sbatch)
EVAL_JOB=$(sbatch --parsable --dependency=afterok:${NORM_JOB} \
  scripts/slurm/evaluate_segmentanytree_for_instance_array.sbatch)
```

For a labelled point-cloud output, also export the verified field:

```bash
--export=ALL,SEGMENTANYTREE_OUTPUT_FORMAT=labelled_point_cloud,SEGMENTANYTREE_INSTANCE_FIELD=<verified_field>
```

Apply verified GPU options to the prediction submission only when required.
The array concurrency limits are conservative and can be adjusted after pilot
resource use is known.

### 9. Summarise The Benchmark

```bash
sbatch --dependency=afterok:${EVAL_JOB} \
  scripts/slurm/summarise_segmentanytree_for_instance.sbatch
```

The summary writes plot, collection and split tables plus benchmark metadata.
Failed or missing plot jobs must be resolved before reporting final values.

## Comparison Controls

- Do not train, tune or select parameters using the test split.
- Do not report test metrics until class filtering, normalisation, tolerance,
  IoU threshold and one-to-one matching are fixed.
- Do not compare with NEWFOR results unless both benchmarks use compatible
  metrics, IoU thresholds, coordinate tolerances and semantic class filters.
- NEWFOR can be added later through another config, dataset adapter, wrapper
  and the same evaluator pattern; it is not implemented here.
- TLS2trees remains a candidate compatibility test for FOR-instance and
  Wytham Woods. Wytham requires documented plot reconstruction from per-tree
  PLY references before a fair multi-method evaluation.
