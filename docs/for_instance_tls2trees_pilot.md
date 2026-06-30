# FOR-instance TLS2trees Accuracy Pilot

## Status And Scope

This workflow prepares the first labelled accuracy pilot:

- dataset: FOR-instance;
- plot: `CULS/plot_1_annotated.las`;
- method: TLS2trees instance stage;
- evaluation mode: leaf-off;
- reference instance field: `treeID`;
- reference semantic classes: `4` (stem) and `6` (woody branches).

The workflow is implemented but has not produced a reported accuracy result.
FOR-instance is UAV laser scanning data, so this run is also a compatibility
test for the TLS2trees instance stage.

Class `5` contains live branches and is excluded from the first accuracy
reference because the configured TLS2trees run produces `.leafoff.ply`
predictions. The converter can retain class `5` as TLS2trees label `1` for
separately labelled future experiments, but leaf-on and leaf-off results must
not be combined.

## Environment And Paths

- Project root: `~/scratch/tree-seg-benchmark`
- Dataset root: `~/data/datasets/for_instance/FORinstance_dataset`
- Python environment: `~/fastscratch/venvs/treebench`
- Module: `miniforge3/25.3.0-python3.12.10`
- Config: `configs/for_instance_tls2trees_accuracy.yml`
- TLS2trees commit: `ca12cb73b2c736d80b020e8025f8d975d42e6f01`

The source LAS file remains read-only. Converted inputs, predictions, logs and
full metric outputs remain outside version control.

## Preflight

Run on Barkla:

```bash
cd ~/scratch/tree-seg-benchmark
module purge
module load miniforge3/25.3.0-python3.12.10
source ~/fastscratch/venvs/treebench/bin/activate

mkdir -p logs/for_instance_tls2trees

test -f ~/data/datasets/for_instance/FORinstance_dataset/CULS/plot_1_annotated.las
test "$(git -C external/TLS2trees rev-parse HEAD)" = \
  "ca12cb73b2c736d80b020e8025f8d975d42e6f01"
python -m pytest
```

Create the log directory before `sbatch`; Slurm opens output files before the
job script runs.

## Submit The Pilot

Submit inventory, conversion, prediction and evaluation with dependencies:

```bash
cd ~/scratch/tree-seg-benchmark
mkdir -p logs/for_instance_tls2trees

INVENTORY_JOB=$(sbatch --parsable \
  scripts/slurm/inspect_for_instance_inventory.sbatch)

CONVERSION_JOB=$(sbatch --parsable --dependency=afterok:"$INVENTORY_JOB" \
  scripts/slurm/convert_for_instance_tls2trees_pilot.sbatch)

RUN_JOB=$(sbatch --parsable --dependency=afterok:"$CONVERSION_JOB" \
  scripts/slurm/run_tls2trees_for_instance_pilot.sbatch)

EVALUATION_JOB=$(sbatch --parsable --dependency=afterok:"$RUN_JOB" \
  scripts/slurm/evaluate_for_instance_tls2trees_pilot.sbatch)

printf 'inventory=%s\nconversion=%s\nrun=%s\nevaluation=%s\n' \
  "$INVENTORY_JOB" "$CONVERSION_JOB" "$RUN_JOB" "$EVALUATION_JOB"
squeue -u "$USER"
```

After conversion, the runner can also be checked without segmentation:

```bash
python scripts/methods/run_tls2trees_for_instance_plot.py \
  --config configs/for_instance_tls2trees_accuracy.yml \
  --plot-path CULS/plot_1_annotated.las \
  --collection CULS \
  --plot-name plot_1_annotated \
  --dry-run
```

The dry-run command must reference the patched instance script, numeric tile
`001`, the CULS converted input and the CULS prediction directory.

## Expected Outputs

- Inventory:
  `results/metadata/for_instance_tls2trees/inventory.{csv,json}`
- Converted tile:
  `data/interim/tls2trees/for_instance/CULS/plot_1_annotated/001.downsample.segmented.ply`
- Conversion metadata:
  `results/metadata/for_instance_tls2trees/conversions/CULS/`
- Predictions:
  `data/predictions/tls2trees/for_instance/CULS/plot_1_annotated/`
- Run metadata:
  `results/metadata/for_instance_tls2trees/runs/CULS/`
- Evaluation metadata:
  `results/metadata/for_instance_tls2trees/evaluations/CULS/`
- Evaluation tables:
  `results/tables/for_instance_tls2trees/CULS/`

The evaluator writes a one-row metrics table, matched pairs, unmatched
predictions, unmatched references and JSON metadata.

## Evaluation Interpretation

Reference coordinates are filtered to semantic classes `4` and `6`, grouped by
positive `treeID`, and quantised at 0.02 m. Predictions and references are
matched one-to-one at IoU 0.5.

Review coordinate alignment, retained point counts, output validity and matched
pairs before reporting precision, recall, F1 or mean matched IoU. A successful
process exit alone is not an accuracy result.

FRDR remains a prediction and operational benchmark because its benchmark LAZ
inputs do not contain individual-tree reference IDs. Wytham Woods remains a
future TLS accuracy candidate that requires a documented scene reconstruction.
