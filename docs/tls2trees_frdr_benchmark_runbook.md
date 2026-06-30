# TLS2trees FRDR Prediction Benchmark Runbook

## Objective

Run the patched TLS2trees instance stage reproducibly across the 16 FRDR plots on Barkla. This workflow produces tree predictions and descriptive output summaries. It does not establish segmentation accuracy.

Do not run this full workflow on the local Mac. Do not modify files under `~/data/datasets/frdr_treeiso`.

The completed run and public-safe per-plot summary are documented in
[`frdr_tls2trees_results.md`](frdr_tls2trees_results.md).

## Paths And Environment

- Barkla project: `~/scratch/tree-seg-benchmark`
- FRDR input data: `~/data/datasets/frdr_treeiso`
- Python environment: `~/fastscratch/venvs/treebench`
- Environment type: Python venv
- Module: `miniforge3/25.3.0-python3.12.10`
- TLS2trees repository: `~/scratch/tree-seg-benchmark/external/TLS2trees`
- Tested TLS2trees commit: `ca12cb73b2c736d80b020e8025f8d975d42e6f01`
- Patched instance script: `scripts/methods/tls2trees_patched/instance_patched.py`
- Config: `configs/frdr_tls2trees_benchmark.yml`

The runner refuses to execute if the checked-out TLS2trees commit differs from the tested commit.

The confirmed inventory contains 16 LAZ files and 205,602,855 points. Every file has a `woods` field. `NSpruce_plot2` additionally contains `woods = 0.0`; the full benchmark uses `unknown_policy: drop`, records the dropped count, and does not silently classify those points.

## Input Mapping

FRDR provides the semantic field `woods`:

- `woods = 1` means wood and is converted to TLS2trees `label = 3`.
- `woods = 2` means non-wood and is converted to TLS2trees `label = 1`.

Each plot is converted into a single numeric tile named `001.downsample.segmented.ply`, with columns `x`, `y`, `z`, `n_z`, and `label`. The tile name must be numeric because TLS2trees parses it as an integer. `tile_index.dat` contains the tested five-field layout: tile ID, X centre, Y centre, Z centre, and PLY path. The PLY deliberately excludes `buffer` and `fn`; the instance script creates those fields internally.

`n_z` currently uses `z - min(z)` for each plot. This local-minimum normalisation is a feasibility approximation, not terrain normalisation. Record it with every result and revisit it before treating outputs as final benchmark predictions.

## Why Semantic Segmentation Is Skipped

The tested workflow maps the existing FRDR `woods` semantic labels directly to the labels consumed by the TLS2trees instance stage. Running `semantic.py` would replace known FRDR wood/non-wood labels with model predictions and is therefore skipped.

The patched `instance_patched.py` is retained outside the external repository so the tested upstream checkout remains unchanged. The configured instance parameters reproduce the successful Barkla r8m spatial-crop feasibility run: 0.5 m slices, stem boundary 1.5-2.0 m, minimum 50 stem points, minimum 100 points per tree, cumulative graph gap 3.0 m, and no leaf attachment.

## Deploy To Barkla

From a local checkout of this protocol repository:

```bash
rsync -av --exclude .git/ ./ \
  barkla:~/scratch/tree-seg-benchmark/
```

The command assumes the SSH host alias is `barkla`. Replace only that host if the local SSH configuration uses a different name.

## Barkla Preflight

```bash
cd ~/scratch/tree-seg-benchmark
module purge
module load miniforge3/25.3.0-python3.12.10
source ~/fastscratch/venvs/treebench/bin/activate

mkdir -p \
  logs/tls2trees_frdr_full \
  results/metadata/tls2trees_conversions \
  results/metadata/tls2trees_runs \
  results/metadata/tls2trees_outputs \
  results/tables

python --version
python -c "import laspy, lazrs, numpy, yaml; print('core imports OK')"
git -C external/TLS2trees rev-parse HEAD
test "$(git -C external/TLS2trees rev-parse HEAD)" = \
  "ca12cb73b2c736d80b020e8025f8d975d42e6f01"
test -f scripts/methods/tls2trees_patched/instance_patched.py
python scripts/methods/tls2trees_patched/instance_patched.py --help
```

Do not continue if the commit check, patched script check, or imports fail.

Check available project-filesystem space before submitting arrays:

```bash
df -h ~/scratch/tree-seg-benchmark
du -h --max-depth=2 ~/scratch/tree-seg-benchmark/data 2>/dev/null | sort -h
export TLS2TREES_MIN_FREE_GB=50
```

Both heavy array scripts repeat this check inside every task and exit before processing when free space is below the threshold. Increase the environment variable if local Barkla storage policy requires a larger reserve.

## Inventory

Submit the read-only dataset inventory:

```bash
cd ~/scratch/tree-seg-benchmark
sbatch scripts/slurm/inspect_frdr_inventory.sbatch
squeue -u "$USER"
```

After completion:

```bash
less -S results/metadata/frdr_dataset_inventory.csv
```

Confirm all 16 configured plot names are present and have a `woods` field. The expected anomaly is `woods = 0.0` in `NSpruce_plot2`; investigate any additional unknown value before conversion.

## Pilot One Plot

Convert only array index 0 (`LPine_plot1`):

```bash
CONVERT_PILOT_JOB=$(sbatch --parsable --array=0-0 \
  scripts/slurm/convert_frdr_to_tls2trees_array.sbatch)
echo "$CONVERT_PILOT_JOB"
squeue -j "$CONVERT_PILOT_JOB"
```

After conversion succeeds:

```bash
ls -lh data/interim/tls2trees/frdr_full/LPine_plot1/
cat data/interim/tls2trees/frdr_full/LPine_plot1/tile_index.dat
cat results/metadata/tls2trees_conversions/LPine_plot1_conversion.json

python scripts/methods/run_tls2trees_instance_for_plot.py \
  --config configs/frdr_tls2trees_benchmark.yml \
  --plot-name LPine_plot1 \
  --dry-run
```

Inspect the dry-run command. It must use `001.downsample.segmented.ply`, the corresponding `tile_index.dat`, the patched script, and the LPine output directory.

Submit only that instance task:

```bash
RUN_PILOT_JOB=$(sbatch --parsable --array=0-0 \
  scripts/slurm/run_tls2trees_frdr_array.sbatch)
echo "$RUN_PILOT_JOB"
squeue -j "$RUN_PILOT_JOB"
```

After it completes:

```bash
cat results/metadata/tls2trees_runs/LPine_plot1_run.json

python scripts/methods/summarise_tls2trees_outputs.py \
  --plot-name LPine_plot1 \
  --output-dir data/predictions/tls2trees/frdr_full/LPine_plot1
```

Confirm the return code is zero and non-empty `.leafoff.ply` files were summarised before submitting the remaining plots.

## Remaining Plot Arrays

Because index 0 was completed as the pilot, submit indices 1 through 15:

```bash
export TLS2TREES_MIN_FREE_GB=50

CONVERT_JOB=$(sbatch --parsable --array=1-15 \
  scripts/slurm/convert_frdr_to_tls2trees_array.sbatch)

RUN_JOB=$(sbatch --parsable --dependency=afterok:"$CONVERT_JOB" --array=1-15 \
  scripts/slurm/run_tls2trees_frdr_array.sbatch)

SUMMARY_JOB=$(sbatch --parsable --dependency=afterok:"$RUN_JOB" \
  scripts/slurm/summarise_tls2trees_frdr_outputs.sbatch)

printf 'conversion=%s\ninstance=%s\nsummary=%s\n' \
  "$CONVERT_JOB" "$RUN_JOB" "$SUMMARY_JOB"
squeue -u "$USER"
```

If any array task fails, inspect its Slurm log and per-plot metadata. Do not blindly resubmit against a non-empty output directory.

The instance array defaults to 32 GiB per task. During the completed benchmark,
`Mixed_plot1` (array index 2) was killed for exceeding that allocation. It
completed when rerun with a 96 GiB allocation and recorded 49.602968 GiB peak
usage:

```bash
sbatch --array=2-2 --mem=96G scripts/slurm/run_tls2trees_frdr_array.sbatch
```

Retain the 32 GiB default for other plots unless scheduler evidence supports a
different allocation.

To summarise successful outputs even when one array task failed:

```bash
sbatch scripts/slurm/summarise_tls2trees_frdr_outputs.sbatch
```

## Expected Outputs

- Converted inputs: `data/interim/tls2trees/frdr_full/<plot>/001.downsample.segmented.ply`
- Tile indexes: `data/interim/tls2trees/frdr_full/<plot>/tile_index.dat`
- Predictions: `data/predictions/tls2trees/frdr_full/<plot>/*.leafoff.ply`
- Conversion metadata: `results/metadata/tls2trees_conversions/`
- Run metadata: `results/metadata/tls2trees_runs/`
- Output metadata: `results/metadata/tls2trees_outputs/`
- Per-tree tables: `results/tables/tls2trees_<plot>_tree_summary.csv`
- Combined plot table: `results/tables/tls2trees_frdr_prediction_summary.csv`
- Scheduler and method logs: `logs/tls2trees_frdr_full/`

## Accuracy Limitation

The FRDR LAZ files do not provide individual-tree instance labels. Predicted tree counts, point counts, runtimes, and output validity can be reported, but instance IoU, precision, recall, and F1 cannot be claimed from these files alone.

The evaluator refuses to run without a reference source and prints:

```text
No reference instance labels supplied; IoU/F1 cannot be computed.
```

When external reference labels become available, use either:

- a directory containing one LAS/LAZ/PLY file per reference tree; or
- one LAS/LAZ/PLY point cloud with an instance-ID field.

Coordinates are matched after quantisation at the selected tolerance. Document that tolerance and the reference-label provenance.

## Cleanup Rules

- Never remove or edit files under `~/data/datasets/frdr_treeiso`.
- Keep JSON/CSV metadata before deleting large generated files.
- The runner refuses non-empty per-plot prediction directories to prevent mixed reruns.
- To rerun one plot, first verify its metadata has been copied, then remove only:
  - `data/interim/tls2trees/frdr_full/<plot>/`
  - `data/predictions/tls2trees/frdr_full/<plot>/`
- Do not remove `external/TLS2trees` or the patched script as part of output cleanup.

## Copy Results Back To The Mac

Run locally:

```bash
cd /path/to/individual-tree-segmentation-benchmarks

rsync -av \
  barkla:~/scratch/tree-seg-benchmark/results/metadata/ \
  results/metadata/

rsync -av \
  barkla:~/scratch/tree-seg-benchmark/results/tables/ \
  results/tables/
```

Prediction PLYs and logs are intentionally excluded from this routine transfer. Copy selected files separately only when needed for validation; they remain ignored by Git.

To copy one completed prediction directory for inspection:

```bash
rsync -av \
  barkla:~/scratch/tree-seg-benchmark/data/predictions/tls2trees/frdr_full/LPine_plot1/ \
  data/predictions/tls2trees/frdr_full/LPine_plot1/
```

## Version-Control Exclusions

Do not commit FRDR LAS/LAZ files, converted PLY tiles, predictions, Slurm or method logs, copied Barkla result packs, or `external/TLS2trees`. Commit only source code, tests, configuration, documentation, and empty `.gitkeep` placeholders.
