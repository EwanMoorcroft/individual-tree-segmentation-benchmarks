# TreeLearn One-Plot FOR-instance Smoke Test

## Scope

This route checks whether TreeLearn can run one FOR-instance development plot
on Barkla with an upstream checkpoint. It does not install TreeLearn, download
weights, train, fine-tune, evaluate or select checkpoints.

## Configured Plot

- Dataset: FOR-instance
- Plot: `CULS/plot_1_annotated.las`
- Split: `dev`
- Expected point count: `1816672`
- Expected reference tree count: `6`

The plot is used because it is the configured small development pilot. The
held-out test split is not touched.

## Slurm Request

Expected runtime: 30-90 minutes after the environment and checkpoint exist.

Resources:

- Partition: `gpu-l40s-low`
- GPU: `--gres=gpu:1`
- CPUs: `12`
- Memory: `128G`
- Wall time: `02:00:00`

Live watch command if under 30 minutes:

```bash
squeue -j <job_id>
tail -f logs/treelearn_for_instance/treelearn_one_plot_smoke_<job_id>.out
```

## Run Through Slurm

```bash
cd ~/scratch/tree-seg-benchmark
mkdir -p logs/treelearn_for_instance

TREELEARN_SMOKE_CONFIRMED=1 \
  sbatch methods/treelearn/slurm/run_for_instance_one_plot_smoke.sbatch
```

Optional overrides:

```bash
TREELEARN_ENV="$HOME/fastscratch/venvs/treelearn" \
TREELEARN_REPO="$HOME/fastscratch/external/TreeLearn" \
TREELEARN_CHECKPOINT="$HOME/fastscratch/treelearn_checkpoints/model_weights_with_small_20241213.pth" \
TREELEARN_DATASET_ROOT="$HOME/data/datasets/for_instance/FORinstance_dataset" \
TREELEARN_SMOKE_CONFIRMED=1 \
  sbatch methods/treelearn/slurm/run_for_instance_one_plot_smoke.sbatch
```

## Output Paths

TreeLearn raw runtime root:

```text
data/interim/treelearn/for_instance/one_plot_smoke/CULS_plot_1_annotated/
```

Generated TreeLearn pipeline config:

```text
data/interim/treelearn/for_instance/one_plot_smoke/CULS_plot_1_annotated/treelearn_pipeline_config.yml
```

Raw TreeLearn full-forest outputs:

```text
data/interim/treelearn/for_instance/one_plot_smoke/CULS_plot_1_annotated/results/full_forest/CULS_plot_1_annotated.laz
data/interim/treelearn/for_instance/one_plot_smoke/CULS_plot_1_annotated/results/full_forest/CULS_plot_1_annotated.npz
```

Raw pointwise output:

```text
data/interim/treelearn/for_instance/one_plot_smoke/CULS_plot_1_annotated/results/pointwise_results/pointwise_results.npz
```

Adapted benchmark outputs:

```text
data/predictions/treelearn/for_instance_smoke/CULS_plot_1_annotated/CULS_plot_1_annotated_treelearn_smoke_predictions.npz
data/predictions/treelearn/for_instance_smoke/CULS_plot_1_annotated/CULS_plot_1_annotated_treelearn_smoke_predictions.las
```

Metadata:

```text
results/metadata/treelearn_for_instance/one_plot_smoke/CULS_plot_1_annotated_treelearn_smoke_metadata.json
```

Logs:

```text
logs/treelearn_for_instance/treelearn_one_plot_smoke_<job_id>.out
logs/treelearn_for_instance/treelearn_one_plot_smoke_<job_id>.err
```

## Success Criteria

- TreeLearn imports from the active environment.
- The configured checkpoint exists and loads.
- The one-plot pipeline exits with status zero.
- The full-forest output exists.
- The adapted prediction NPZ exists.
- Prediction row count equals the source point count.
- Source and prediction coordinates are row-aligned within tolerance.
- Predicted instance labels contain at least one positive tree ID.
- Metadata JSON records runtime, checkpoint path and output paths.

## Failure Indicators

- Missing TreeLearn environment, checkout or checkpoint.
- CUDA, PyTorch or spconv import failure.
- Tile generation or feature calculation exceeds the Slurm memory request.
- Pipeline exits before writing full-forest output.
- Output point count differs from source point count.
- Row-wise coordinate check fails.
- Prediction contains only non-tree or unassigned labels.
- Any test-split path is used.

## Interpretation

Passing this smoke test means the TreeLearn inference path, checkpoint loading
and output adaptation are viable for one FOR-instance development plot. It is
not an accuracy result and does not justify running full training.
