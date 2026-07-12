# TreeLearn

## Method Summary

TreeLearn is a deep learning method for individual-tree segmentation in forest
point clouds. The current repository route is limited to a guarded, run-scoped
one-plot FOR-instance development smoke using released TreeLearn weights. It
performs inference and fixed harmonised evaluation, but does not train or
fine-tune weights and cannot submit held-out test data.

## Upstream Repository And Citation

The upstream implementation is `ecker-lab/TreeLearn`:
<https://github.com/ecker-lab/TreeLearn>.

The method paper is:
Henrich et al., TreeLearn: A deep learning method for segmenting individual
trees from ground-based LiDAR forest point clouds.

## Training Mode Support

The one-plot smoke route is `published_pretrained`: it loads the upstream
default `model_weights_20241213.pth` checkpoint and performs inference only.
The checkpoint SHA-256 is frozen at submission and recorded in run metadata.

The candidate full benchmark modes remain:

- `published_pretrained`
- `fine_tuned_on_dev`
- `retrained_from_dev`

Fine-tuning and retraining are not implemented in this first route.

## Input Requirements

The smoke test uses one FOR-instance LAS file with `treeID` and
`classification` dimensions. The configured plot is `CULS/plot_1_annotated.las`
from the development split.

Tree material follows the shared FOR-instance protocol:

- tree classes: `4`, `5`, `6`
- ignored classes: `0`, `1`, `2`, `3`
- reference instance field: `treeID`
- ignored reference labels: `0`, `-1`

## Output Contract

TreeLearn raw outputs stay under `data/interim/treelearn/...` and full
prediction outputs stay under `data/predictions/treelearn/...`.

Each smoke run uses a new run ID and writes:

- an aligned compressed prediction file with `pred_tree_id`,
  `target_tree_id`, `classification`, `pred_classification` and
  `source_row_index`;
- a LAS copy of the source plot with `pred_treeID` and
  `pred_classification` dimensions;
- the original-resolution upstream full-forest LAZ and NPZ plus the upstream
  voxel-level diagnostic `pointwise_results.npz`; and
- metrics, matched pairs, unmatched predictions and unmatched references.

The aligned prediction NPZ is the evaluation input. Positive TreeLearn
instances map to prediction semantic class `4`; labels `0` and `-1` map to
background. The upstream `pointwise_results.npz` is retained for diagnostics
but is not source-row aligned and is never used as the primary evaluation
input.

## FOR-instance Compatibility

The smoke test checks installation, checkpoint loading, one-plot inference,
row preservation, prediction adaptation and evaluator compatibility. Its
development score is diagnostic evidence only and must not be used to alter
the frozen checkpoint, IoU threshold or post-processing.

## Barkla Environment

The route pins upstream commit
`fd240ce7caa4c444fe3418aca454dc578bc557d4` and expects a Python 3.10,
PyTorch 2.0.0/CUDA 11.8 environment on Barkla. A non-destructive guarded setup
job creates missing prerequisites and reuses existing paths only when they
already pass validation. Defaults are recorded in
[`configs/for_instance_one_plot_smoke.yml`](configs/for_instance_one_plot_smoke.yml).

The Slurm entrypoint expects:

- `TREELEARN_ENV`, defaulting to `~/fastscratch/venvs/treelearn`;
- `TREELEARN_REPO`, defaulting to `~/fastscratch/external/TreeLearn`; and
- `TREELEARN_CHECKPOINT`, defaulting to the upstream December 2024 default
  checkpoint recorded in the config.

The dependent CPU evaluation also requires the repository's existing
`~/fastscratch/venvs/treebench` environment.

## Slurm Workflow

Use
[`slurm/setup_treelearn_environment.sbatch`](slurm/setup_treelearn_environment.sbatch)
only when prerequisites are absent. Submit the run through
[`slurm/submit_for_instance_one_plot_smoke.sh`](slurm/submit_for_instance_one_plot_smoke.sh),
which freezes the checkpoint hash and schedules inference followed by CPU
evaluation. It also freezes the clean benchmark checkout commit. The route is
guarded by `TREELEARN_SMOKE_CONFIRMED=1` and refuses existing run roots.

Expected runtime: 30-90 minutes after the environment and checkpoint exist.

Resources: one L40S GPU, 12 CPUs, 128 GB RAM and two hours wall time on
`gpu-l40s-low`.

## Evaluation Route

The dependent evaluation job uses `for_instance_pointwise_v1`, the union of
reference-tree and predicted-tree points, maximum-cardinality one-to-one
matching and IoU `>= 0.5`. It writes matched and unmatched instance tables.
No prediction-size filtering or threshold selection is permitted.

## Known Limitations

- The route does not train, fine-tune or select a checkpoint.
- The first smoke uses one development plot only.
- The setup follows upstream dependency pins, but upstream leaves some pip
  packages only partially pinned; the resolved Barkla environment must be
  retained before a full benchmark.
- A manual alignment review is still required after the smoke passes.
- The smoke cannot establish cross-site performance.

## Current Benchmark Status

TreeLearn is pending. The guarded one-plot published-checkpoint development
smoke and shared evaluation route are ready but have not yet been run. Full
development and held-out test arrays are intentionally absent.
