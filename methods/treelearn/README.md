# TreeLearn

## Method Summary

TreeLearn is a deep learning method for individual-tree segmentation in forest
point clouds. The current repository route is limited to a one-plot
FOR-instance smoke test using released TreeLearn weights. It does not train or
fine-tune weights.

## Upstream Repository And Citation

The upstream implementation is `ecker-lab/TreeLearn`:
<https://github.com/ecker-lab/TreeLearn>.

The method paper is:
Henrich et al., TreeLearn: A deep learning method for segmenting individual
trees from ground-based LiDAR forest point clouds.

## Training Mode Support

The one-plot smoke route is `published_pretrained`: it loads an upstream
checkpoint and performs inference only.

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

The smoke adapter writes:

- an aligned compressed prediction file with `pred_tree_id`,
  `target_tree_id`, `classification` and `source_row_index`; and
- a LAS copy of the source plot with an added `pred_treeID` dimension.

The aligned prediction file is the preferred evaluation input if the smoke test
is later connected to the harmonised evaluator.

## FOR-instance Compatibility

The smoke test checks only installation, checkpoint loading, one-plot
inference, row preservation and prediction adaptation. It is not an accuracy
benchmark and it must not be used for checkpoint choice.

## Barkla Environment

The route expects an external TreeLearn checkout and a Python environment on
Barkla. Defaults are recorded in
[`configs/for_instance_one_plot_smoke.yml`](configs/for_instance_one_plot_smoke.yml).

The Slurm entrypoint expects:

- `TREELEARN_ENV`, defaulting to `~/fastscratch/venvs/treelearn`;
- `TREELEARN_REPO`, defaulting to `~/fastscratch/external/TreeLearn`; and
- `TREELEARN_CHECKPOINT`, defaulting to the small-tree upstream weights path
  recorded in the config.

## Slurm Workflow

Use
[`slurm/run_for_instance_one_plot_smoke.sbatch`](slurm/run_for_instance_one_plot_smoke.sbatch)
for the first Barkla smoke test. It is guarded by
`TREELEARN_SMOKE_CONFIRMED=1`.

Expected runtime: 30-90 minutes after the environment and checkpoint exist.

Resources: one L40S GPU, 12 CPUs, 128 GB RAM and two hours wall time on
`gpu-l40s-low`.

## Evaluation Route

No evaluation job is included in this first route. The output adapter produces
point-aligned prediction arrays so a later route can call the shared
FOR-instance evaluator without coordinate rematching.

## Known Limitations

- The route does not install TreeLearn.
- The route does not download checkpoints.
- The route does not train, fine-tune or select a checkpoint.
- The first smoke uses one development plot only.
- The upstream LAS loader has its own FOR-instance class mapping, so any later
  fine-tuning route must make the semantic mapping explicit.

## Current Benchmark Status

TreeLearn is pending. The only implemented route is the guarded one-plot
published-checkpoint smoke test.
