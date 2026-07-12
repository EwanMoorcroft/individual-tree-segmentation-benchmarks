# TreeLearn

## Method Summary

TreeLearn is a deep learning method for individual-tree segmentation in forest
point clouds. The repository now has two guarded FOR-instance routes using the
released TreeLearn weights: the completed one-plot development smoke and a
completed 21-plot development-only evaluation. Both perform inference and
fixed harmonised evaluation without training or fine-tuning. Neither route can
submit held-out test data.

## Upstream Repository And Citation

The upstream implementation is `ecker-lab/TreeLearn`:
<https://github.com/ecker-lab/TreeLearn>.

The method paper is:
Henrich et al., TreeLearn: A deep learning method for segmenting individual
trees from ground-based LiDAR forest point clouds.

## Training Mode Support

Both routes use `published_pretrained`: they load the upstream default
`model_weights_20241213.pth` checkpoint and perform inference only. The frozen
checkpoint has MD5 `56a3d78f689ae7f1190906b975700311` and SHA-256
`5df2f92828f92755bc12e114eaebe83f7ecea94a74c25a6170b68844cc5e19bb`.
Each submission rechecks the checkpoint and records its identity in run
metadata.

The candidate full benchmark modes remain:

- `published_pretrained`
- `fine_tuned_on_dev`
- `retrained_from_dev`

Fine-tuning and retraining are not implemented in this first route.

## Input Requirements

The accepted smoke uses `CULS/plot_1_annotated.las`. The full development route
freezes the exact 21 locally available paths identified as `dev` in
`data_split_metadata.csv`: CULS 2, NIBIO 14, RMIT 1, SCION 3 and TUWIEN 1.
Every input LAS must contain `treeID` and `classification` dimensions. The
manifest records each input hash, point count, reference-tree count and split
metadata hash before the array is submitted.

Tree material follows the shared FOR-instance protocol:

- tree classes: `4`, `5`, `6`
- ignored classes: `0`, `1`, `2`, `3`
- reference instance field: `treeID`
- ignored reference labels: `0`, `-1`

## Output Contract

TreeLearn raw outputs stay under `data/interim/treelearn/...` and full
prediction outputs stay under `data/predictions/treelearn/...`.

Each run uses a new run ID. For every plot it writes:

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

The smoke checked installation, checkpoint loading, one-plot inference, row
preservation, prediction adaptation and evaluator compatibility. Run
`treelearn_for-instance_published_pretrained_dev_smoke_20260712_135205` passed
those checks and was manually accepted for full development-only evaluation.
Its F1 is diagnostic evidence only and was not used to alter the checkpoint,
IoU threshold or post-processing.

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

After the accepted smoke evidence and its five retained artefacts pass the
preparation gate, the guarded full route is submitted through
[`slurm/submit_for_instance_development.sh`](slurm/submit_for_instance_development.sh).
Its dependency chain freezes the exact 21-plot development manifest, runs at
most two concurrent combined inference/evaluation GPU tasks, accounts for
every task, then builds per-plot, per-site and whole-development summaries.
See the
[`full development runbook`](docs/development_evaluation.md). No held-out test
job is part of the dependency chain.

## Evaluation Route

Evaluation uses `for_instance_pointwise_v1`, the union of reference-tree and
predicted-tree points, maximum-cardinality one-to-one matching and IoU
`>= 0.5`. It writes matched and unmatched instance tables. The full
development summary aggregates TP, FP and FN counts before computing micro
metrics and reports CULS, NIBIO, RMIT, SCION and TUWIEN separately. No
prediction-size filtering or threshold selection is permitted.

## Known Limitations

- The route does not train, fine-tune or select a checkpoint.
- The accepted smoke score represents one CULS development plot only and is
  not an overall or cross-site estimate.
- The setup follows upstream dependency pins, but upstream leaves some pip
  packages only partially pinned; the resolved Barkla environment must be
  retained with the run evidence.
- The completed result is development-only and must not be presented as a
  held-out benchmark score.
- The held-out test split remains locked and no TreeLearn test route exists.

## Current Benchmark Status

The guarded one-plot published-checkpoint development smoke is complete and
accepted. It evaluated 1,816,672 source-aligned points with F1 `0.705882`,
precision `0.545455`, recall `1.000000`, TP `6`, FP `5` and FN `0`. Row count
and order were preserved, the maximum coordinate delta was
`0.00027683842927217484` m and all five raw/aligned prediction artefacts were
retained. The exact 21-plot full development run
`treelearn_for-instance_published_pretrained_development_20260712_150030`
also completed with zero failures. Its mean plot F1 is `0.515571` and its
count-aggregated micro F1 is `0.510760` (micro precision `0.415888`, micro
recall `0.661710`). CULS has the highest site mean F1 (`0.715010`) and NIBIO
the lowest (`0.446965`). All 105 raw and aligned prediction artefacts, totalling
9,645,423,654 bytes, passed retention verification. See the
[`completed development result`](docs/development_results_20260712.md). No
held-out test result exists.
