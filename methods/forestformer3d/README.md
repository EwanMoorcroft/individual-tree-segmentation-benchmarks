# ForestFormer3D

## Method Summary

ForestFormer3D is an end-to-end neural framework for semantic and individual
tree segmentation of forest LiDAR point clouds. This adapter targets exactly
two original FOR-instance variants: the unchanged official checkpoint and a
development-only fine-tune initialised from that checkpoint.

## Upstream Repository And Citation

The official source is
[`SmartForest-no/ForestFormer3D`](https://github.com/SmartForest-no/ForestFormer3D),
pinned here to commit
`6a75c3735e4a4108d02ee944a8b93177f2360a4f`. The paper is Xiang et al.,
*ForestFormer3D: A Unified Framework for End-to-End Segmentation of Forest
LiDAR 3D Point Clouds*, ICCV 2025
([arXiv:2506.16991](https://arxiv.org/abs/2506.16991)).

The source repository states CC BY-NC 4.0. The official Zenodo checkpoint
record states GPL-3.0-or-later. Both statements are retained without asserting
a legal resolution. No upstream source, checkpoint or container is committed.

## Training Mode Support

`published_pretrained` uses `epoch_3000_fix.pth` unchanged.
`fine_tuned_on_dev` will start from the identical checkpoint and may update
weights only on the canonical seed-42 16/5 subdivision of the 21 original
development plots. Its schedule and checkpoint-selection rule remain locked
until environment qualification and development smoke timing are complete.

The checkpoint exposure audit passes: all 11 original held-out plots map
exactly to the official V2 test inventory, none maps to V2 training or
validation, and all 21 original development plots map to V2 training or
validation. This establishes the exposure gate only; runtime and alignment
gates remain.

## Input Requirements

Inputs are the repository's frozen 32 original FOR-instance LAS files.
Development commands may read only the 21 development paths. Every source must
contain finite XYZ coordinates, `classification` and `treeID`.

Conversion will assign `source_row_index = 0..N-1` before transformation and
will supply XYZ to the model. Reference labels remain in a separate evaluation
sidecar unless an official loader bookkeeping requirement is proven unable to
affect predictions.

## Output Contract

The accepted harmonised output must have one row per original source point and
contain `pred_tree_id`, `target_tree_id`, `classification`,
`pred_classification` and `source_row_index`. Raw official outputs and aligned
arrays are retained separately. Rounded-coordinate rematching is not an
accepted primary correspondence route.

## FOR-instance Compatibility

The shared result uses classes 4, 5 and 6, ignores semantic classes 0 through
3 and reference instance labels 0 and -1, applies the union reference/predicted
tree mask, and performs maximum-cardinality one-to-one matching at IoU
greater than or equal to 0.5.

The exposure gate has passed. Full compatibility still requires checkpoint
runner loading, ground-truth-label independence, deterministic conversion,
exact row alignment and a manually reviewed development smoke.

## Barkla Environment

The official PyTorch 1.13.1/CUDA 11.6 base was qualified on an A100-SXM4-80GB
with compute capability 8.0. Barkla cannot execute the rootful
package-installation steps in the auditable image recipe because subordinate
ID mapping and the `fakeroot` helper are unavailable. The executable route
combines the hash-qualified base SIF with a user-owned Conda toolchain,
rootless virtual environment and exact source checkouts.

See [`docs/environment.md`](docs/environment.md).

## Slurm Workflow

`slurm/submit_environment_build.sh` submits a CPU rootless-environment build
followed by an A100 validation job. It refuses dirty or incorrect benchmark
branches, existing environment/run roots, build-script drift, base-SIF drift
and checkpoint hash mismatch.

Every submission writes a shell-safe state file and starts
`slurm/monitor_workflow.sh --watch 30` automatically. The monitor refreshes
queue, accounting and expected-file state every 30 seconds and exits on a
terminal Slurm state. Ctrl-C stops monitoring without cancelling jobs.

## Evaluation Route

Development smoke and full evaluation will use the shared
`for_instance_pointwise_v1` evaluator after source-row normalisation. The
unchanged checkpoint is diagnosed on development first. Fine-tuning selection
uses development validation only. Held-out submission remains unavailable
until a frozen readiness record is reviewed and explicitly authorised.

## Known Limitations

The official inference code selects its whole-plot route using a filename
substring and reads ground-truth arrays for output bookkeeping. Both behaviours
require counterfactual validation. The official `tools/test.py` also appears to
use `torch.load` without importing `torch`. No upstream modelling source will
be patched to conceal these issues.

Several official Docker dependencies are intentionally unpinned. The branch
therefore records the qualified base-SIF hash, Conda explicit specification
and complete pip inventory. The first successful resolution must become an
exact reusable lock before prediction runs.

## Current Benchmark Status

Upstream source, base environment, checkpoint identity, checkpoint exposure
and rootless feasibility are qualified. The composite dependency build and
checkpoint-load validation are the next gate. No development or held-out inference has run,
and no ForestFormer3D accuracy result exists.
