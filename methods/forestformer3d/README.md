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
`fine_tuned_on_dev` starts from the identical checkpoint and may update weights
only on the canonical seed-42 16/5 subdivision of the 21 original development
plots. Its single-configuration 35-epoch schedule, 280-step budget and
five-checkpoint validation-only selection rule are frozen in
[`docs/fine_tuning_protocol.md`](docs/fine_tuning_protocol.md).

The checkpoint exposure audit passes: all 11 original held-out plots map
exactly to the official V2 test inventory, none maps to V2 training or
validation, and all 21 original development plots map to V2 training or
validation. This establishes the exposure gate only; runtime and alignment
gates remain.

## Input Requirements

Inputs are the repository's frozen 32 original FOR-instance LAS files.
Development commands may read only the 21 development paths. Every source must
contain finite XYZ coordinates, `classification` and `treeID`.

Conversion assigns `source_row_index = 0..N-1` before transformation and
supplies XYZ to the model. Shared tree-material classes 4, 5 and 6 map to the
official internal wood label; all other classes map to loader-required ground.
Reference labels remain in an evaluation sidecar. The official loader also
receives masks, so the smoke runs the same XYZ once with mapped reference masks
and once with all-zero masks and requires identical raw prediction fields.

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

The qualified chain completed as jobs `9895305` and `9895306`. It imported the
official runtime stack, loaded the checkpoint as a mapping and confirmed CUDA
11.6 on an A100 at compute capability 8.0. The observed Conda toolchain
resolution is now the exact checked-in lock used by subsequent builds.

Every submission writes a shell-safe state file and starts
`slurm/monitor_workflow.sh --watch 30` automatically. The monitor refreshes
queue, accounting and expected-file state every 30 seconds and exits on a
terminal Slurm state. Ctrl-C stops monitoring without cancelling jobs.

`slurm/submit_one_plot_smoke.sh` is hard-coded to development plot
`CULS/plot_1_annotated.las`, checks the frozen split-metadata, environment,
base-image and checkpoint hashes, then runs both label-counterfactual cases in
one A100 allocation.

The smoke completed on an A100 and the benchmark operator accepted the manual
geometry alignment review. The public-safe decision record is
`examples/manual_alignment_confirmation_20260723.json`.

`slurm/submit_development_preflight.sh` freezes and inventories only the 21
development paths. Once that manifest is reviewed,
`slurm/submit_published_pretrained_development.sh` runs the unchanged official
checkpoint as a 21-task A100 array with maximum concurrency two, followed by a
CPU completeness, aggregation and retention gate. Both submitters start the
30-second live monitor. See
[`docs/published_pretrained_development.md`](docs/published_pretrained_development.md).

The published-pretrained development run completed and an independent CPU
verification re-hashed every retained artefact and rechecked all source-row
identity arrays. Fine-tuning preparation, exact-load smoke and full training
have separate guarded submitters; none can read held-out data.

## Evaluation Route

Development smoke and full evaluation will use the shared
`for_instance_pointwise_v1` evaluator after source-row normalisation. The
unchanged checkpoint is diagnosed on development first. Fine-tuning selection
uses development validation only. Held-out submission remains unavailable
until a frozen readiness record is reviewed and explicitly authorised.

## Known Limitations

The official inference code selects its whole-plot route using a filename
substring and reads ground-truth arrays for output bookkeeping. The smoke
retains the required `test` filename, pins and audits the effective prediction
source, and verifies identical model-facing point tensors counterfactually.
Prediction differences caused by the upstream nondeterministic CUDA reduction
are measured in the validation record rather than treated as label dependence.
The official `tools/test.py` uses `torch.load` without
importing `torch`; the adapter supplies that missing module through
`runpy.init_globals` after hashing the unchanged file. No upstream modelling
source is patched. The published checkpoint is already in the fixed sparse
layout, although the entrypoint applies the fix unconditionally. The adapter
accepts only the published hash and supplies the inverse layout so the
entrypoint reconstructs the original tensors exactly before loading.

Several official Docker dependencies are intentionally unpinned. The branch
therefore records the qualified base-SIF hash, Conda explicit specification
and complete pip inventory. The first successful Conda resolution is now the
exact reusable lock, and previously floating direct pip requirements are pinned
to the versions that passed A100 validation.

## Current Benchmark Status

Upstream source, base environment, checkpoint identity, checkpoint exposure,
the composite A100 runtime, label-independence evidence, exact row alignment
and the manually reviewed one-plot development smoke are qualified. The full
21-plot published-pretrained development diagnostic and independent retention
verification are complete. Development-only fine-tuning is frozen at the
preparation and exact-load-smoke gate. No held-out inference is available. No
held-out ForestFormer3D accuracy result exists.
