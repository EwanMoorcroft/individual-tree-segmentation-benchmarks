# ForAINet

## Method Summary

This directory integrates the original ForAINet release as the reproducible
fallback for the unavailable ForAINetV2 release. The selected method slug is
`forainet`. The integration is development-only until every readiness gate in
the FOR-instance method-adapter protocol has passed.

## Upstream Repository And Citation

The pinned source is [`prs-eth/ForAINet`](https://github.com/prs-eth/ForAINet)
at commit `5fe600ae8f2fe913ae8740f475f0261a702f2a72`. The source repository is
licensed under BSD 3-Clause. The method is described in Xiang et al.,
"Automated forest inventory: analysis of high-density airborne LiDAR point
clouds with 3D deep learning", Remote Sensing of Environment 305 (2024), 114078,
<https://doi.org/10.1016/j.rse.2024.114078>.

The ForestFormer3D paper reports `ForAINetV2_R8` and `ForAINetV2_R16`, but no
complete official ForAINetV2 implementation, checkpoint, configuration and
fine-tuning package was released. The evidence and fallback decision are
recorded in [`docs/upstream_qualification.md`](docs/upstream_qualification.md).

## Training Mode Support

Two future result roles are separated:

- `published_pretrained`: the unchanged official `PointGroup-PAPER.pt` file;
- `fine_tuned_on_dev`: official checkpoint weight initialisation followed by
  updates on the fixed 16 development-training plots and selection on the
  fixed five development-validation plots.

The official checkpoint-initialisation hook is
`models.PointGroup-PAPER.path_pretrained` with `weight_name=latest` and an
empty training checkpoint directory. It is a weight-initialisation route, not
a resume route. No training or held-out inference has been run by this branch.

## Input Requirements

The source of truth is the original 32-file FOR-instance LAS catalogue. Every
input must be a repository-catalogued relative path with `classification` and
`treeID` fields. Alignment sidecars retain the exact integer source row,
source hash, point count, semantic values and positive reference-tree count.

The official five-class preparation maps original classes 1, 2, 4, 5 and 6 to
model classes 0 through 4 and drops class 3 outliers. A Barkla qualification
run must prove the retained-row inverse map before the official converter can
be accepted. Development and held-out roots are physically separate.

## Output Contract

The primary retained artefact is a compressed, source-row-aligned array with:

- `pred_tree_id`;
- `target_tree_id`;
- `classification`;
- `pred_classification`; and
- `source_row_index`.

Raw official outputs remain separate. The adapter accepts only stable integer
row identifiers. It rejects missing, duplicated, out-of-range or conflicting
rows and never falls back to rounded-coordinate matching.

## FOR-instance Compatibility

The canonical evaluation protocol is `for_instance_pointwise_v1`: reference
tree classes 4, 5 and 6, ignored reference classes 0 through 3, ignored
reference IDs 0 and -1, the union of predicted and reference tree points,
IoU `>= 0.5`, and maximum-cardinality one-to-one matching.

The exposure audit identifies all 11 operational held-out plots as exact
matches to the official test-only list. The public release does not bind its
42/14 train/validation subdivision to a file manifest, so the 21 operational
development rows are conservatively labelled `train_or_validation`.

## Barkla Environment

ForAINet must use an isolated Apptainer image rather than the shared utility
environment or the SegmentAnyTree container. Barkla provides Apptainer 1.3.6.
The image, checkpoint, upstream checkout, converted data, predictions and logs
remain external to Git and are addressed through environment variables
documented in the smoke configuration.

The upstream release targets Python 3.8, PyTorch 1.9 with CUDA 11.1 and compiled
MinkowskiEngine/TorchSparse components. The method-local definition pins the
CUDA base-image digest and every Git dependency. It excludes two private
author-local packages that are not imported by the official inference route.
Extensions target CUDA architecture 8.0 and qualification therefore uses an
A100; CUDA 11.1 predates native L40S/Ada support. The resolved SIF digest and
package inventory must be retained before inference.

## Slurm Workflow

The committed Slurm chain covers a guarded CPU image build followed by
asset/environment qualification on an A100. Job names use the `forai_` prefix
and submission refuses colliding evidence roots. The development-only sidecar,
normalisation and evaluation components are ready, but an inference submission
job is intentionally withheld until the container and complete checkpoint load
have been verified.

Full-development, fine-tuning and test submission routes remain deliberately
blocked until the preceding evidence exists. In particular, no held-out job
can be submitted from the current scaffold.

## Evaluation Route

`scripts/runtime/normalise_forainet_predictions.py` reconstructs official
post-merge predictions by source-row identifier and writes the harmonised
array. `scripts/evaluation/evaluate_for_instance.py` applies the shared
pointwise protocol and writes per-plot metrics, matches, unmatched predictions
and unmatched references.

## Known Limitations

- No complete official ForAINetV2 release was located.
- The checkpoint provider publishes no checksum or immutable source tag.
- The exact official 42/14 train/validation membership is not released.
- The root-mapped fakeroot extraction probe passed, but the full legacy image
  build and package-import report are still pending.
- Current upstream source/checkpoint compatibility still needs a clean Barkla
  import and full state-dict load report.
- Official conversion, tiling, merging and label-independence still require a
  retained development smoke and manual alignment confirmation.
- No runtime or memory estimate is evidence-backed yet.

## Current Benchmark Status

Status: `fakeroot_probe_passed; pinned_image_build_not_run`.

The method is not eligible for the held-out ranking. The next gate is a clean
environment/checkpoint-load test followed by the frozen
`CULS/plot_1_annotated.las` development smoke. No shared repository files are
modified by this branch.
