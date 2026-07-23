# ForAINet

## Method Summary

This directory integrates the original ForAINet release as the reproducible
fallback for the unavailable ForAINetV2 release. The selected method slug is
`forainet`. The integration is development-only until every readiness gate in
the FOR-instance method-adapter protocol has passed. The environment, image
and complete checkpoint-load gates have passed. The guarded one-plot
development smoke and manual XY/XZ alignment review are accepted.

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
a resume route. The fixed fine-tuning plan uses the official exclusive epoch
limit 150, 3,000 samples per epoch, batch size four, FP32 Adam optimiser and
augmentation pipeline. The pinned trainer starts a new run at epoch label one,
so `range(1, 150)` performs 149 effective epochs and ends at label 149. It also
hard-codes training RNG seed 2022; seed 42 is used separately by the official
`random.sample` train/validation split. The predeclared candidate epochs are
30, 60, 90, 120 and 149, selected only by canonical five-plot validation
micro-F1, then lower false positives and earlier epoch. No training or held-out
inference has been run by this branch.

## Input Requirements

The source of truth is the original 32-file FOR-instance LAS catalogue. Every
input must be a repository-catalogued relative path with `classification` and
`treeID` fields. Alignment sidecars retain the exact integer source row,
source hash, point count, semantic values and positive reference-tree count.

The official five-class preparation maps original classes 1, 2, 4, 5 and 6 to
model classes 0 through 4 and drops class 3 outliers for labelled training and
evaluation. Benchmark inference cannot use that reference-label-dependent
filter: all source rows are retained, and the loader receives constant
low-vegetation and unassigned-instance bookkeeping values. Real
`classification` and `treeID` values remain only in the evaluation sidecar.
Development and held-out roots are physically separate.

## Output Contract

The primary retained artefact is a compressed, source-row-aligned array with:

- `pred_tree_id`;
- `target_tree_id`;
- `classification`;
- `pred_classification`; and
- `source_row_index`.

Raw official outputs remain separate. The adapter accepts only stable integer
row identifiers. It rejects missing, duplicated, out-of-range or conflicting
rows and never falls back to rounded-coordinate matching. Official
non-negative, zero-based instance IDs are shifted by one so that the first
official cluster remains a valid positive benchmark tree ID; official `-1`
remains unassigned.

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

The committed Slurm chain covers a guarded user-local Apptainer/fakeroot probe,
a CPU image build, asset/environment qualification on an A100, a guarded
one-plot development smoke, and the 21-plot published-checkpoint development
diagnostic. Job names use the `forai_` prefix and submission refuses colliding
evidence roots. The runtime uses the official 50 m tiler with 5 m overlap,
`eval.py`, and official merger; it then enforces exact source-row normalisation
and shared-protocol evaluation.

The saved checkpoint configuration may contain a non-portable data-root field.
Each run stages a derivative archive that changes only this path to the
run-local relative data root and verifies every model tensor exactly against the
unchanged official archive. Both hashes and the exact metadata-only change are
retained; no weight is altered.

The label-independence probe changes both artificial semantic and instance
bookkeeping fields. The pinned upstream reporter can divide by zero after
writing complete predictions for those deliberately nonreference labels. Only
that exact post-output traceback is accepted; both official prediction arrays
must still exist and match the primary arrays exactly. Its provenance records
separate hashes for the complete PLY files and for canonicalised `preds`
arrays, because the deliberately changed bookkeeping labels make the complete
files differ even when predictions are identical.

Fine-tuning preparation remains blocked on the completed 21-plot published
development final gate. Its adapter reproduces the pinned official Setting-1
mapping: class 3 and tree-class points carrying a stuff-instance ID are
removed, and classes 0, 1, 2, 4, 5 and 6 map to PLY labels 0, 1, 2, 3, 4 and 5.
Only the 21 hash-frozen development sources can be converted; the fixed seed-42
split yields 16 training and five `_val.ply` files. Test submission remains
separately blocked and no held-out job can be submitted from the current
scaffold.

## Evaluation Route

`scripts/runtime/normalise_forainet_predictions.py` reconstructs official
post-merge predictions by source-row identifier and writes the harmonised
array. `scripts/evaluation/evaluate_for_instance.py` applies the shared
pointwise protocol and writes per-plot metrics, matches, unmatched predictions
and unmatched references.

`scripts/provenance/build_alignment_review.py` creates a deterministic,
local-only XY/XZ comparison of reference and predicted instances from the
development smoke. It validates exact source-row order, removes global
coordinate offsets from the figure and writes a hash-linked JSON review
record outside the immutable run root. The figure and raw coordinates are not
committed. The accepted smoke record under `examples/` closes this gate for
the full-development route.

## Known Limitations

- No complete official ForAINetV2 release was located.
- The checkpoint provider publishes no checksum or immutable source tag.
- The exact official 42/14 train/validation membership is not released.
- Root-mapped `apt` is unavailable, so rebuilding depends on the qualified
  pinned user-local toolchain.
- The accepted development smoke took 458 seconds on one A100 and recorded
  approximately 9.3 GB peak resident memory. Full-development estimates must
  additionally use the frozen 21-plot point inventory.

## Current Benchmark Status

Status: `development_smoke_and_manual_alignment_accepted;
full_development_published_checkpoint_ready`.

The method is not eligible for the held-out ranking. The next gate is the
guarded 21-plot published-checkpoint development diagnostic. No shared
repository files are modified by this branch.
