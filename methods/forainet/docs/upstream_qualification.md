# Upstream and checkpoint qualification

Retrieved 2026-07-22.

## Selection

The official ForestFormer3D paper names `ForAINetV2_R8` and
`ForAINetV2_R16` as ForAINet baselines retrained on FOR-instanceV2. The
authors' repository and Zenodo release do not provide a distinct ForAINetV2
model implementation, either checkpoint, an exact configuration bound to a
checkpoint, or a checkpoint-initialised training route. The
`data/ForAINetV2` directory in ForestFormer3D is dataset support rather than a
method release.

The integration therefore selects the original official ForAINet release and
the method slug `forainet`. It does not use or download FOR-instanceV2 or V3
point-cloud data.

## Source and licence

- Repository: <https://github.com/prs-eth/ForAINet>
- Commit: `5fe600ae8f2fe913ae8740f475f0261a702f2a72`
- Branch at retrieval: `main`
- Repository licence: BSD 3-Clause
- Release/tag binding: none
- Paper: <https://doi.org/10.1016/j.rse.2024.114078>

Components adapted from torch-points3d and other dependencies retain their
own licences. The upstream source is executed externally and is not vendored.

## Checkpoint

- Provider: official ForAINet repository README
- Filename: `PointGroup-PAPER.pt`
- Provider URL:
  <https://www.dropbox.com/scl/fi/mv4nxe60cco86fd2u9f3z/PointGroup-PAPER.pt?rlkey=ua6093kehk0youpo8g3a6g0nm&st=wiqv3a0u&dl=0>
- Size: 665,805,463 bytes
- Provider checksum: not published
- Locally computed SHA-256:
  `97c03ce81621dc4193e55d2ca2294861b1f4421c94d192799e5fe031f9d35861`
- Locally computed MD5: `2f8f0622bdce0c15779a811003140d26`

The checkpoint archive contains a `PointGroup-PAPER` run configuration named
`setting1_classes5_mixtree2`, 149 completed epochs, radius 8 m, grid size
0.2 m, four input features, five semantic classes, stuff classes 0 and 1,
thing classes 2, 3 and 4, and a `latest` model state. This distinguishes it
from a generic PointGroup or SegmentAnyTree checkpoint.

## Official configuration and routes

The associated source configuration uses:

- `treeins_set1.TreeinsFusedDataset`;
- five-class setting 1;
- TreeMix training;
- 8 m cylinders;
- 0.2 m first/grid subsampling;
- full-resolution tracking with `SaveOriginalPosId`;
- one voting run; and
- the official PointGroup three-head architecture and cluster settings.

Inference uses the upstream Hydra `eval.py` entrypoint, an explicit
`data.fold`, `checkpoint_dir`, `model_name=PointGroup-PAPER` and
`weight_name=latest`.

Checkpoint-initialised fine-tuning uses the upstream `train.py` route with an
empty `training.checkpoint_dir`, the same model/data configuration and
`models.PointGroup-PAPER.path_pretrained` pointing to the frozen official
checkpoint. The loader accepts same-shaped parameters with `strict=False`;
therefore Barkla qualification must record all loaded, missing and unexpected
keys and must reject an incomplete load.

## Remaining qualification gates

The checkpoint and source were independently acquired on Barkla and passed the
frozen size, hash and commit checks. A minimal Apptainer 1.3.6 root-mapped
fakeroot extraction build also succeeded on 2026-07-23. The first full build
then failed when `apt` attempted to change identity inside that single-UID
namespace. The account has no `/etc/subuid` entry and no system fakeroot helper.
Barkla does provide the prerequisites for the official relocatable
unprivileged Apptainer installer and a large node-local ext filesystem. A
pinned user-local Apptainer 1.3.6 plus fakeroot `apt` probe is therefore the
next gate. The official installer's automatic Koji lookup did not locate a
1.3.6 EL8 package, so the retry uses the checksum-verified RPM published on the
official v1.3.6 GitHub release. The failed build and installer evidence remain
retained.

The committed definition derives from official Dockerfile blob
`1daea67cfae9e44a0de439f06896320d9723c209`. It pins the CUDA base image and
MinkowskiEngine, TorchSparse and hdbscan source commits. It targets an A100
(compute capability 8.0) because the official CUDA 11.1 toolchain predates
native L40S/Ada support.

1. Install and verify the pinned user-local build toolchain.
2. Build the pinned image and record its SHA-256 and package inventory.
3. Load the checkpoint against the selected architecture and prove complete
   parameter compatibility.
4. Prove prediction invariance when bookkeeping reference labels are changed.
5. Prove that official full-resolution output maps back to retained source rows
   without coordinate matching.
