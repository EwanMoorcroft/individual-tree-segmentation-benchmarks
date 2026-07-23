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

## Environment and checkpoint qualification

The checkpoint and source were independently acquired on Barkla and passed the
frozen size, hash and commit checks. The pinned relocatable Apptainer 1.3.6
toolchain then passed its fakeroot package-install probe. The completed A100
image has SHA-256
`ad0df684209014c52421dc213cd0e15ddbb47214c00fac264e829f68dc17812d`.
It pins NumPy 1.23.5 because the official evaluation code uses the legacy
`np.int` API removed in NumPy 1.24.

The official checkpoint was loaded on an A100 80 GB with Python 3.8.10,
NumPy 1.23.5, PyTorch 1.9.0+cu111, MinkowskiEngine 0.5.4, torch-geometric 1.7.2,
TorchSparse 1.4.0 and hdbscan 0.8.28. All 755 checkpoint tensors matched all
755 model tensors; there were no missing, unexpected or shape-mismatched keys.
The retained load report has SHA-256
`693c9e69f99d943a74ecf7318c5eb1c3968a437616fcc909fde5130579c93b1c`.

The committed definition derives from official Dockerfile blob
`1daea67cfae9e44a0de439f06896320d9723c209`. It pins the CUDA base image and
MinkowskiEngine, TorchSparse and hdbscan source commits. It targets an A100
(compute capability 8.0) because the official CUDA 11.1 toolchain predates
native L40S/Ada support.

The remaining development-smoke gates are:

1. prove prediction independence from reference-label bookkeeping;
2. prove that the official tiler and merger return exactly one prediction for
   every original source row without coordinate matching; and
3. complete manual alignment review before full-development inference.
