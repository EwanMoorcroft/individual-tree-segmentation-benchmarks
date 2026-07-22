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

1. Acquire the checkpoint again into an immutable Barkla asset root and verify
   its byte size and SHA-256.
2. Build or obtain a pinned image and record its SHA-256 and package inventory.
3. Confirm the pinned source is clean and at the expected commit.
4. Load the checkpoint against the selected architecture and prove complete
   parameter compatibility.
5. Prove prediction invariance when bookkeeping reference labels are changed.
6. Prove that official full-resolution output maps back to retained source rows
   without coordinate matching.
