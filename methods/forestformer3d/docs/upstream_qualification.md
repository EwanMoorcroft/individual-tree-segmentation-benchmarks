# ForestFormer3D upstream qualification

## Pinned identities

- official repository: `https://github.com/SmartForest-no/ForestFormer3D`
- source commit: `6a75c3735e4a4108d02ee944a8b93177f2360a4f`
- official checkpoint record: `https://doi.org/10.5281/zenodo.16742708`
- checkpoint archive: `clean_forestformer.zip`
- supplied and verified MD5:
  `553d67379331966509076f3fbb409e57`
- locally verified archive SHA-256:
  `916ad481d2136a840f08a0aea983ece78866e8b659eb7e3b1027aefa21396ce0`
- extracted checkpoint: `epoch_3000_fix.pth`, 230,154,197 bytes
- checkpoint SHA-256:
  `01037a648596832238ac72ea2f5eef87ceaf5aeb399e56ff4b760ba1ed1c777e`

The Barkla source checkout was clean at the pinned commit. The original archive
and extracted checkpoint are retained outside Git.

## Official interfaces

The official training entrypoint is `tools/train.py`. The official inference
entrypoint is `tools/test.py`; the released checkpoint is paired with
`configs/oneformer3d_qs_radius16_qp300_2many.py`. The official README instructs
users to install three bundled replacement files into MMEngine and
MMDetection3D. The image recipe follows that documented installation and
verifies each replacement file before copying it.

The full image must still prove that the checkpoint loads through the official
framework for inference and can initialise the official training runner.
Loading the file with `torch.load` is necessary but not sufficient.

## Environment evidence

The official Docker base resolves to
`pytorch/pytorch@sha256:58d848c38665fd3ed20bee65918255cb083637c860eb4fae67face2fb2ff5702`.
A converted base SIF was verified with SHA-256
`4a35d5a57c1d57061f899b514329ad8ec2bf74a9ff31d103c0a53a289e07c84f`.
On Barkla it exposed PyTorch 1.13.1, CUDA 11.6 and cuDNN 8.4.0.27 to an
A100-SXM4-80GB at compute capability 8.0.

## Licensing

The code repository states CC BY-NC 4.0. The Zenodo record labels its files
GPL-3.0-or-later. This branch records both primary-source statements and does
not interpret the discrepancy. The work is non-commercial academic research.
No third-party source or binary artifact is redistributed through Git.

## Qualification status

The upstream identity and checkpoint byte identity are qualified. Final
upstream qualification remains conditional on full-image imports, official
runner checkpoint loading, prediction-label independence and source-row output
alignment without modifying upstream modelling code.
