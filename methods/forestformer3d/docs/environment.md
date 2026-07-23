# ForestFormer3D environment

## Reproduction route

Barkla cannot execute the rootful package-installation steps in
`container/forestformer3d.def`: the user has no `/etc/subuid` mapping and the
cluster's Apptainer module has no `fakeroot` helper. The first build therefore
failed before any Python dependency was installed. The definition remains the
auditable translation of the official Dockerfile, but it is not the Barkla
execution route.

The Barkla route is a composite environment:

1. the Docker base converted to a SIF and verified by SHA-256;
2. a user-owned Conda toolchain prefix for GCC 9, Git, Ninja, OpenBLAS and
   graphical runtime libraries;
3. a rootless virtual environment inheriting PyTorch 1.13.1/CUDA 11.6 from the
   SIF;
4. exact upstream source checkouts; and
5. the authors' three replacement files copied into the virtual environment.

The selected host environment directory is always mounted at
`/ff3d_environment` inside Apptainer. Conda, virtual-environment and
compiled-extension prefixes therefore remain stable and do not embed a
username or timestamped host path. `/environment` is not used because
Apptainer reserves it as a symlink to its generated environment script.
Conda, pip, CUDA, PyTorch-extension and plotting caches are redirected beneath
the same timestamped environment directory; the base image's default
`$HOME/.conda` cache is not writable on Barkla.

The rootless builder, exact Conda lock, CPU job and GPU-validation job are
hashed together before submission. The resolved Conda explicit specification,
`pip freeze`, source commits, base SIF hash and validation JSON are retained.
The first successful Conda resolution is checked in at
`locks/conda_toolchain_explicit_linux-64.txt`; its SHA-256 is
`2ed0298fc8dbeae38dc0d431c614647af5acde80b2c878b1d12858042c850f71`.
Previously unpinned direct pip requirements are fixed to the versions observed
in that successful environment.

The only interactive-command normalization is `pip uninstall -y
torch-cluster`; `-y` prevents a batch build prompt and does not alter the
selected package. Upstream modelling source is not edited.

## Build and validation chain

`slurm/submit_environment_build.sh`:

1. validates a clean `method/forestformer3d` checkout;
2. hashes the recipe and official checkpoint;
3. refuses colliding environment, run or state paths;
4. builds the external environment on a CPU node;
5. validates the composite runtime on one A100;
6. imports the sparse libraries and ForestFormer3D package;
7. verifies the retained source and installed replacement hashes;
8. loads the official checkpoint as a mapping;
9. records base-SIF inspection and SHA-256, Conda and pip resolutions, and a
   validation JSON; and
10. starts a 30-second live monitor automatically.

The environment is not accepted if any dependent validation job fails. Build
queue time is reported separately from runtime. If Barkla reports
`DependencyNeverSatisfied`, the monitor cancels only that workflow-owned
downstream job and treats it as terminal.

## Qualified environment

CPU build job `9895305` completed in 35 minutes 50 seconds and A100 validation
job `9895306` completed in 36 seconds. Validation proved:

- clean ForestFormer3D source at the pinned commit and exact source hashes;
- exact installed hashes for all three author replacement files;
- the official checkpoint SHA-256 and successful mapping load;
- imports of MMEngine, MMDetection, MMDetection3D, MMSegmentation,
  MinkowskiEngine, spconv, torch-points-kernels, torch-cluster and OneFormer3D;
- PyTorch 1.13.1 with CUDA 11.6; and
- an NVIDIA A100-SXM4-80GB at compute capability 8.0.

The retained pip manifest SHA-256 is
`b48648ead7fe20afad5af9cc1a2a272277dd34988d82a5a1159fd0ac78578456`;
the validation JSON SHA-256 is
`378aaa25faec5c81a5a900aa2ae41e2f0765fa7aa5d46ae921e369b3622bb517`.
MinkowskiEngine's unset `OMP_NUM_THREADS` warning and MMDetection3D's pinned
Numba deprecation warning are non-fatal; inference jobs must set their thread
count explicitly.

## Remaining gates

Environment validation does not prove the official test runner is scientifically
admissible. A later development-only gate must establish official checkpoint
loading through the runner, dummy-label independence, deterministic output and
exact full-row correspondence.
