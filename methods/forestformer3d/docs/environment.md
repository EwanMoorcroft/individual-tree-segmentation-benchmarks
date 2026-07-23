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

The build script is hashed before submission. The resolved Conda explicit
specification, `pip freeze`, source commits, base SIF hash and validation JSON
are retained. The first successful resolution remains a qualification artefact;
its explicit Conda specification must be promoted to a frozen input before any
benchmark prediction run.

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
queue time is reported separately from runtime. A dependency with scheduler
reason `DependencyNeverSatisfied` is terminal for monitoring purposes.

## Remaining gates

Environment validation does not prove the official test runner is scientifically
admissible. A later development-only gate must establish official checkpoint
loading through the runner, dummy-label independence, deterministic output and
exact full-row correspondence.
