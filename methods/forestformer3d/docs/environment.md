# ForestFormer3D environment

## Reproduction route

The environment is built as an immutable Apptainer SIF from
`container/forestformer3d.def`. The definition pins the Docker base by registry
digest and the authors' source by full commit. It reproduces the official
Dockerfile dependency commands and the README's three documented replacement
copies.

The only interactive-command normalization is `pip uninstall -y
torch-cluster`; `-y` prevents a batch build prompt and does not alter the
selected package. Upstream modelling source is not edited.

## Build and validation chain

`slurm/submit_environment_build.sh`:

1. validates a clean `method/forestformer3d` checkout;
2. hashes the recipe and official checkpoint;
3. refuses colliding image, run or state paths;
4. builds on a CPU node with fastscratch cache and temporary storage;
5. validates the resulting image on one A100;
6. imports the sparse libraries and ForestFormer3D package;
7. verifies the embedded source and installed replacement hashes;
8. loads the official checkpoint as a mapping;
9. records SIF inspection, SHA-256, `pip freeze` and a validation JSON; and
10. starts a 30-second live monitor automatically.

The image is not accepted if any dependent validation job fails. Build queue
time is reported separately from runtime.

## Remaining gates

Environment validation does not prove the official test runner is scientifically
admissible. A later development-only gate must establish official checkpoint
loading through the runner, dummy-label independence, deterministic output and
exact full-row correspondence.
