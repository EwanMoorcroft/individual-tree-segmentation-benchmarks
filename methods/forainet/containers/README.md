# ForAINet Apptainer image

The definition is a qualification image for the official ForAINet source at
commit `5fe600ae8f2fe913ae8740f475f0261a702f2a72`. It derives from the upstream
Dockerfile blob `1daea67cfae9e44a0de439f06896320d9723c209`, but replaces mutable Git
references and the mutable CUDA tag with immutable revisions.

The pinned base is the Linux/amd64 manifest for
`nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04`, digest
`sha256:83e4b2841034cdf45ea5b9a5b472eb2c07b1b23d4836d32666a881db29a8dceb`.
MinkowskiEngine 0.5.4, TorchSparse 1.4.0 and hdbscan 0.8.28 are installed from
the exact commits recorded in the definition. The private author-local
`pylidar` and `rios` requirement paths are intentionally excluded because the
selected official inference and checkpoint-load path does not import them.

CUDA extensions are compiled only for compute capability 8.0. Qualification
therefore targets an A100, not an L40S. CUDA 11.1 predates native Ada (8.9)
support; using an A100 avoids making the published-checkpoint comparison depend
on forward-compatibility behaviour.

The definition must be built from the benchmark repository root because its
`%files` section addresses the method-local lock file. Use the guarded Slurm
submission wrapper rather than invoking `apptainer build` manually. The image,
cache, build evidence and package inventory remain outside Git.

The system and Python package inputs follow the upstream recipe and pin direct
package versions, but they are not a wheel-by-wheel hash lock and Ubuntu's
package index is external. The first passing SIF is therefore frozen by its
SHA-256 plus the retained definition, lock file, image metadata and complete
`pip freeze`. A later rebuild is not substituted without requalification.

The Barkla root-mapped fakeroot probe completed on 2026-07-23 with Apptainer
1.3.6. It warned that the account has no `/etc/subuid` entry and no `fakeroot`
helper, but successfully extracted an OCI image and created a SIF. The full
legacy package build remains a separate evidence gate until it completes.
