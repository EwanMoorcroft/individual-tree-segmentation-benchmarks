# ForAINet Apptainer image

The definition is a qualification image for the official ForAINet source at
commit `5fe600ae8f2fe913ae8740f475f0261a702f2a72`. It derives from the upstream
Dockerfile blob `1daea67cfae9e44a0de439f06896320d9723c209`, but replaces mutable Git
references and the mutable CUDA tag with immutable revisions.

The pinned base is the Linux/amd64 manifest for
`nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04`, addressed in the definition as
digest-only `nvidia/cuda@sha256:...` for Apptainer 1.3.6 compatibility. Its
digest is
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
1.3.6. It successfully extracted an OCI image and created a SIF, but the first
full build proved that `apt` cannot change to its `_apt` user without a
subordinate-ID mapping or fakeroot helper. Fast-scratch also lacks the xattr
support used by the rootless extractor.

The guarded toolchain job downloads the official Apptainer v1.3.6
`install-unprivileged.sh`, verifies SHA-256
`41574717e85e03cdf40597819c927250d0772186b943b8869c8ec8dfcb5b86d1`, and
installs its EL8 compatibility bundle outside Git. The installer's automatic
Koji lookup no longer locates the 1.3.6 EL8 package, so the job instead downloads
the official GitHub release RPM and verifies its published SHA-256
`1890dd3df87b06b0a9b2845b81b5709c0033fcca5673b03cc69ce9cb755e9605` before
passing that local RPM to the official installer. This supplies the
old-glibc-compatible fakeroot helper recommended for older Ubuntu containers.
It must install a package into the exact digest-pinned CUDA base before the
ForAINet image can be retried. Both the toolchain probe and full image build use
node-local `/tmp`; the final SIF, cache and evidence remain in persistent
external storage.
