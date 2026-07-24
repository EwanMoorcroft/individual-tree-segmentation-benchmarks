# ForestFormer3D failures and recoveries

## 2026-07-23 development array task-parent omission

- Array job `9895688` created the immutable run root and logs but not the
  shared `tasks/` parent.
- All 21 elements failed at their first per-plot `mkdir` in one or two seconds,
  before opening a development LAS, starting Apptainer or running inference.
- Dependent summary job `9895689` was cancelled.
- Recovery creates `tasks/` before `sbatch`, retains the failed run unchanged
  and submits a new immutable run root.
- A structural regression test requires the parent creation in the submitter.

## 2026-07-23 development raw-work-directory collision

- Recovery attempt `9895714` successfully prepared development inputs, then
  failed because the adapter pre-created and bind-mounted `raw/` while the
  guarded official runner requires its work directory not to exist.
- No official inference began. Tasks 0 through 8 failed at the identical
  guard; task 9 and the remaining elements were cancelled, along with summary
  job `9895715`.
- Recovery bind-mounts the existing task root and passes its absent `raw`
  child as the runner work directory. The runner therefore creates the child
  atomically while outputs still land in the immutable task root.

## 2026-07-23 rootful Apptainer build

- Build job: `9894850`
- State: `FAILED`, exit code `255:0`
- First causal error: Apptainer reported no `/etc/subuid` mapping and no
  `fakeroot` command. During `apt-get update`, `_apt` UID/GID transitions failed
  with `Operation not permitted`.
- Dependent validation job `9894851` was cancelled without running.
- No partial SIF remained.
- Recovery: retain the verified base SIF and build a user-owned Conda
  toolchain plus rootless virtual environment. Do not retry the rootful recipe
  unchanged.

## 2026-07-23 rootless capability probe

- Probe job: `9894987`
- State: `COMPLETED`, exit code `0:0`
- Confirmed base: Python 3.10.8, pip 22.3.1, Conda 22.11.1, PyTorch 1.13.1,
  CUDA 11.6, `nvcc` 11.6.124 and CMake 3.22.1.
- Confirmed a user-writable virtual environment can inherit the base PyTorch
  installation and that compute-node HTTPS access works.
- Missing base tools: Git and Ninja. These are supplied through the external
  toolchain prefix.

## 2026-07-23 rootless build-path qualification

- The initial external-environment attempt did not enter the container because
  the host path was mounted onto `/environment`, which Apptainer reserves as a
  symlink to `.singularity.d/env/90-environment.sh`.
- Bind probe `9895092` proved both the logical and canonical fastscratch paths
  are valid directory sources when mounted onto an ordinary destination.
- The stable internal prefix is `/ff3d_environment`.
- The next attempt entered the builder and reached Conda, then failed because
  Conda selected the non-writable default cache `$HOME/.conda/pkgs`.
- Recovery redirects all dependency and compiler caches under the timestamped
  environment root. Home-directory ownership is not changed.
- The following attempt successfully resolved and installed the 2.9 GB
  toolchain, then stopped during activation because Conda's compiler activation
  script references `ADDR2LINE` before assignment and is incompatible with
  Bash nounset mode.
- Recovery suspends nounset only for the Conda hook and activation statements,
  then restores it before source retrieval or dependency installation.
- The next build passed activation, created the rootless virtual environment,
  fetched all five exact source revisions, and installed MMEngine,
  MMDetection, MMDetection3D, MMSegmentation and the CUDA 11.6 MMCV wheel.
- MinkowskiEngine then failed during CUDA compilation because `cblas.h` was
  absent from the compiler include search path. OpenBLAS itself was present in
  the external toolchain.
- Recovery uses the pinned MinkowskiEngine setup interface
  `--blas_include_dirs` and `--blas_library_dirs`, validates the header and
  shared library first, and exports matching compiler search paths.
- The corrected build compiled and installed the CUDA MinkowskiEngine extension
  and built `torch-scatter` 2.0.9. Segmentator configuration then stopped
  because `torch.utils.cmake_prefix_path` was called as a function even though
  PyTorch 1.13.1 exposes it as a string property.
- Recovery passes that property directly to CMake. This changes only the build
  interface expression; it does not modify Segmentator or model source.
- Build job `9895175` then completed the full rootless environment in 38 minutes
  30 seconds and wrote the pinned pip and Conda manifests. Its A100 validation
  job `9895176` reached module imports, then failed when Apptainer's `--nv`
  injection placed the Rocky Linux host `libGLX.so.0` ahead of the Ubuntu 18
  container libraries. That host library requires GLIBC 2.34, which the
  qualified base image intentionally does not provide.
- Recovery preserves the official `opencv-python` package and all built
  artifacts. The validation command reorders the already-bound library
  directories inside the container so the compatible external-toolchain GLVND
  library is selected first while `/.singularity.d/libs` remains available for
  the host NVIDIA driver. A validation-only retry must establish both successful
  OpenCV import and CUDA visibility before this recovery is accepted.
- Validation-only job `9895299` proved that path order alone is insufficient:
  Conda's `libglvnd` package contains `libGLdispatch` but not the split
  `libGL.so.1` or `libGLX.so.0` libraries. The cached, same-channel solve
  provides both as `libgl=1.7.0` and `libglx=1.7.0`. Recovery pins those
  packages in the rootless toolchain and validates both sonames before source
  retrieval. Because the completed environment is immutable, this correction
  requires a fresh timestamped environment build rather than an in-place
  package amendment.
- Fresh build job `9895305` passed the GL soname gate and completed the
  rootless environment in 35 minutes 50 seconds. Dependent A100 job `9895306`
  completed in 36 seconds with all required imports, exact source and
  replacement hashes, official checkpoint mapping load, CUDA availability and
  compute capability 8.0. The recovery is accepted.

## Recorded observations

- A foreground `srun` GPU probe was cancelled before allocation. It read no
  data and produced no runtime evidence. A detached replacement probe completed.
- The official inference entrypoint appears to use `torch.load` without an
  explicit `torch` import.
- Whole-plot inference is selected using the substring `test` in the input
  path. Development staging must not conceal this behaviour.
- Official inference reads reference arrays for output bookkeeping. Static
  inspection is insufficient to prove prediction independence.
- The authors' faster inference repository changes iteration count,
  configuration and post-processing and is not the unchanged default route.

## 2026-07-23 first official-runner smoke

- Development smoke job `9895488` failed after 32 seconds during the first
  reference-label inference. Preprocessing completed and accessed only
  `CULS/plot_1_annotated.las`; no prediction PLY was produced.
- The official runner reported shape mismatch for all 49 sparse-convolution
  weights, then failed in the decoder. The published archive contains only
  `epoch_3000_fix.pth`, already in the RSKC layout accepted by the qualified
  spconv loader, while pinned `tools/test.py` applies the same fix
  unconditionally.
- A direct runtime probe established that the archived `input_conv` weight
  loads successfully into the qualified spconv layer. The recovery accepts only
  the exact published checkpoint hash, applies the inverse permutation before
  invoking the unchanged official entrypoint, and verifies that the
  entrypoint's own permutation reconstructs every archived tensor exactly.
- The failed run root and logs remain retained. Any retry must use a fresh
  timestamped run root.
- Recovery run `9895601` proved that all checkpoint keys load without mismatch
  and reached the first real inference region. It failed after 29 seconds
  because strict deterministic-algorithm mode rejects upstream CUDA
  `index_reduce_(reduce='amax')`, which PyTorch 1.13.1 marks as lacking a
  deterministic implementation.
- The next retry kept the identical seed in both fresh processes but disabled
  strict deterministic-algorithm enforcement.
- Recovery run `9895608` loaded the checkpoint cleanly, processed all 64
  whole-plot regions and wrote a 1,816,672-row official PLY. It then failed in
  upstream `UnifiedSegMetric`, which indexes full-plot ground truth using the
  last-region in-memory prediction and raised `IndexError`. The later effective
  duplicate `predict` method also writes `<name>.ply`, not the
  `<name>_final_results.ply` path used by the earlier overridden definition.
- Recovery uses the effective filename and replaces only the incompatible
  post-inference metric with a registered no-op. The complete official PLY is
  validated independently for fields and row identity.
- Jobs `9895627` and `9895638` each completed reference and dummy whole-plot
  PLYs, then failed the original bitwise counterfactual gate. Same-label repeat
  differences were larger than cross-label differences, demonstrating that the
  pinned CUDA reduction remains nondeterministic despite identical seeds.
- A reversed-order diagnostic, job `9895654`, completed dummy first and
  reference second. Four clustered semantic rows that had appeared stable by
  label across the first two jobs disappeared under reversal. Across all
  comparisons XYZ remained exact, and no stable instance-label effect was
  observed.
- Recovery does not weaken this into an output-tolerance threshold. It requires
  exact reference/dummy fingerprints of the tensor entering `predict`, pins and
  AST-audits the effective prediction source to prove that ground truth is not
  consumed by prediction, verifies the deliberately different loader labels,
  and records every raw prediction difference as diagnostic evidence.

## Recovery rules

Failed environment builds retain their run root, incomplete environment and
logs. A retry uses a new run ID and environment path; existing evidence is not
overwritten or deleted by a reusable script. Dependency failure prevents
validation submission from running.

Model-source defects are not repaired silently. If official interfaces cannot
produce row-aligned predictions without changing model logic, work stops with
the exact upstream block.

## Independent development-retention verification

- Verification job `9912217` failed in two seconds before reading any retained
  artefact. Direct execution of the new verifier did not place the repository
  root on `sys.path`, so `shared.for_instance_manifest` could not be imported.
- Commit `2a2b8cf` added the same explicit repository-root import used by the
  other method scripts. The failed verification root and scheduler logs remain
  unchanged as provenance.
- Immutable replacement job `9912219` completed in 40 seconds. It re-hashed all
  294 retained artefacts (16,565,931,840 bytes), loaded all 21 harmonised
  predictions, verified exact source-row identity, reconciled the aggregate
  counts and confirmed `held_out_access=false`.
