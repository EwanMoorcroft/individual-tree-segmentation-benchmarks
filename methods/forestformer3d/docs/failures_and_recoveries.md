# ForestFormer3D failures and recoveries

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

## Recovery rules

Failed environment builds retain their run root, incomplete environment and
logs. A retry uses a new run ID and environment path; existing evidence is not
overwritten or deleted by a reusable script. Dependency failure prevents
validation submission from running.

Model-source defects are not repaired silently. If official interfaces cannot
produce row-aligned predictions without changing model logic, work stops with
the exact upstream block.
