# TLS2trees Published-Default Development Smoke

## Purpose And Boundary

This workflow runs one reproducibly selected FOR-instance development plot
through the published-default TLS2trees route. It validates the complete method
boundary: exact split metadata, label-stripped conversion, semantic inference,
instance segmentation, source-row alignment, both leaf targets, harmonised
evaluation, and resource capture.

The route cannot submit a held-out-test job, a tuning job, a full-development
array, or a different method variant. The submission wrapper accepts only
`variant=published_default` and `split=development`. It resolves Stage 0 index
zero from the frozen median-point-count selection instead of embedding a plot
path or dataset task index.

Do not use the legacy oracle-semantic pilot scripts for this smoke.

## Immutable Dependency Chain

The submission wrapper creates this chain:

1. inventory the exact 21-plot development split and freeze the five-site
   Stage 0 selection;
2. convert the selected plot into label-stripped 10 m tiles with 0.02 m voxel
   representatives and a source-row sidecar;
3. run the bundled FSCT semantic model on one GPU;
4. run the published-default TLS2trees instance and leaf-attachment stage;
5. project predictions to source rows and write separate aligned leaf-off and
   leaf-on artefacts;
6. evaluate leaf-off and leaf-on independently;
7. require both coordinate/evaluation gates to pass; and
8. write a compact run and target summary.

Every output is isolated by variant, split and UTC run ID. Existing run roots
are never overwritten. The benchmark checkout, upstream checkout, bundled
model, manifest and stage outputs are checked before use.

## Required Barkla State

The public workflow assumes these locations, all of which can be overridden by
the named environment variables:

| Item | Default | Override |
| --- | --- | --- |
| Benchmark Git checkout | `$HOME/scratch/tree-seg-benchmark` | `TLS2TREES_PROJECT_ROOT` |
| FOR-instance dataset | `$HOME/data/datasets/for_instance/FORinstance_dataset` | `TLS2TREES_DATASET_ROOT` |
| Upstream TLS2trees Git checkout | `<benchmark checkout>/external/TLS2trees` | `TLS2TREES_UPSTREAM_REPO` |
| Benchmark/evaluation venv | `$HOME/fastscratch/venvs/treebench` | `TLS2TREES_TREEBENCH_ENV` |
| TLS2trees semantic/instance venv | `$HOME/fastscratch/venvs/tls2trees` | `TLS2TREES_METHOD_ENV` |

The upstream checkout must be clean at commit
`ca12cb73b2c736d80b020e8025f8d975d42e6f01`. The bundled model SHA-256 must be
`1a8bb6372394600f7c4b15f76beb98c32cb47ed25f8f729a84117ccfa410e72b`.
The benchmark checkout must also be clean. Generated data, predictions,
metadata, tables and logs must remain ignored by Git.

The standard `treebench` environment intentionally contains benchmark utility
packages, not PyTorch. Do not add method packages to it and do not reuse the
TreeLearn or TreeX environments. TLS2trees uses a separate Python 3.9
compatibility environment matching the upstream PyTorch 1.9.0 CUDA 11.1 and
PyG 1.7.2 stack. This reproduces the historical runtime rather than changing a
method parameter.

Barkla provides CUDA 12 modules, whereas this historical PyTorch wheel links
against CUDA 11 libraries including `libcusparse.so.11`. The isolated prefix
therefore pins conda-forge `cudatoolkit=11.1.1=h6406543_8`, a Linux build
available before PyTorch 1.9 was released, and adds only that prefix's `lib`
directory to `LD_LIBRARY_PATH`. The validator also freezes the Conda record's
artifact MD5 (`4851e7f19b684e517dc8e6b5b375dda0`). This is a runtime
compatibility dependency, not a TLS2trees parameter. CUDA 12 libraries must
not be renamed or symlinked to satisfy the CUDA 11 ABI.

These CUDA 11.1 binaries predate the L40S GPU generation. L40S compatibility
is not assumed from successful imports: the setup job is a feasibility gate
whose compiled-operator and full-model checks must pass on the allocated GPU.
If it reports an unsupported architecture or `no kernel image`, stop and
review a newer runtime as an explicit compatibility change; do not silently
change packages or submit the benchmark.

## One-Time Method Environment Setup

The setup is a guarded GPU Slurm job. It installs into an isolated Conda
prefix, disables user-site packages, validates exact versions, exercises the
compiled CUDA `fps`, `radius`, `knn_interpolate` and `global_max_pool`
operators, loads the bundled FSCT state dict, runs a synthetic model forward
pass and only then writes the validated `.tls2trees_setup_complete.json`. The
marker includes the exact Conda CUDA package build and resolved runtime-library
evidence.

The wrapper records Python, NumPy, PyTorch and CUDA seeds, but does not enable
`torch.use_deterministic_algorithms(True)`. PyTorch 1.9 raises instead of
running the CUDA `scatter_add` used by the published PyG/TLS2trees pipeline
under that enforcement. Seeded best-effort execution is therefore the faithful
published-runtime policy. The setup gate runs the same synthetic model input
twice and records exact equality and maximum absolute output delta rather than
silently claiming bitwise determinism.

Pinned upstream instance segmentation removes graph-isolated stem origins but
does not check whether any remain before calling NetworkX. The compatibility
wrapper records an empty graph-source set as `completed_no_predictions`, with
one hashed reason file per semantic tile. It does not create a synthetic tree,
relax a stem rule or change a published parameter. The automated smoke gate
still refuses to authorise broader execution when either target emits zero
instances.

An audited retry after this exact failure uses
`resume_published_default_dev_smoke_from_instance.sh`. It requires the original
state file, a failed instance job, an empty raw-output directory and the
NetworkX `sources must not be empty` evidence. The retry archives the failed
metadata and tile logs, reuses the immutable conversion and semantic outputs,
and submits only instance and downstream jobs. It never silently retries a
different failure or overwrites the failed attempt.

The default Conda source is the canonical conda-forge channel at
`https://conda.anaconda.org/conda-forge`. If that hostname is unavailable but
the public conda-forge mirror at `https://prefix.dev/conda-forge` is reachable,
set `TLS2TREES_CONDA_CHANNEL` to the mirror URL. The setup accepts only these
two HTTPS sources, uses `--override-channels`, checks both required repodata
endpoints before creating a new prefix, and records the selected source in the
guarded sidecar next to the prefix and alongside the explicit Conda package
record. Writing the sidecar before prefix creation keeps the source provenance
recoverable if Conda leaves a partial prefix. A
partial-prefix resume must use the recorded source and refuses a conflicting
override. Changing the transport source does not change any TLS2trees method
parameter or package-version pin.

Run on the Barkla login node:

```bash
cd "$HOME/scratch/tree-seg-benchmark"
mkdir -p logs/tls2trees_for_instance

TLS2TREES_SETUP_CONFIRMED=1 \
TLS2TREES_METHOD_ENV="$HOME/fastscratch/venvs/tls2trees" \
  sbatch --parsable \
  methods/tls2trees/slurm/for_instance/setup_tls2trees_environment.sbatch
```

When the canonical conda-forge hostname fails its DNS check and both mirror
repodata URLs have returned HTTP 200, add the reviewed mirror override:

```bash
TLS2TREES_SETUP_CONFIRMED=1 \
TLS2TREES_METHOD_ENV="$HOME/fastscratch/venvs/tls2trees" \
TLS2TREES_CONDA_CHANNEL="https://prefix.dev/conda-forge" \
  sbatch --parsable \
  methods/tls2trees/slurm/for_instance/setup_tls2trees_environment.sbatch
```

The job requests `gpu-l40s-low`, eight CPUs, 32 GiB, one GPU and two hours.
It performs no dataset access and submits no benchmark work. If setup stops
after creating a partial prefix, inspect the failure first; resume the same
reviewed setup with `TLS2TREES_SETUP_RESUME_PARTIAL=1`. Never delete or
silently replace an existing prefix.

If the inspected log proves that every pinned package and CUDA runtime library
was installed and the failure occurred only in the final GPU validator, also
set `TLS2TREES_SETUP_VALIDATE_PARTIAL_ONLY=1`. This guarded recovery skips all
Conda and pip network operations, validates the existing prefix against the
exact package contract, runs the compiled PyG operations and bundled model on
the allocated GPU, and writes the completion marker only after every gate
passes. Do not use validation-only recovery for an interrupted installation.

Monitor with the printed job ID:

```bash
squeue -j <job_id>
sacct -X -j <job_id> \
  --format=JobID,JobName%30,State,Elapsed,TotalCPU,AllocCPUS,MaxRSS,ExitCode
tail -n 80 -f logs/tls2trees_for_instance/tls2trees_env_setup_<job_id>.out
```

Success requires `TLS2TREES_ENVIRONMENT_SETUP_VALIDATED`, compiled PyG
operations `passed`, bundled model forward `passed`, and a completion marker.
Any package-resolution, CUDA architecture, compiled-operation or model-forward
error is a hard stop before the benchmark smoke.

## Resource Requests

| Stage | Partition | CPU | Memory | GPU | Wall time | Reason |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| One-time environment setup | `gpu-l40s-low` | 8 | 32 GiB | 1 | 2 h | Pinned stack install and real CUDA/model validation |
| Inventory | `nodes` | 2 | 16 GiB | 0 | 2 h | Headers and immutable input hashes |
| Conversion | `nodes` | 4 | 64 GiB | 0 | 2 h | LAS geometry, voxel map and tile writes |
| Semantic | `gpu-l40s-low` | 10 | 64 GiB | 1 | 4 h | Upstream bundled FSCT inference |
| Instance | `nodes` | 4 | 96 GiB | 0 | 8 h | Graph construction and leaf attachment |
| Alignment adapter | `nodes` | 4 | 64 GiB | 0 | 2 h | Coordinate recovery and source-row assignment |
| Evaluation, each target | `nodes` | 4 | 64 GiB | 0 | 2 h | Reference load and pointwise matching |
| Gate and summary | `nodes` | 2 | 16 GiB | 0 | 30 min each | JSON/CSV validation and aggregation |

These are conservative smoke allocations, not final array requests. The 96 GiB
instance request reflects the earlier TLS2trees observation that one different
plot exceeded 32 GiB and used about 49.6 GiB. Record the smoke's elapsed time,
MaxRSS and method metadata before changing resources. The wrapper also requires
at least 50 GiB free under the project filesystem; override the byte threshold
with `TLS2TREES_SMOKE_MIN_FREE_BYTES` only after checking local storage policy.

## Preflight

Run on Barkla:

```bash
cd "$HOME/scratch/tree-seg-benchmark"

git status --short --branch
test -z "$(git status --porcelain)"

test "$(git -C external/TLS2trees rev-parse HEAD)" = \
  "ca12cb73b2c736d80b020e8025f8d975d42e6f01"
test -z "$(git -C external/TLS2trees status --porcelain)"
sha256sum external/TLS2trees/tls2trees/fsct/model/model.pth

test -f "$HOME/data/datasets/for_instance/FORinstance_dataset/data_split_metadata.csv"
test -x "$HOME/fastscratch/venvs/treebench/bin/python"
test -x "${TLS2TREES_METHOD_ENV:-$HOME/fastscratch/venvs/tls2trees}/bin/python"
test -f "${TLS2TREES_METHOD_ENV:-$HOME/fastscratch/venvs/tls2trees}/.tls2trees_setup_complete.json"
df -h "$HOME/scratch/tree-seg-benchmark"
```

Stop if either checkout is dirty, the upstream commit differs, the model hash
differs, the dataset metadata is absent, or the project filesystem has less
than 50 GiB free.

## Submit Exactly One Development Smoke

```bash
cd "$HOME/scratch/tree-seg-benchmark"
export TLS2TREES_METHOD_ENV="$HOME/fastscratch/venvs/tls2trees"
export TLS2TREES_DEV_SMOKE_CONFIRMED=1
bash methods/tls2trees/slurm/for_instance/submit_published_default_dev_smoke.sh
```

The command prints the run ID, seven stage job IDs plus the two target-specific
evaluation job IDs, and an absolute state-file path. Save that state-file path.
No production array is submitted.

## Monitor With Low Noise

Set `STATE_FILE` to the exact path printed by submission:

```bash
STATE_FILE="$HOME/fastscratch/tls2trees_for_instance_smoke_states/REPLACE_WITH_PRINTED_RUN_ID.env"
bash methods/tls2trees/slurm/for_instance/monitor_published_default_dev_smoke.sh \
  "$STATE_FILE"
```

The monitor shows only the chain's `squeue` and `sacct` rows, then the gate or
summary status. It does not log out of the shell. If a job fails, inspect only
its matching error file:

```bash
cd "$HOME/scratch/tree-seg-benchmark"
ls -1t logs/tls2trees_for_instance/*.err | head
```

Use the job ID from `sacct` to choose the relevant file. Do not resubmit into
the same run root.

## Success Criteria

The smoke is successful only when:

- every chain job is `COMPLETED` with exit code `0:0`;
- the manifest reports exactly 21 development plots and no held-out access;
- conversion metadata reports label stripping, no reference fields passed to
  TLS2trees, the published 10 m/0.02 m geometry settings and a valid source map;
- semantic and instance metadata report `completed`;
- the adapter writes aligned NPZ and alignment metadata for both targets;
- both target metrics report `safe_for_scoring=true`, source-row order complete,
  at least one predicted instance, and the correct target contract;
- the final gate reports `status=passed_automated_gates`; and
- `run_summary.json` reports `development_smoke_completed` and records semantic,
  instance and adapter resource evidence.

Expected run-level outputs are:

```text
data/predictions/tls2trees/for_instance/published_default/development/<run_id>/<safe_plot_id>/
results/metadata/tls2trees/for_instance/published_default/workflow/development/<run_id>/
results/tables/tls2trees/for_instance/published_default/workflow/development/<run_id>/
logs/tls2trees_for_instance/
```

Raw tree PLYs and aligned source-row predictions are retained under the
immutable plot root so alignment or metrics can be re-audited without rerunning
the expensive method stages.

## Failure Indicators And Stop Rule

Stop at this one-plot route if any of these occurs:

- split count/hash/path rejection or any held-out-test access;
- reference semantic or instance fields appear in method input;
- model, upstream commit, config or input checksum drift;
- CUDA, PyTorch Geometric or upstream import failure;
- missing semantic tile, leaf target or prediction metadata;
- out-of-memory, timeout or non-zero method return code;
- raw-frame recording, source-row uniqueness or multiplicity failure;
- a target is unsafe for scoring or emits no instances; or
- a run directory already exists.

After success, review the selected plot, alignment diagnostics, metrics,
runtime, MaxRSS and retained artefacts. This workflow deliberately stops at the
manual review gate. It does not authorise a five-site Stage 0 run, parameter
search, full-development evaluation or held-out-test evaluation.

## Separate Fixed Published-Default Held-Out Baseline

The development smoke above remains a bounded compatibility check. The
separate full-test route uses the frozen published-default parameters in
[`for_instance_published_default.yml`](../configs/for_instance_published_default.yml)
without selecting or changing them from FOR-instance accuracy metrics. It
evaluates the exact supplied 11-plot test split once, writes both leaf targets,
and retains 22 source-row-aligned prediction files. Valid empty predictions
are completed zero-valued metrics, not failed jobs.

Review and hash the committed published configuration, then submit from a
clean Barkla checkout:

```bash
cd "$HOME/scratch/tree-seg-benchmark"

PUBLISHED_CONFIG="methods/tls2trees/configs/for_instance_published_default.yml"
PUBLISHED_CONFIG_SHA256=$(sha256sum "$PUBLISHED_CONFIG" | awk '{print $1}')

TLS2TREES_PUBLISHED_DEFAULT_TEST_CONFIRMED=1 \
TLS2TREES_REVIEWED_PUBLISHED_DEFAULT_CONFIG_SHA256="$PUBLISHED_CONFIG_SHA256" \
  bash methods/tls2trees/slurm/for_instance/\
submit_published_default_held_out_test.sh
```

An existing completed development-tuned held-out state file may be supplied as
the final positional argument. Its semantic cache is reused only when the
manifest, source input, conversion, model, config and every semantic-output
hash match exactly. A mismatch automatically runs a dedicated GPU semantic
task; it never accepts a partial or approximate cache match.

Monitor the exact state pointer created by submission:

```bash
STATE_FILE="$(
  tr -d '\r\n' \
    < logs/tls2trees_for_instance/latest_published_default_test_state_file.txt
)"

bash methods/tls2trees/slurm/for_instance/\
monitor_published_default_held_out_test.sh \
  "$STATE_FILE"
```

Completion requires `22/22` valid metrics, no failed semantic or evaluation
tasks, `configuration_changed_after_test=false`, and a 22-file hash-verified
retention manifest. Run-scoped outputs are written beneath:

```text
data/predictions/tls2trees/for_instance/published_default/test/<run_id>/
results/metadata/tls2trees/for_instance/published_default/test/<run_id>/
results/tables/tls2trees/for_instance/published_default/test/<run_id>/
```

The summary JSON, plot CSV, target summary CSV and retention manifest are the
only inputs accepted by the guarded public-result finaliser. After the monitor
reports `published_default_test_completed`, export the public-safe evidence and
registry rows with the same reviewed configuration hash:

```bash
TLS2TREES_PUBLISHED_DEFAULT_RESULTS_CONFIRMED=1 \
TLS2TREES_REVIEWED_PUBLISHED_DEFAULT_CONFIG_SHA256="$PUBLISHED_CONFIG_SHA256" \
  bash methods/tls2trees/slurm/for_instance/\
finalise_published_default_results.sh \
  "$STATE_FILE"
```

The command prints the submitted job ID and stores it in
`logs/tls2trees_for_instance/latest_published_default_finalisation_job_id.txt`.
Use the exact job monitor and receipt check in the
[`Final publication order`](for_instance_benchmark.md#final-publication-order)
section; do not pull or edit the checkout while that job is pending.

Run this finaliser only after any earlier tracked-result publication has been
verified and committed. If its Slurm job is interrupted after writing only part
of the exact public bundle, rerun the same command with
`TLS2TREES_PUBLISHED_DEFAULT_RESULTS_RECOVERY_CONFIRMED=1`. Recovery validates a
narrow allowlist of finaliser-owned files and refuses staged, deleted, renamed
or unrelated worktree changes.

The finalisation job writes eight small public artifacts under
`methods/tls2trees/examples/` and strictly upserts one headline, one leaf-off
diagnostic and one retention-registry row. Raw point clouds, predictions, state
files, receipts and Barkla paths remain outside Git. The receipt verifies the
tracked bundle but is not committed. A changed headline also requires the
public workbook rebuild, four-sheet visual check and synchronization test
specified in the final-publication runbook before commit.
