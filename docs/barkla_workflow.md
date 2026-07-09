# Barkla Access And Slurm Workflow

This runbook separates local access, login-node coordination,
visualisation-node inspection and Slurm execution. Login-node commands in this
document are limited to lightweight setup, repository synchronisation,
scheduler queries, job submission and bounded log inspection. Training,
large inference, large preprocessing and full evaluation must run through
Slurm.

## Command Locations

| Location label | Use for | Do not use for |
| --- | --- | --- |
| Run on Mac | SSH configuration, SSH connections and local file transfer commands | Remote training or evaluation |
| Run on Barkla login node | Lightweight repository, environment and scheduler coordination | Model training, large inference, full evaluation or large preprocessing |
| Run on Barkla visualisation node | Short interactive inspection of small outputs and plots | Long-running batch work |
| Run through Slurm | Training, large inference, large preprocessing, full evaluation and batch summaries | Manual interactive debugging |

## SSH MFA Minimisation

SSH multiplexing reuses one authenticated master connection. It reduces repeat
MFA requests during the configured persistence window; it does not bypass MFA.

Expected runtime: 1-2 minutes.
Expected memory: negligible.
Recommended partition: not applicable.
Output paths: `~/.ssh/config` on the Mac.
Success criteria: later `ssh barkla1` sessions reuse the existing connection.
Failure indicators: SSH reports config syntax errors, rejects permissions or
asks for MFA for every new session.

### Run on Mac

```bash
nano ~/.ssh/config
```

Add or update this SSH configuration:

### Run on Mac

```sshconfig
Host barkla1
    HostName <barkla_login1_host>
    User <barkla_username>
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 8h
    ServerAliveInterval 60
    ServerAliveCountMax 5
    ForwardAgent no

Host barkla2
    HostName <barkla_login2_host>
    User <barkla_username>
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 8h
    ServerAliveInterval 60
    ServerAliveCountMax 5
    ForwardAgent no

Host barklaviz1
    HostName <barkla_visualisation_host>
    User <barkla_username>
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 8h
    ServerAliveInterval 60
    ServerAliveCountMax 5
    ForwardAgent no
```

Expected runtime: under 1 minute.
Expected memory: negligible.
Recommended partition: not applicable.
Output paths: SSH control socket under `~/.ssh/`.
Success criteria: the first connection authenticates and later connections
open without a new MFA request while the master connection persists.
Failure indicators: `Bad owner or permissions`, no control socket appears, or
`ssh -O check barkla1` cannot find a master connection.

### Run on Mac

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/config
ssh barkla1
```

Expected runtime: under 1 minute after the first connection is active.
Expected memory: negligible.
Recommended partition: not applicable.
Output paths: SSH control socket under `~/.ssh/`.
Success criteria: the control master is listed or reported as running.
Failure indicators: no matching control socket exists.

### Run on Mac

```bash
ls ~/.ssh/cm-*
ssh -O check barkla1
```

Expected runtime: under 1 minute.
Expected memory: negligible.
Recommended partition: not applicable.
Output paths: none.
Success criteria: SSH exits the master connection cleanly.
Failure indicators: SSH reports that no control master is running.

### Run on Mac

```bash
ssh -O exit barkla1
```

## Repository Setup

Use the login node only for lightweight file-system and Git coordination. Do
not launch Python training, inference or evaluation commands here.

Expected runtime: 1-5 minutes, depending on network and repository state.
Expected memory: negligible.
Recommended partition: not applicable.
Output paths: `~/scratch/tree-seg-benchmark`.
Success criteria: the repository exists and `git status` can run from its root.
Failure indicators: clone failure, missing `.git`, or checkout path mismatch.

### Run on Barkla login node

```bash
mkdir -p "$HOME/scratch"
cd "$HOME/scratch"
git clone https://github.com/EwanMoorcroft/individual-tree-segmentation-benchmarks.git tree-seg-benchmark
cd "$HOME/scratch/tree-seg-benchmark"
git status --short --branch
```

For an existing checkout:

Expected runtime: under 1 minute, unless the remote has large updates.
Expected memory: negligible.
Recommended partition: not applicable.
Output paths: existing checkout under `~/scratch/tree-seg-benchmark`.
Success criteria: the branch updates cleanly.
Failure indicators: merge conflicts, uncommitted local changes blocking the
pull, or the directory is not a Git repository.

### Run on Barkla login node

```bash
cd "$HOME/scratch/tree-seg-benchmark"
git status --short --branch
git pull --ff-only
```

## Lightweight Barkla Environment Check

Run the environment check before submitting a new method workflow or after a
Barkla module change. The script performs only lightweight checks: scheduler
visibility, module availability, repository root checks, expected directory
presence and optional virtual-environment activation.

Expected runtime: under 1 minute.
Expected memory: negligible.
Recommended partition: not applicable.
Output paths: terminal output only.
Success criteria: scheduler commands respond, the repository root is correct,
and the expected Python environment is visible.
Failure indicators: missing repository root, missing modules, missing virtual
environment or unavailable Slurm commands.

### Run on Barkla login node

```bash
cd "$HOME/scratch/tree-seg-benchmark"
bash scripts/check_barkla_environment.sh
```

## Scheduler Snapshot

Use this before diagnosing a failed job. Check the whole chain before treating
an earlier failure as the active blocker; a later replacement job may already
have superseded it.

Expected runtime: under 1 minute.
Expected memory: negligible.
Recommended partition: not applicable.
Output paths: terminal output only.
Success criteria: active and recent jobs are visible.
Failure indicators: Slurm accounting is unavailable or the job ID is absent.

### Run on Barkla login node

```bash
cd "$HOME/scratch/tree-seg-benchmark"
squeue -u "$USER"
sacct -u "$USER" --starttime today --format=JobID,JobName%35,State,Elapsed,ExitCode
```

## Job Submission Pattern

Submission is a login-node task because `sbatch` only registers work with the
scheduler. The job body must perform the heavy computation on allocated Slurm
resources.

Expected runtime: under 1 minute for submission.
Expected memory: negligible on the login node.
Recommended partition: determined by the Slurm file.
Output paths: `logs/slurm/`.
Success criteria: `sbatch` returns a job ID and `squeue` shows the queued or
running job.
Failure indicators: missing log directory, rejected partition, rejected
resource request or missing Slurm file.

### Run on Barkla login node

```bash
cd "$HOME/scratch/tree-seg-benchmark"
mkdir -p logs/slurm
sbatch methods/<method>/slurm/<stage>/<job>.sbatch
squeue -u "$USER"
```

For jobs expected to finish in under 30 minutes:

Expected runtime: interactive watch only; stop it after the smoke job finishes.
Expected memory: negligible on the login node.
Recommended partition: determined by the submitted job.
Live watch command: included below.
Output paths: `logs/slurm/`.
Success criteria: the job reaches `COMPLETED` and the last log lines show the
expected success message.
Failure indicators: `FAILED`, `CANCELLED`, `OUT_OF_MEMORY`, Python traceback
or missing expected output files.

### Run on Barkla login node

```bash
cd "$HOME/scratch/tree-seg-benchmark"
watch -n 10 'squeue -u "$USER"; echo; tail -n 60 logs/slurm/*.out 2>/dev/null'
```

For long jobs, avoid live watching. Use bounded checks:

Expected runtime: under 1 minute per check.
Expected memory: negligible on the login node.
Recommended partition: determined by the submitted job.
Output paths: `logs/slurm/`.
Success criteria: the job remains queued/running or completed cleanly.
Failure indicators: failed state, non-zero exit code or error log tail.

### Run on Barkla login node

```bash
cd "$HOME/scratch/tree-seg-benchmark"
squeue -j <JOB_ID>
sacct -j <JOB_ID> --format=JobID,JobName%35,Partition,State,Elapsed,MaxRSS,ExitCode
tail -n 80 logs/slurm/*<JOB_ID>*.out 2>/dev/null
tail -n 80 logs/slurm/*<JOB_ID>*.err 2>/dev/null
```

## Visualisation Node Use

Use the visualisation node for short, interactive inspection of small outputs.
Do not run full summaries, conversions, inference or training here.

Expected runtime: under 5 minutes.
Expected memory: low; inspect small files only.
Recommended partition: not applicable.
Output paths: terminal output or local visualisation artefacts.
Success criteria: small output files can be opened or listed without starting
a batch workload.
Failure indicators: the command begins scanning large trees, consumes high CPU
or should clearly be converted to a Slurm job.

### Run on Mac

```bash
ssh barklaviz1
```

### Run on Barkla visualisation node

```bash
cd "$HOME/scratch/tree-seg-benchmark"
find results -maxdepth 3 -type f \( -name '*.csv' -o -name '*.json' \) | head -n 40
```

## Heavy Work Rule

The following work must run through Slurm:

- model training;
- deep learning inference;
- full dataset preprocessing;
- full evaluation;
- large CSV or JSON aggregation;
- any task with unknown runtime over a few minutes.

When runtime is unknown, submit a one-plot smoke test first with a 30-minute
wall time.

## Public-Safe Output Rule

Do not commit raw datasets, logs, checkpoints, prediction arrays, private
paths or machine-specific raw outputs. Public documentation may describe output
locations using repository-relative paths such as `logs/slurm/`,
`results/metadata/` and `results/tables/`, or user-relative Barkla paths such
as `~/scratch/tree-seg-benchmark`.
