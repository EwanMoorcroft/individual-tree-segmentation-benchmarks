# Slurm Resource Guide

This guide defines conservative Slurm defaults for the benchmark repository.
Start every new method or changed workflow with a smoke job before a full run.
Use login nodes only to inspect partitions, create log directories, submit jobs
and read bounded log tails.

## Partition Discovery

Confirm available partitions before choosing resources. Do not edit a Slurm
file to request a partition that is not present in `sinfo`.

Expected runtime: under 1 minute.
Expected memory: negligible on the login node.
Recommended partition: not applicable.
Output paths: terminal output only.
Success criteria: partitions and generic resources are listed.
Failure indicators: `sinfo` is unavailable or expected GPU resources are not
listed.

### Run on Barkla login node

```bash
sinfo -o "%P %a %l %D %G" | sed -n '1,40p'
```

## Baseline Resource Classes

| Work type | Recommended partition | Time | Memory | CPU | GPU | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Scheduler/environment check | none | under 1 minute | negligible | login-node shell only | 0 | Lightweight only |
| CPU inventory or metadata check | `nodes` or available CPU partition | 00:30:00 | 8-16G | 1-4 | 0 | Prefer smaller arrays over one large job |
| CPU evaluation or CSV aggregation | `nodes` or available CPU partition | 01:00:00 | 32G | 4-8 | 0 | Increase only after memory evidence |
| GPU smoke inference or training | `gpu-l40s-low` where available | 00:30:00 | 32G | 4-8 | 1 | One plot or one tiny training slice |
| GPU training or large inference | `gpu-l40s` where available | 04:00:00 to 24:00:00 | 64G | 8-16 | 1 | Single GPU unless the method has proven multi-GPU support |

Only request more resources when logs show a specific need, such as
out-of-memory, CPU starvation or a wall-time limit.

## Required Slurm Fields

Every committed Slurm file should include:

- job name;
- partition;
- wall time;
- memory;
- `cpus-per-task` where relevant;
- GPU request where relevant;
- `logs/slurm` or method-specific ignored log output path;
- environment setup;
- repository root check;
- start and end timestamps;
- command echoing before the main command;
- failure behaviour with `set -euo pipefail` and an error trap;
- expected runtime, memory, output paths, success criteria and failure
  indicators in adjacent documentation.

## Smoke-Test Slurm Template

Use this for one-plot or one-batch GPU checks before a full GPU run. Replace
the placeholder command with a method-specific smoke command.

Expected runtime: 5-30 minutes on one GPU node.
Expected memory: 32G.
Recommended partition: `gpu-l40s-low` where available, otherwise the smallest
available GPU partition.
Live watch command: use the short-job watch command below.
Output paths: `logs/slurm/%x_%j.out`, `logs/slurm/%x_%j.err` and the
method-specific smoke output directory.
Success criteria: environment loads, the repository check passes, one small
input completes and expected smoke outputs are present.
Failure indicators: missing module, missing virtual environment, missing input,
Python traceback, CUDA failure, out-of-memory or no output file.

### Run through Slurm

```bash
#!/bin/bash -l
#SBATCH --job-name=treebench_smoke
#SBATCH --partition=gpu-l40s-low
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail
trap 'echo "[error] ${BASH_COMMAND} failed at line ${LINENO}" >&2' ERR
set -x

echo "[start] $(date -Is)"

module purge
module load miniforge3/25.3.0-python3.12.10
source "$HOME/fastscratch/venvs/treebench/bin/activate"

REPO_ROOT="$HOME/scratch/tree-seg-benchmark"
test -d "$REPO_ROOT/.git"
cd "$REPO_ROOT"

nvidia-smi

# Replace this with a one-plot or one-batch method command.
bash methods/<method>/slurm/<stage>/<smoke-task>.sh

echo "[end] $(date -Is)"
```

Submit and watch a smoke job:

Expected runtime: under 1 minute to submit; 5-30 minutes for the Slurm job.
Expected memory: negligible on the login node.
Recommended partition: determined by the Slurm file.
Live watch command: included below.
Output paths: `logs/slurm/`.
Success criteria: `sbatch` returns a job ID and the log shows `[end]`.
Failure indicators: rejected resources, failed job state or error log output.

### Run on Barkla login node

```bash
cd "$HOME/scratch/tree-seg-benchmark"
mkdir -p logs/slurm
sbatch methods/<method>/slurm/<stage>/<smoke-job>.sbatch
watch -n 10 'squeue -u "$USER"; echo; tail -n 60 logs/slurm/*.out 2>/dev/null'
```

## GPU Training Slurm Template

Use this only after a smoke job passes. Keep the default to one GPU unless the
method has a tested distributed-training route.

Expected runtime: 2-24 hours on one GPU node, depending on method and dataset
slice.
Expected memory: 64G.
Recommended partition: `gpu-l40s` where available.
Live watch command: not recommended for long training; use bounded checks.
Output paths: `logs/slurm/%x_%j.out`, `logs/slurm/%x_%j.err`,
`results/metadata/` and an ignored checkpoint/run directory outside committed
source files.
Success criteria: checkpoints and training metadata are written, the final log
line records completion and no held-out test data are used for tuning.
Failure indicators: out-of-memory, wall-time cancellation, missing checkpoint,
test-set leakage, Python traceback or repeated requeue without progress.

### Run through Slurm

```bash
#!/bin/bash -l
#SBATCH --job-name=treebench_train
#SBATCH --partition=gpu-l40s
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail
trap 'echo "[error] ${BASH_COMMAND} failed at line ${LINENO}" >&2' ERR
set -x

echo "[start] $(date -Is)"

module purge
module load miniforge3/25.3.0-python3.12.10
source "$HOME/fastscratch/venvs/treebench/bin/activate"

REPO_ROOT="$HOME/scratch/tree-seg-benchmark"
test -d "$REPO_ROOT/.git"
cd "$REPO_ROOT"

mkdir -p logs/slurm results/metadata
nvidia-smi

# Replace this with the method-specific training command.
bash methods/<method>/slurm/training/<train-task>.sh

echo "[end] $(date -Is)"
```

Submit and check a long GPU job:

Expected runtime: under 1 minute per scheduler check after submission.
Expected memory: negligible on the login node.
Recommended partition: determined by the Slurm file.
Live watch command: not recommended.
Output paths: `logs/slurm/`.
Success criteria: job is queued, running or completed with exit code `0:0`.
Failure indicators: failed state, non-zero exit code, out-of-memory or wall-time
limit.

### Run on Barkla login node

```bash
cd "$HOME/scratch/tree-seg-benchmark"
mkdir -p logs/slurm
sbatch methods/<method>/slurm/training/<train-job>.sbatch
squeue -u "$USER"
sacct -u "$USER" --starttime today --format=JobID,JobName%35,Partition,State,Elapsed,MaxRSS,ExitCode
```

## CPU Evaluation Slurm Template

Use CPU jobs for inventory, format conversion, point-wise evaluation,
metadata checks and CSV/JSON aggregation. Move to arrays when many plots can
run independently.

Expected runtime: 10-60 minutes on one CPU node for small to medium
evaluations.
Expected memory: 32G.
Recommended partition: `nodes` or the available CPU partition shown by
`sinfo`.
Live watch command: use the short-job watch command if the job is expected
under 30 minutes.
Output paths: `logs/slurm/%x_%j.out`, `logs/slurm/%x_%j.err`,
`results/metadata/` and `results/tables/`.
Success criteria: every expected plot has a metrics file and the summary script
exits with code `0`.
Failure indicators: missing prediction/reference files, coordinate alignment
failure, memory exhaustion, Python traceback or partial output tables.

### Run through Slurm

```bash
#!/bin/bash -l
#SBATCH --job-name=treebench_eval
#SBATCH --partition=nodes
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail
trap 'echo "[error] ${BASH_COMMAND} failed at line ${LINENO}" >&2' ERR
set -x

echo "[start] $(date -Is)"

module purge
module load miniforge3/25.3.0-python3.12.10
source "$HOME/fastscratch/venvs/treebench/bin/activate"

REPO_ROOT="$HOME/scratch/tree-seg-benchmark"
test -d "$REPO_ROOT/.git"
cd "$REPO_ROOT"

mkdir -p logs/slurm results/metadata results/tables

# Replace this with the method-specific CPU evaluation command.
bash methods/<method>/slurm/evaluation/<evaluation-task>.sh

echo "[end] $(date -Is)"
```

Submit and watch a short CPU evaluation:

Expected runtime: under 1 minute to submit; 10-30 minutes for the Slurm job.
Expected memory: negligible on the login node.
Recommended partition: determined by the Slurm file.
Live watch command: included below.
Output paths: `logs/slurm/`, `results/metadata/`, `results/tables/`.
Success criteria: job completes and expected metrics files are present.
Failure indicators: failed job state, missing metrics, traceback or alignment
failure.

### Run on Barkla login node

```bash
cd "$HOME/scratch/tree-seg-benchmark"
mkdir -p logs/slurm
sbatch methods/<method>/slurm/evaluation/<evaluation-job>.sbatch
watch -n 10 'squeue -u "$USER"; echo; tail -n 60 logs/slurm/*.out 2>/dev/null'
```

## Resource Escalation Rules

Increase one resource at a time and only from evidence:

- increase memory after `OUT_OF_MEMORY`, `oom-kill` or clear allocation errors;
- increase time after a clean wall-time limit with useful progress;
- increase CPU count only for code that uses multiple workers or OpenMP;
- request more than one GPU only when the method has a tested multi-GPU path;
- split plot-independent work into arrays before requesting a much larger
  single job.

Record the reason for resource changes in method documentation or run metadata.
