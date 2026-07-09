#!/bin/bash -l

# Run on Barkla login node. This script is intentionally lightweight: it only
# checks repository, module, scheduler and Python-environment visibility.

set -u

EXPECTED_REPO="${BARKLA_TREEBENCH_REPO:-$HOME/scratch/tree-seg-benchmark}"
EXPECTED_VENV="${BARKLA_TREEBENCH_VENV:-$HOME/fastscratch/venvs/treebench}"
REQUIRED_MODULE="${BARKLA_TREEBENCH_MODULE:-miniforge3/25.3.0-python3.12.10}"

failures=0
warnings=0

section() {
    printf '\n== %s ==\n' "$1"
}

warn() {
    printf '[warn] %s\n' "$1" >&2
    warnings=$((warnings + 1))
}

fail() {
    printf '[fail] %s\n' "$1" >&2
    failures=$((failures + 1))
}

check_command() {
    if command -v "$1" >/dev/null 2>&1; then
        printf '[ok] command available: %s\n' "$1"
    else
        fail "command unavailable: $1"
    fi
}

run_optional() {
    printf '+ %s\n' "$*"
    if "$@"; then
        return 0
    fi
    warn "command returned non-zero: $*"
    return 0
}

echo "Run on Barkla login node"
echo "Expected runtime: under 1 minute"
echo "Expected memory: negligible"

section "Host"
run_optional hostname
run_optional id -un
printf 'PWD=%s\n' "$PWD"

section "Repository"
if [ -d "$EXPECTED_REPO/.git" ]; then
    printf '[ok] expected repository exists: %s\n' "$EXPECTED_REPO"
else
    fail "expected repository is missing or is not a Git checkout: $EXPECTED_REPO"
fi

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    printf '[ok] current directory is inside a Git checkout\n'
    run_optional git rev-parse --show-toplevel
    run_optional git status --short --branch
else
    warn "current directory is not inside a Git checkout"
fi

section "Work Areas"
for path in "$HOME/scratch" "$HOME/data/datasets" "$HOME/fastscratch/venvs" "$HOME/fastscratch/runs"; do
    if [ -d "$path" ]; then
        printf '[ok] directory exists: %s\n' "$path"
    else
        warn "directory missing: $path"
    fi
done

section "Commands"
check_command git
check_command sinfo
check_command squeue
check_command sbatch
check_command sacct

section "Modules"
if type module >/dev/null 2>&1; then
    printf '[ok] environment modules available\n'
    printf '+ module -t avail miniforge3\n'
    module -t avail miniforge3 2>&1 | sed -n '1,20p' || warn "could not list miniforge3 modules"
    printf '+ module -t avail apptainer\n'
    module -t avail apptainer 2>&1 | sed -n '1,20p' || warn "could not list apptainer modules"
    printf '+ module purge\n'
    module purge || warn "module purge failed"
    printf '+ module load %s\n' "$REQUIRED_MODULE"
    module load "$REQUIRED_MODULE" || warn "could not load module: $REQUIRED_MODULE"
else
    fail "environment module command is unavailable"
fi

section "Python Environment"
if [ -f "$EXPECTED_VENV/bin/activate" ]; then
    printf '[ok] virtual environment exists: %s\n' "$EXPECTED_VENV"
    # shellcheck disable=SC1090
    source "$EXPECTED_VENV/bin/activate"
    run_optional python --version
    run_optional python -c 'import sys; print(sys.executable)'
else
    warn "virtual environment missing: $EXPECTED_VENV"
fi

section "Slurm Snapshot"
run_optional sinfo -o '%P %a %l %D %G'
run_optional squeue -u "$USER"
run_optional sacct -u "$USER" --starttime today --format=JobID,JobName%35,State,Elapsed,ExitCode

section "Summary"
printf 'warnings=%s failures=%s\n' "$warnings" "$failures"

if [ "$failures" -gt 0 ]; then
    exit 1
fi
