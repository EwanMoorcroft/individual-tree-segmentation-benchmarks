#!/bin/bash -l
#SBATCH --job-name=tls2trees_frdr
#SBATCH --partition=nodes
#SBATCH --time=06:00:00
#SBATCH --array=0-15
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=logs/tls2trees_frdr_full/%x_%A_%a.out
#SBATCH --error=logs/tls2trees_frdr_full/%x_%A_%a.err

set -euo pipefail

module purge
module load miniforge3/25.3.0-python3.12.10
source ~/fastscratch/venvs/treebench/bin/activate

cd "$HOME/scratch/tree-seg-benchmark"
mkdir -p \
  logs/tls2trees_frdr_full \
  results/metadata/tls2trees_runs \
  data/predictions/tls2trees/frdr_full

MIN_FREE_GB="${TLS2TREES_MIN_FREE_GB:-50}"
FREE_KB=$(df -Pk "$PWD" | awk 'NR == 2 {print $4}')
REQUIRED_KB=$((MIN_FREE_GB * 1024 * 1024))
if (( FREE_KB < REQUIRED_KB )); then
  echo "Insufficient free disk space: require ${MIN_FREE_GB} GiB before TLS2trees." >&2
  df -h "$PWD" >&2
  exit 3
fi

PLOTS=(
  LPine_plot1
  LPine_plot2
  Mixed_plot1
  NPoplar_plot1
  NPoplar_plot2
  NSpruce_plot1
  NSpruce_plot2
  NSpruce_plot3
  RPine_plot1
  SBirch_plot1
  SMaple_plot1
  SPine_plot1
  SPine_plot2
  SPine_plot3
  TAspen_plot1
  TAspen_plot2
)

PLOT_NAME="${PLOTS[$SLURM_ARRAY_TASK_ID]}"

echo "job_id=${SLURM_JOB_ID:-unset}"
echo "array_job_id=${SLURM_ARRAY_JOB_ID:-unset}"
echo "array_task_id=${SLURM_ARRAY_TASK_ID}"
echo "plot_name=${PLOT_NAME}"
echo "hostname=$(hostname)"
echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "cpus_per_task=${SLURM_CPUS_PER_TASK:-unset}"

python scripts/methods/run_tls2trees_instance_for_plot.py \
  --config configs/frdr_tls2trees_benchmark.yml \
  --plot-name "$PLOT_NAME"

echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
