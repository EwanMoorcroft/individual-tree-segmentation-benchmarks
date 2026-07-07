#!/bin/bash -l
#SBATCH --job-name=treex_eval_dev
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-0
#SBATCH --output=logs/treex_for_instance/evaluate_treex_for_instance_array_%A_%a.out
#SBATCH --error=logs/treex_for_instance/evaluate_treex_for_instance_array_%A_%a.err

set -euo pipefail

cd ~/scratch/tree-seg-benchmark

module purge
module load miniforge3/25.3.0-python3.12.10
source ~/fastscratch/venvs/treex/bin/activate

EVAL_LIST="results/treex_for_instance/treex_eval_summary_list.csv"

LINE_NO=$((SLURM_ARRAY_TASK_ID + 2))
ROW=$(sed -n "${LINE_NO}p" "$EVAL_LIST")

if [[ -z "$ROW" ]]; then
  echo "No row for task ${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

PLOT_ID=$(echo "$ROW" | cut -d',' -f1)
SAFE_PLOT_ID=$(echo "$ROW" | cut -d',' -f2)
SUMMARY_JSON=$(echo "$ROW" | cut -d',' -f3)
PRED_NPZ=$(echo "$ROW" | cut -d',' -f4)

echo "hostname=$(hostname)"
echo "date_utc=$(date -u)"
echo "array_job_id=${SLURM_ARRAY_JOB_ID}"
echo "array_task_id=${SLURM_ARRAY_TASK_ID}"
echo "plot_id=${PLOT_ID}"
echo "safe_plot_id=${SAFE_PLOT_ID}"
echo "summary_json=${SUMMARY_JSON}"
echo "prediction_npz=${PRED_NPZ}"

if [[ ! -f "$PRED_NPZ" ]]; then
  echo "Missing prediction NPZ: $PRED_NPZ"
  exit 1
fi

python methods/treex/scripts/evaluate_treex_for_instance_plot.py \
  --prediction-npz "$PRED_NPZ" \
  --plot-id "$PLOT_ID" \
  --metrics-json "results/treex_for_instance/${SAFE_PLOT_ID}_treex_metrics.json" \
  --metrics-csv "results/treex_for_instance/${SAFE_PLOT_ID}_treex_metrics.csv" \
  --matches-csv "results/treex_for_instance/${SAFE_PLOT_ID}_treex_matches.csv" \
  --diagnostics-csv "results/treex_for_instance/${SAFE_PLOT_ID}_treex_prediction_diagnostics.csv"

echo "Done."
