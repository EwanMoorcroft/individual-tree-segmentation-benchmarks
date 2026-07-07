#!/bin/bash -l
#SBATCH --job-name=treex_for_test
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-0
#SBATCH --output=logs/treex_for_instance/run_treex_for_instance_test_array_%A_%a.out
#SBATCH --error=logs/treex_for_instance/run_treex_for_instance_test_array_%A_%a.err

set -euo pipefail

cd ~/scratch/tree-seg-benchmark

module purge
module load miniforge3/25.3.0-python3.12.10
source ~/fastscratch/venvs/treex/bin/activate

PLOT_LIST="results/treex_for_instance/treex_for_instance_test_plots.csv"

if [[ ! -f "$PLOT_LIST" ]]; then
  echo "Missing plot list: $PLOT_LIST"
  exit 1
fi

LINE_NO=$((SLURM_ARRAY_TASK_ID + 2))
ROW=$(sed -n "${LINE_NO}p" "$PLOT_LIST")

if [[ -z "$ROW" ]]; then
  echo "No row for task ${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

PLOT_ID=$(echo "$ROW" | cut -d',' -f1)
INPUT_LAS=$(echo "$ROW" | cut -d',' -f2)

SAFE_PLOT_ID=$(echo "$PLOT_ID" | tr '/' '_')
OUTPUT_DIR="data/predictions/treex/for_instance_test/${SAFE_PLOT_ID}"
SUMMARY_JSON="results/treex_for_instance/${SAFE_PLOT_ID}_treex_test_summary.json"
PERFORMANCE_CSV="results/treex_for_instance/${SAFE_PLOT_ID}_treex_test_performance.csv"

echo "hostname=$(hostname)"
echo "date_utc=$(date -u)"
echo "array_job_id=${SLURM_ARRAY_JOB_ID}"
echo "array_task_id=${SLURM_ARRAY_TASK_ID}"
echo "line_no=${LINE_NO}"
echo "row=${ROW}"
echo "plot_id=${PLOT_ID}"
echo "input_las=${INPUT_LAS}"
echo "output_dir=${OUTPUT_DIR}"
echo "summary_json=${SUMMARY_JSON}"
echo "performance_csv=${PERFORMANCE_CSV}"

if [[ ! -f "$INPUT_LAS" ]]; then
  echo "Input LAS does not exist: $INPUT_LAS"
  exit 1
fi

python methods/treex/scripts/run_treex_for_instance_plot.py \
  --input-las "$INPUT_LAS" \
  --output-dir "$OUTPUT_DIR" \
  --summary-json "$SUMMARY_JSON" \
  --performance-csv "$PERFORMANCE_CSV" \
  --plot-id "$PLOT_ID" \
  --workers "$SLURM_CPUS_PER_TASK"

echo "Done."
