# SegmentAnyTree On FOR-instance

## Status

The full benchmark completed prediction, normalisation and labelled evaluation
for all 32 FOR-instance LAS files. The result set contains 21 development plots
and 11 test plots across CULS, NIBIO, RMIT, SCION and TUWIEN. No plot is missing
from the final summary.

The complete results and their limitations are documented in
[`segmentanytree_for_instance_results.md`](segmentanytree_for_instance_results.md).
The public workbook and CSV tables are in [`examples/`](../examples/).

## Dataset And Evaluation Labels

FOR-instance provides plot-wise individual-tree labels in `treeID` and semantic
labels in `classification`. This benchmark includes classes `4` (stem), `5`
(live branches) and `6` (woody branches). Classes `0`, `1`, `2` and `3`, and
non-positive tree IDs, are ignored during reference construction.

The inventory contains 32 LAS files, 151,478,959 points and 1,130 positive
reference trees. Split labels are read from `data_split_metadata.csv` and
retained in every evaluation row.

## Method And External Dependency

SegmentAnyTree remains an external dependency:

- repository: <https://github.com/SmartForest-no/SegmentAnyTree>;
- tested external commit:
  `a3561ed8447bbb7938f059ba65a3e9c97d6e2ee9`;
- Barkla checkout: `external/SegmentAnyTree`;
- container source: `docker://maciekwielgosz/segment-any-tree:latest`;
- Barkla image: `~/scratch/containers/segment-any-tree_latest.sif`.

The external checkout, approximately 7.4 GB SIF image, model checkpoint, raw
data and prediction outputs are excluded from this repository.

## Working Barkla Setup

The validated execution route uses Barkla2, Rocky Linux 9, Apptainer 1.3.6 and
NVIDIA L40S GPUs. Repository utilities use
`miniforge3/25.3.0-python3.12.10` with
`~/fastscratch/venvs/treebench`. Inference runs under Python 3.8 inside the
container.

The full prediction array ran on `gpu-l40s-low`. Each task requested one GPU,
eight CPUs and 48 GiB RAM. The completed tasks used at most 9.608 GiB and the
longest observed per-plot runtime was comfortably below one hour, so the
reusable prediction script now requests a one-hour limit.

Podman was tested first, but its rootless runtime directory under `/run/user`
was unavailable in the Slurm job. Apptainer provided GPU visibility and
reliable host-directory binds, so it is the supported route for this
benchmark.

The upstream container launched under Apptainer, but package-version
inconsistencies had to be resolved before inference could run on Barkla:

- pandas initially failed while importing `DtypeArg` from `pandas._typing`;
- scikit-learn later failed while importing `METRIC_MAPPING64`;
- the working read-only userbase at
  `~/fastscratch/segmentanytree_pyuser_v1` contains NumPy 1.24.4, pandas
  1.5.3, SciPy 1.10.1 and scikit-learn 1.3.2.

The immutable container also required a writable processed-data cache. During
clustering, the upstream one-worker `multiprocessing.Pool` blocked after CUDA
initialisation. The opt-in `sitecustomize.py` compatibility layer runs those
one-worker maps serially. The LAS exporter also required signed 16-bit,
rounded `scan_angle` values. The runtime patch preparation script applies that
narrow correction without changing the external checkout.

Further deployment detail is in
[`segmentanytree_barkla_debug_log.md`](segmentanytree_barkla_debug_log.md).

## Pilot Validation

The canonical pilot file is `CULS/plot_1_annotated.las`. It contains 1,816,672
input points and six positive reference trees. The corrected pilot chain
completed with Slurm jobs 9548698, 9548699 and 9548700 for prediction,
normalisation and evaluation respectively.

The canonical rerun produced 20 predicted trees, six true positives, 14 false
positives and no false negatives. Its precision was 0.300000, recall 1.000000,
F1 0.461538 and mean matched IoU 0.850730. These values supersede the earlier
investigation record that required separate postprocessing.

To repeat the pilot:

```bash
cd ~/scratch/tree-seg-benchmark
mkdir -p logs/segmentanytree_for_instance

SAT_PILOT=$(sbatch --parsable \
  --export=ALL,SEGMENTANYTREE_EXECUTE=1 \
  scripts/slurm/run_segmentanytree_for_instance_pilot_apptainer.sbatch)

SAT_NORM=$(sbatch --parsable \
  --array=0 \
  --dependency=afterok:${SAT_PILOT} \
  scripts/slurm/normalise_segmentanytree_for_instance_array.sbatch)

SAT_EVAL=$(sbatch --parsable \
  --array=0 \
  --dependency=afterok:${SAT_NORM} \
  scripts/slurm/evaluate_segmentanytree_for_instance_array.sbatch)
```

Use `install_segmentanytree_python_stack.sbatch` and
`test_segmentanytree_python_stack_repair.sbatch` only when deliberately
creating or rebuilding the controlled userbase.

## Full Benchmark Workflow

The completed run used the following reusable dependency chain:

```bash
cd ~/scratch/tree-seg-benchmark
mkdir -p logs/segmentanytree_for_instance

PRED_JOB=$(sbatch --parsable \
  --export=ALL,SEGMENTANYTREE_EXECUTE=1 \
  scripts/slurm/run_segmentanytree_for_instance_array.sbatch)

NORM_JOB=$(sbatch --parsable \
  --dependency=afterok:${PRED_JOB} \
  scripts/slurm/normalise_segmentanytree_for_instance_array.sbatch)

EVAL_JOB=$(sbatch --parsable \
  --dependency=afterok:${NORM_JOB} \
  scripts/slurm/evaluate_segmentanytree_for_instance_array.sbatch)

SUMMARY_JOB=$(sbatch --parsable \
  --dependency=afterok:${EVAL_JOB} \
  scripts/slurm/summarise_segmentanytree_for_instance.sbatch)
```

The completed full-array jobs were 9548701 for prediction, 9548702 for
normalisation, 9548703 for evaluation and 9548704 for summarisation. All 32
tasks in each array completed successfully.

The prediction job invokes the container interface:

```text
bash run_inference.sh /sat_input /sat_output true
```

The Python wrapper remains useful for plot selection and dry-run metadata, but
it does not execute this benchmark natively:

```bash
python scripts/methods/run_segmentanytree_for_instance.py \
  --plot-path CULS/plot_1_annotated.las \
  --dry-run
```

## Working Output Locations

Large outputs remain ignored by Git:

- labelled predictions:
  `data/predictions/segmentanytree/for_instance/<collection>/<plot>/final_results/`;
- normalised instances:
  `data/interim/segmentanytree/for_instance/<collection>/<plot>/normalised_predictions/`;
- run and evaluation metadata:
  `results/metadata/segmentanytree_for_instance/`;
- per-plot metrics and match assignments:
  `results/tables/segmentanytree_for_instance/per_plot/`;
- full plot table:
  `results/tables/segmentanytree_for_instance_plot_metrics.csv`;
- collection and split summaries:
  `results/tables/segmentanytree_for_instance_summary_by_collection.csv` and
  `results/tables/segmentanytree_for_instance_summary_by_split.csv`;
- scheduler logs: `logs/segmentanytree_for_instance/`.

The final labelled predictions use `PredInstance` and `PredSemantic`. The
normaliser writes one XYZ PLY per positive `PredInstance`. Evaluation matches
those coordinates to the unchanged source LAS at a 0.02 m tolerance and uses a
0.5 IoU threshold.

## Validation After A Rerun

After any rerun:

1. confirm 32 completed rows and no missing relative path;
2. confirm each final LAZ contains positive `PredInstance` values;
3. compare output and input point counts;
4. confirm 32 normalisation and 32 evaluation metadata records;
5. rebuild the summary and public-safe tables;
6. inspect the NIBIO collection separately; and
7. retain the supplied split labels and unchanged evaluation thresholds.
