# SegmentAnyTree On FOR-instance

## Status

SegmentAnyTree inference completed for all 32 FOR-instance LAS files: 21
development plots and 11 test plots across CULS, NIBIO, RMIT, SCION and
TUWIEN. The first evaluation also completed computationally, but it recovered
point correspondence from exported coordinates. Those accuracy values are
provisional because the export is not yet confirmed to preserve one row per
source point.

The provisional results and their limitations are documented in
[`segmentanytree_for_instance_results.md`](segmentanytree_for_instance_results.md).
They must not be used as final benchmark results until the point-aligned
evaluation described below has completed.

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

The completed jobs ran `eval.py` with the supplied `PointGroup-PAPER.pt`
checkpoint. They did not run `train.py`, did not update model weights and did
not train SegmentAnyTree on the local copy of FOR-instance. The checkpoint
SHA-256 and its precise upstream training provenance must be recorded before
claiming a paper reproduction.

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

## Pilot Execution

The canonical pilot file is `CULS/plot_1_annotated.las`. It contains 1,816,672
input points and six positive reference trees. The corrected pilot chain
completed with Slurm jobs 9548698, 9548699 and 9548700 for prediction,
normalisation and the original coordinate-rematched evaluation respectively.
The prediction is valid execution evidence. Its earlier accuracy values remain
provisional until point correspondence is revalidated.

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

## Completed Inference Workflow

The earlier full run used the following dependency chain:

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

The full-array jobs were 9548701 for prediction, 9548702 for normalisation,
9548703 for the coordinate-rematched evaluation and 9548704 for
summarisation. All tasks completed, but successful process exits do not make
the resulting accuracy values paper-comparable.

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

## Output Locations

Large outputs remain ignored by Git:

- labelled predictions:
  `data/predictions/segmentanytree/for_instance/<collection>/<plot>/final_results/`;
- coordinate-normalised instances from the provisional evaluator:
  `data/interim/segmentanytree/for_instance/<collection>/<plot>/normalised_predictions/`;
- run, audit and evaluation metadata:
  `results/metadata/segmentanytree_for_instance/`;
- provisional per-plot metrics and match assignments:
  `results/tables/segmentanytree_for_instance/per_plot/`;
- full plot table:
  `results/tables/segmentanytree_for_instance_plot_metrics.csv`;
- collection and split summaries:
  `results/tables/segmentanytree_for_instance_summary_by_collection.csv` and
  `results/tables/segmentanytree_for_instance_summary_by_split.csv`;
- scheduler logs: `logs/segmentanytree_for_instance/`.

The final labelled predictions use `PredInstance` and `PredSemantic`. The
earlier normaliser wrote one XYZ PLY per positive `PredInstance` and matched
those coordinates back to the source LAS at a 0.02 m tolerance. This is not
the paper-aligned evaluation route.

## Paper-Aligned Evaluation

The SegmentAnyTree paper evaluates FOR-instance using the supplied test split.
Predicted and reference instance IDs are compared point by point and a
prediction is accepted at an IoU threshold of 0.5. The released implementation
uses aligned `preds` and `gt` arrays from semantic and instance evaluation PLY
files. It does not reconstruct correspondence from rounded XYZ coordinates.

The repository reports two policies from the same point-wise IoU matrix:

- `paper_compatible`: each prediction is compared with its best reference,
  matching the released SegmentAnyTree evaluator;
- `harmonized`: a strict one-to-one assignment for comparisons across methods.

The primary published-comparison table must contain only the 11 test plots.
Development plots may be used for diagnostics, but not to select settings after
test results have been inspected.

Before submitting more GPU inference, inspect the existing outputs and audit
the final exports:

```bash
cd ~/scratch/tree-seg-benchmark

INSPECT_JOB=$(sbatch --parsable \
  --array=0-10 \
  --export=ALL,FOR_INSTANCE_SPLIT=test \
  scripts/slurm/inspect_segmentanytree_internal_outputs.sbatch)

AUDIT_JOB=$(sbatch --parsable \
  --array=0-10 \
  --export=ALL,FOR_INSTANCE_SPLIT=test \
  scripts/slurm/audit_segmentanytree_for_instance_export_array.sbatch)

echo "INSPECT_JOB=$INSPECT_JOB"
echo "AUDIT_JOB=$AUDIT_JOB"
```

The audit intentionally fails a task when the final LAZ changes point count or
coordinate multiplicity. A failed audit does not invalidate the model
prediction; it means accuracy must be calculated from the aligned internal
evaluation files instead.

After both jobs leave the queue:

```bash
python scripts/evaluation/summarise_segmentanytree_revalidation.py
```

This writes
`results/tables/segmentanytree_for_instance/revalidation_diagnostics.csv` and
prints the export status and internal candidate counts for each inspected
plot.

If an export passes, the final-LAZ diagnostic evaluator can be submitted for
the test split:

```bash
POINTWISE_JOB=$(sbatch --parsable \
  --array=0-10 \
  --export=ALL,FOR_INSTANCE_SPLIT=test \
  scripts/slurm/evaluate_segmentanytree_pointwise_array.sbatch)
```

Do not submit this evaluator as a dependency on `afterok` for the audit array:
an unsafe export is an expected diagnostic outcome. If internal aligned files
are present, use those files as the primary evaluation input after their names
and fields have been confirmed from the inventory JSON.

## Acceptance Checks

Accept a rerun only when:

1. the checkpoint SHA-256, external commit, container route and package
   versions are recorded;
2. all 11 test plots are present and no development plot is included in the
   primary published-comparison table;
3. prediction and reference labels have stable row-level correspondence;
4. the released paper-compatible policy and the harmonized one-to-one policy
   are reported separately;
5. the IoU threshold and semantic masks are fixed before reading test scores;
6. NIBIO is inspected separately because the provisional result differed
   sharply from the published reference value; and
7. the public workbook is rebuilt only from the validated point-wise results.
