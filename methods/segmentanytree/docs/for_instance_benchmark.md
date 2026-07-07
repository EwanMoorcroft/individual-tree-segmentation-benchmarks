# SegmentAnyTree On FOR-instance

## Status

SegmentAnyTree inference completed for all 32 FOR-instance LAS files: 21
development plots and 11 test plots across CULS, NIBIO, RMIT, SCION and
TUWIEN. That run used the released checkpoint and its final exports failed
point-correspondence checks. Its accuracy values are provisional diagnostic
outputs, not final dissertation results.

The provisional results and their limitations are documented in
[`provisional_released_checkpoint_results.md`](provisional_released_checkpoint_results.md).
They must not be used as final benchmark results. The corrected primary
experiment will train a new SegmentAnyTree model from FOR-instance development
data, select it with a fixed internal development validation split and then
evaluate the frozen checkpoint on the held-out test split using aligned
point-wise outputs.

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
copied by the container has SHA-256
`0b4d74b4644e37a16f59008ad0f5c62894fc4d2d906f3abd803bbfc5b5dd803a`.
That hash was consistent across the inspected outputs. The checkpoint's
committed Hydra overrides identify `treeins_rad8`,
`area4_ablation_3heads_5`, contain an `epochs=100` override and label the job
as a mixed training run rather than a local ULS-only retraining. Its exact
augmentation membership is not fully established from the committed metadata,
so the inference result is not presented as an exact scenario reproduction.

## Corrected Training Protocol

The pinned upstream training route is `train.py` with the panoptic
`treeins_rad8` dataset and `area4_ablation_3heads_5` model configuration. The
upstream conversion script provides the following reproducible preparation:

- read the supplied development/test metadata;
- choose 25% of development plots for validation with random seed 42;
- write `x`, `y`, `z`, `intensity`, `semantic_seg` and `treeID` fields;
- map classes 4, 5 and 6 to binary tree class 2;
- map classes 1 and 2 to non-tree class 1;
- map classes 0 and 3 to ignored class 0; and
- retain all points rather than removing ground, low vegetation or out-points.

For the current 21 development plots, this produces 16 training plots and 5
validation plots. The 11 test records are retained in the ignored manifest but
are never converted into the training data root.

The primary corrected run is `retrained_from_dev`, corresponding to the
paper's ULS-only scenario. Fine-tuning the released mixed-domain checkpoint is
a different experiment and will only be considered after the from-scratch
reproduction. The pinned checkpoint resume implementation restores its saved
run configuration and optimizer state, so it is not treated as a generic
fine-tuning interface without an explicit, tested compatibility change.

The relevant public configuration is
[`for_instance_training.yml`](../configs/for_instance_training.yml).
Converted PLY files, split manifests containing machine paths, checkpoints and
training logs remain outside Git.

### Current Barkla Checkpoint

The initial full run produced an epoch-30 checkpoint before its next MeanShift
stage became impractically slow. The resumed epoch-45 checkpoint reached mean
aligned F1 `0.4580` across the five fixed development validation plots. A
two-epoch continuation completed in `01:43:26` and improved every validation
plot, reaching mean F1 `0.5127` at epoch 47.

A further two-epoch continuation to epoch 49 is running as job `9668753` when
last recorded. The 11 held-out test plots have not been submitted. Current
results, resource observations and the checkpoint decision gate are recorded
in
[`training_progress_20260706.md`](training_progress_20260706.md).
The original July 4 source hashes and job chain remain in
[`running_full_training_20260704.md`](running_full_training_20260704.md) as
historical provenance.

Future submissions should use the guarded workflow documented in
[`../slurm/README.md`](../slurm/README.md), which derives the five-task
validation range from the split manifest.

The pinned trainer loops to, but not including, `training.epochs`; the wrapper
therefore passes a stop value one greater than the requested epoch count and
records both values. Do not submit the guarded test scripts until the full
checkpoint is selected from development validation and its settings are
frozen.

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
[`barkla_debug_log.md`](barkla_debug_log.md).

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
  methods/segmentanytree/slurm/inference/run_segmentanytree_for_instance_pilot_apptainer.sbatch)

SAT_NORM=$(sbatch --parsable \
  --array=0 \
  --dependency=afterok:${SAT_PILOT} \
  methods/segmentanytree/slurm/evaluation/normalise_segmentanytree_for_instance_array.sbatch)

SAT_EVAL=$(sbatch --parsable \
  --array=0 \
  --dependency=afterok:${SAT_NORM} \
  methods/segmentanytree/slurm/evaluation/evaluate_segmentanytree_for_instance_array.sbatch)
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
  methods/segmentanytree/slurm/inference/run_segmentanytree_for_instance_array.sbatch)

NORM_JOB=$(sbatch --parsable \
  --dependency=afterok:${PRED_JOB} \
  methods/segmentanytree/slurm/evaluation/normalise_segmentanytree_for_instance_array.sbatch)

EVAL_JOB=$(sbatch --parsable \
  --dependency=afterok:${NORM_JOB} \
  methods/segmentanytree/slurm/evaluation/evaluate_segmentanytree_for_instance_array.sbatch)

SUMMARY_JOB=$(sbatch --parsable \
  --dependency=afterok:${EVAL_JOB} \
  methods/segmentanytree/slurm/evaluation/summarise_segmentanytree_for_instance.sbatch)
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
python methods/segmentanytree/scripts/runtime/run_segmentanytree_for_instance.py \
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
  methods/segmentanytree/slurm/evaluation/inspect_segmentanytree_internal_outputs.sbatch)

AUDIT_JOB=$(sbatch --parsable \
  --array=0-10 \
  --export=ALL,FOR_INSTANCE_SPLIT=test \
  methods/segmentanytree/slurm/evaluation/audit_segmentanytree_for_instance_export_array.sbatch)

echo "INSPECT_JOB=$INSPECT_JOB"
echo "AUDIT_JOB=$AUDIT_JOB"
```

The audit intentionally fails a task when the final LAZ changes point count or
coordinate multiplicity. A failed audit does not invalidate the model
prediction; it means accuracy must be calculated from the aligned internal
evaluation files instead.

After both jobs leave the queue:

```bash
python methods/segmentanytree/scripts/evaluation/summarise_segmentanytree_revalidation.py
```

This writes
`results/tables/segmentanytree_for_instance/revalidation_diagnostics.csv` and
prints the export status and internal candidate counts for each inspected
plot.

The first test-split audit snapshot contained nine completed audits and all
nine failed; the other two tasks were still running when the snapshot was
recorded. Examples of output row inflation were 66 rows for CULS
`plot_2_annotated`, 104,470 rows for NIBIO `plot_1_annotated`, 164,290 rows
for NIBIO `plot_22_annotated`, and 2,056,634 rows for TUWIEN `test`. These
failures are sufficient to reject the final-LAZ route for the accepted
benchmark.

The released tracker computes aligned full-resolution instance predictions and
retains the matching ground-truth array in memory, but the corresponding
`to_eval_ply` call is commented out in the pinned upstream source. The rerun
uses a narrow runtime patch to write
`Instance_results_forEval_0.ply` before the faulty export merge. The upstream
semantic evaluation PLY is retained unchanged. The dedicated evaluation route
stops before final LAZ merging.

Validate this route on the first CULS and NIBIO test plots before submitting
all 11 test plots:

```bash
PAPER_PILOT=$(sbatch --parsable \
  --array=0-1%1 \
  --export=ALL,SEGMENTANYTREE_EXECUTE=1 \
  methods/segmentanytree/slurm/inference/run_segmentanytree_for_instance_paper_test_array.sbatch)

PAPER_EVAL=$(sbatch --parsable \
  --array=0-1%1 \
  --dependency=afterok:${PAPER_PILOT} \
  methods/segmentanytree/slurm/evaluation/evaluate_segmentanytree_paper_test_array.sbatch)
```

Only expand this to array indices `0-10` after both pilot evaluations contain
the expected aligned point counts, checkpoint hash and plausible metrics.

## Acceptance Checks

Accept a rerun only when:

1. the new checkpoint SHA-256, external commit, container route and package
   versions are recorded;
2. the split manifest contains 16 training, 5 validation and 11 held-out test
   records, with no converted test PLY in the training root;
3. model selection uses development validation results only;
4. all 11 test plots are present and no development plot is included in the
   final headline table;
5. prediction and reference labels have stable row-level correspondence;
6. the released paper-compatible policy and harmonised one-to-one policy are
   reported separately;
7. the IoU threshold and semantic masks are fixed before the final test job;
8. NIBIO is inspected separately without tuning against its test scores; and
9. the public workbook is rebuilt only from validated trained-model results.
