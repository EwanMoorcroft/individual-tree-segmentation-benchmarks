# SegmentAnyTree On FOR-instance

## Status

The prediction, normalisation and labelled evaluation pilot completed for
`CULS/plot_1_annotated.las`, a development-split plot containing 1,816,672
input points and six positive reference trees. The full 32-file benchmark has
not been run.

The pilot produced a labelled LAZ file, 21 normalised predicted instances and
an evaluation against the original `treeID` reference. At an IoU threshold of
0.5 and coordinate tolerance of 0.02 m, all six reference trees were matched:

| Metric | Pilot value |
| --- | ---: |
| Predicted trees | 21 |
| Reference trees | 6 |
| True positives | 6 |
| False positives | 15 |
| False negatives | 0 |
| Precision | 0.285714 |
| Recall | 1.000000 |
| F1 | 0.444444 |
| Mean matched IoU | 0.850764 |
| Median matched IoU | 0.862035 |

These are pilot results, not full benchmark results. The public-safe record is
in
[`examples/segmentanytree_for_instance_pilot_metrics.csv`](../examples/segmentanytree_for_instance_pilot_metrics.csv).

## Dataset And Evaluation Labels

FOR-instance provides plot-wise individual-tree labels in `treeID` and semantic
labels in `classification`. This benchmark includes classes `4` (stem), `5`
(live branches) and `6` (woody branches). Classes `0`, `1`, `2` and `3`, and
non-positive tree IDs, are ignored during reference construction.

The pilot is assigned to the `dev` split in `data_split_metadata.csv`. Split
labels are retained for every array task; evaluation plots must not be used for
parameter tuning.

## Method And External Dependency

SegmentAnyTree remains an external dependency:

- repository: <https://github.com/SmartForest-no/SegmentAnyTree>;
- tested external commit:
  `a3561ed8447bbb7938f059ba65a3e9c97d6e2ee9`;
- Barkla checkout: `external/SegmentAnyTree`;
- container source: `docker://maciekwielgosz/segment-any-tree:latest`;
- Barkla image: `~/scratch/containers/segment-any-tree_latest.sif`.

The external checkout, approximately 7.4 GB SIF image, model checkpoint and
predictions are excluded from this repository.

## Barkla Execution Route

The working route uses Barkla2, Rocky Linux 9, the `gpu-l40s` partition and an
NVIDIA L40S GPU. Repository utilities use
`miniforge3/25.3.0-python3.12.10` and
`~/fastscratch/venvs/treebench`; inference uses Apptainer 1.3.6 and Python 3.8
inside the container.

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

The immutable container also required a writable bind for its processed-data
cache. During clustering, the upstream single-process
`multiprocessing.Pool` blocked after CUDA initialisation. The opt-in
`sitecustomize.py` compatibility layer runs those one-worker maps serially.
The final LAS export required signed 16-bit, rounded `scan_angle` values. The
runtime patch preparation script applies that narrow correction to the tested
external source without modifying or copying the external repository into
Git.

Further detail is recorded in
[`segmentanytree_barkla_debug_log.md`](segmentanytree_barkla_debug_log.md).

## Pilot Commands

The successful investigation used
`run_segmentanytree_for_instance_pilot_apptainer_v3.sbatch`. Inference and
instance prediction finished in that job, but the upstream final LAS export
failed. `postprocess_segmentanytree_pilot_v1.sbatch` reran the exporter with the
signed `scan_angle` correction, and
`evaluate_segmentanytree_pilot_v2.sbatch` completed the labelled evaluation.
Those numbered investigation scripts were consolidated rather than retained.

The reusable pilot now applies both runtime corrections:

```bash
cd ~/scratch/tree-seg-benchmark
mkdir -p logs/segmentanytree_for_instance

SAT_CONTAINER=$(sbatch --parsable \
  scripts/slurm/test_segmentanytree_apptainer.sbatch)

SAT_STACK=$(sbatch --parsable \
  --dependency=afterok:${SAT_CONTAINER} \
  scripts/slurm/install_segmentanytree_python_stack.sbatch)

SAT_STACK_TEST=$(sbatch --parsable \
  --dependency=afterok:${SAT_STACK} \
  scripts/slurm/test_segmentanytree_python_stack_repair.sbatch)

SAT_PILOT=$(sbatch --parsable \
  --dependency=afterok:${SAT_STACK_TEST} \
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

The install job is needed only when creating or deliberately rebuilding the
controlled userbase. The prediction script leaves an existing final LAZ
unchanged and refuses a non-empty partial output directory. Archive the
previous ignored pilot output before a clean reproduction.

The prediction job invokes the container interface:

```text
bash run_inference.sh /sat_input /sat_output true
```

The Python wrapper remains useful for selection and dry-run metadata, but it
does not execute this benchmark natively:

```bash
python scripts/methods/run_segmentanytree_for_instance.py \
  --plot-path CULS/plot_1_annotated.las \
  --dry-run
```

## Pilot Outputs

Full outputs remain ignored by Git:

- labelled prediction:
  `data/predictions/segmentanytree/for_instance/CULS/plot_1_annotated/final_results/plot_1_annotated_out.laz`;
- normalised instances:
  `data/interim/segmentanytree/for_instance/CULS/plot_1_annotated/normalised_predictions/`;
- run and evaluation metadata:
  `results/metadata/segmentanytree_for_instance/`;
- pilot metrics:
  `results/tables/segmentanytree_for_instance/per_plot/CULS_plot_1_annotated.csv`;
- matched pairs:
  `results/tables/segmentanytree_for_instance/per_plot/CULS_plot_1_annotated_matches.csv`;
- scheduler logs: `logs/segmentanytree_for_instance/`.

The final prediction contains `PredInstance` and `PredSemantic`. It contains
1,816,728 points, 56 more than the source LAS because the upstream merge uses
coordinate joins across intermediate outputs. The normaliser separates
positive `PredInstance` values into one XYZ PLY per predicted tree. Evaluation
then matches those coordinates to the unchanged source LAS at the configured
0.02 m tolerance. This point-count difference remains a scaling risk and must
be checked on every plot.

The recorded pilot runtime was 175 seconds across inference and repaired
postprocessing, with peak memory of 6.334 GB. The postprocessing repair took 35
seconds. These measurements are pilot observations and should not be treated
as full-array resource estimates.

## Full Benchmark

Before submitting all 32 files, confirm the canonical pilot script reproduces
the final LAZ without a separate repair job and inspect the 15 unmatched pilot
predictions. Then submit the dependency chain:

```bash
PRED_JOB=$(sbatch --parsable \
  --export=ALL,SEGMENTANYTREE_EXECUTE=1 \
  scripts/slurm/run_segmentanytree_for_instance_array.sbatch)

NORM_JOB=$(sbatch --parsable \
  --dependency=afterok:${PRED_JOB} \
  scripts/slurm/normalise_segmentanytree_for_instance_array.sbatch)

EVAL_JOB=$(sbatch --parsable \
  --dependency=afterok:${NORM_JOB} \
  scripts/slurm/evaluate_segmentanytree_for_instance_array.sbatch)

sbatch --dependency=afterok:${EVAL_JOB} \
  scripts/slurm/summarise_segmentanytree_for_instance.sbatch
```

For each plot, verify completion status, final point count, positive predicted
instance count, coordinate matching, runtime and peak memory. Resolve missing
or failed tasks before reporting collection, split or overall results.
