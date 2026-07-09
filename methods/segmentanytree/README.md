# SegmentAnyTree

## Method Summary

SegmentAnyTree is the completed trained labelled-accuracy run for
FOR-instance. The method uses a PointGroup-style instance segmentation model
through the released SegmentAnyTree code and container interface.

## Upstream Repository And Citation

The released SegmentAnyTree repository and container are external
dependencies. They are not copied into this repository. The upstream project
and paper are recorded in [`docs/for_instance_benchmark.md`](docs/for_instance_benchmark.md)
and [`configs/for_instance_benchmark.yml`](configs/for_instance_benchmark.yml).

## Training Mode Support

The primary experiment trains from scratch on the supplied FOR-instance
development split. A fixed seed-42 partition assigns 16 plots to training and
five to internal validation. The 11 test plots were held out until checkpoint
selection and aligned-output checks were complete.

The accepted run is declared as `retrained_from_dev`. The released-checkpoint
route is retained separately as `published_pretrained` and provisional only.
Starting from the released mixed-domain checkpoint and updating weights on the
development split is recorded as `fine_tuned_on_dev`; the attempted follow-up
run is rejected because it produced semantic tree predictions but no accepted
instance predictions in the aligned held-out smoke check.

## Input Requirements

Inputs are the 32 annotated FOR-instance LAS files with `treeID` and
`classification` fields. Tree material uses classes `4`, `5` and `6`; classes
`0`, `1`, `2` and `3` are ignored for the current point-wise instance
evaluation.

## Output Contract

The preferred evaluation input is one aligned predicted semantic label and one
aligned predicted instance label per source point. Final labelled LAZ exports
must pass point-count and coordinate-multiset checks before they can be used
for accuracy.

## FOR-instance Compatibility

The accepted experiment follows `for_instance_pointwise_v1`. It preserves the
supplied development/test boundary and uses development validation only for
checkpoint selection.

## Barkla Environment

The workflow uses the Barkla module and Apptainer settings documented in the
method config and Slurm runbook. The external checkout, container image,
datasets, checkpoints, predictions and logs stay outside Git.

## Slurm Workflow

Start with:

- [`docs/for_instance_benchmark.md`](docs/for_instance_benchmark.md) for the
  benchmark runbook;
- [`docs/running_full_training_20260704.md`](docs/running_full_training_20260704.md)
  for the initial full-run provenance;
- [`docs/training_progress_20260706.md`](docs/training_progress_20260706.md)
  for resume, validation and optimization progress;
- [`slurm/README.md`](slurm/README.md) for the canonical submission sequence;
- [`configs/for_instance_training.yml`](configs/for_instance_training.yml) for
  the fixed training protocol; and
- [`examples/README.md`](examples/README.md) for the distinction between
  provisional diagnostics and accepted results.

Current canonical equivalents are:

- preparation: [`scripts/data/prepare_segmentanytree_for_instance_training.py`](scripts/data/prepare_segmentanytree_for_instance_training.py);
- inference: [`scripts/runtime/run_segmentanytree_for_instance.py`](scripts/runtime/run_segmentanytree_for_instance.py);
- prediction adaptation: [`scripts/runtime/normalise_segmentanytree_predictions.py`](scripts/runtime/normalise_segmentanytree_predictions.py);
- summarisation: [`scripts/evaluation/summarise_for_instance_segmentanytree_benchmark.py`](scripts/evaluation/summarise_for_instance_segmentanytree_benchmark.py);
- training Slurm entrypoint: [`slurm/training/train_segmentanytree_for_instance_full.sbatch`](slurm/training/train_segmentanytree_for_instance_full.sbatch);
- inference Slurm entrypoint: [`slurm/inference/run_segmentanytree_for_instance_test_from_checkpoint.sbatch`](slurm/inference/run_segmentanytree_for_instance_test_from_checkpoint.sbatch); and
- evaluation Slurm entrypoint: [`slurm/evaluation/evaluate_segmentanytree_for_instance_test_from_checkpoint.sbatch`](slurm/evaluation/evaluate_segmentanytree_for_instance_test_from_checkpoint.sbatch).

## Evaluation Route

The accepted result uses aligned internal prediction arrays and the shared
one-to-one point-wise evaluator. The provisional coordinate-rematched
released-checkpoint tables remain diagnostic evidence only.

## Known Limitations

Failure-mode diagnostics show that the low held-out score is mainly caused by
over-segmentation and background-confusion false positives. TUWIEN and RMIT
are the weakest site-transfer cases; NIBIO has relatively high recall but low
precision. A validation-only post-processing sweep is retained as a diagnostic
ablation and does not replace the accepted unfiltered test result.

## Current Benchmark Status

The accepted checkpoint is `sat_for_quicktune_to49_20260706_140730`. The
continuation `sat_for_quicktune_to55_20260707_214305` is rejected because
validation fell to `0.451`. The `fine_tuned_on_dev` follow-up
`segmentanytree_for-instance_fine_tuned_on_dev_20260708_215054_full` is also
rejected because its instance output was all background on the audited
held-out plots. The current status is recorded in
[`../../BENCHMARKS.md`](../../BENCHMARKS.md).
