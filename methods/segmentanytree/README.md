# SegmentAnyTree

## Method Summary

SegmentAnyTree is evaluated on FOR-instance with the released pretrained model
and, separately, after fine-tuning those released weights on development data.
The method uses a PointGroup-style instance segmentation model through the
released SegmentAnyTree code and container interface.

## Upstream Repository And Citation

The released SegmentAnyTree repository and container are external
dependencies. They are not copied into this repository. The upstream project
and paper are recorded in [`docs/for_instance_benchmark.md`](docs/for_instance_benchmark.md)
and [`configs/for_instance_benchmark.yml`](configs/for_instance_benchmark.yml).

## Training Mode Support

The current comparison has two target variants: `published_pretrained`, which
does not update weights, and `fine_tuned_on_dev`, which starts from the same
released checkpoint and updates weights using 16 development training plots.
Five development plots gate checkpoint selection before the 11 held-out test
plots are evaluated. Training from scratch is not part of the current plan.

The completed `retrained_from_dev` run and the rejected 8 July fine-tune remain
historical evidence. Their predictions and results are retained, but neither
is a current target or initial checkpoint. The result roles are recorded in
[`examples/for_instance_result_registry.csv`](examples/for_instance_result_registry.csv).

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

The comparison follows `for_instance_pointwise_v1`. It preserves the
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
  provisional diagnostics, historical results and current targets.

Current canonical equivalents are:

- preparation: [`scripts/data/prepare_segmentanytree_for_instance_training.py`](scripts/data/prepare_segmentanytree_for_instance_training.py);
- inference: [`scripts/runtime/run_segmentanytree_for_instance.py`](scripts/runtime/run_segmentanytree_for_instance.py);
- prediction adaptation: [`scripts/runtime/normalise_segmentanytree_predictions.py`](scripts/runtime/normalise_segmentanytree_predictions.py);
- summarisation: [`scripts/evaluation/summarise_for_instance_segmentanytree_benchmark.py`](scripts/evaluation/summarise_for_instance_segmentanytree_benchmark.py);
- training Slurm entrypoint: [`slurm/training/train_segmentanytree_for_instance_full.sbatch`](slurm/training/train_segmentanytree_for_instance_full.sbatch);
- inference Slurm entrypoint: [`slurm/inference/run_segmentanytree_for_instance_test_from_checkpoint.sbatch`](slurm/inference/run_segmentanytree_for_instance_test_from_checkpoint.sbatch); and
- evaluation Slurm entrypoint: [`slurm/evaluation/evaluate_segmentanytree_for_instance_test_from_checkpoint.sbatch`](slurm/evaluation/evaluate_segmentanytree_for_instance_test_from_checkpoint.sbatch).

## Evaluation Route

Both target results must use aligned internal prediction arrays and the shared
one-to-one point-wise evaluator. The provisional coordinate-rematched
released-checkpoint tables remain diagnostic evidence only and cannot stand in
for the pending aligned pretrained result.

On the 11 held-out plots, the retained historical from-scratch checkpoint has
mean plot F1 `0.4825`
and micro F1 `0.4692` (TP=202, FP=336, FN=121). The
[`final aggregate`](examples/sat_final_test_aligned_summary_sat_for_quicktune_to49_20260706_140730.csv)
and [`provenance manifest`](examples/sat_final_test_aligned_provenance_sat_for_quicktune_to49_20260706_140730.json)
are authoritative for that historical run. The `0.4798` failure-mode tables
are an older diagnostic snapshot for the same checkpoint, not an alternate
result. No metric is reported yet for either current target.

## Known Limitations

Failure-mode diagnostics show that the low held-out score is mainly caused by
over-segmentation and background-confusion false positives. TUWIEN and RMIT
are the weakest site-transfer cases; NIBIO has relatively high recall but low
precision. A validation-only post-processing sweep is retained as a diagnostic
ablation and does not replace the historical unfiltered test result.

## Current Benchmark Status

The retained historical checkpoint is `sat_for_quicktune_to49_20260706_140730`. The
continuation `sat_for_quicktune_to55_20260707_214305` is rejected because
validation fell to `0.451`. The `fine_tuned_on_dev` follow-up
`segmentanytree_for-instance_fine_tuned_on_dev_20260708_215054_full` is also
rejected because its instance output was all background on the audited
held-out plots. The current status is recorded in
[`../../BENCHMARKS.md`](../../BENCHMARKS.md).

The released-pretrained and replacement fine-tuned target results are pending.
The first guarded route in [`slurm/README.md`](slurm/README.md) is now an
isolated released-pretrained smoke on one development plot. It writes each run
to a unique output root, validates the complete released model bundle and
aligned non-zero instance output, and submits neither held-out test inference
nor fine-tuning. Repeated run identifiers archive partial outputs before
retrying.
The rejected 8 July run remains historical evidence and is never used as an
initial checkpoint or target result.
