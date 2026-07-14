# SegmentAnyTree

## Method Summary

SegmentAnyTree is evaluated on FOR-instance with the released pretrained model
and, separately, after fine-tuning those released weights on development data.
The method uses a PointGroup-style instance segmentation model through the
released SegmentAnyTree code and container interface.

## Upstream Repository And Citation

The released SegmentAnyTree repository and container are external
dependencies. They are not copied into this repository. The upstream project
is [`SmartForest-no/SegmentAnyTree`](https://github.com/SmartForest-no/SegmentAnyTree),
and the method is described in the
[`SegmentAnyTree` paper](https://doi.org/10.1016/j.rse.2024.114367). Pinned
run-specific details are recorded in
[`docs/for_instance_benchmark.md`](docs/for_instance_benchmark.md) and
[`configs/for_instance_benchmark.yml`](configs/for_instance_benchmark.yml).

## Training Mode Support

The completed comparison has two target variants: `published_pretrained`, which
does not update weights, and `fine_tuned_on_dev`, which starts from the same
released checkpoint and updates weights using 16 development training plots.
Five development plots gate checkpoint selection before the 11 held-out test
plots are evaluated. Training from scratch is not part of the target comparison.

The completed `retrained_from_dev` run and the rejected 8 July fine-tune remain
historical evidence. Their predictions and results are retained, but neither
is a completed target or initial checkpoint. The result roles are recorded in
the method-specific
[`examples/segmentanytree_result_registry.csv`](examples/segmentanytree_result_registry.csv).

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

The completed comparison follows `for_instance_pointwise_v1`. It preserves the
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
- [`docs/final_results_20260711.md`](docs/final_results_20260711.md) for the
  completed target results and retained-artifact contract;
- [`slurm/README.md`](slurm/README.md) for the canonical submission sequence;
- [`configs/for_instance_training.yml`](configs/for_instance_training.yml) for
  the fixed training protocol; and
- [`examples/README.md`](examples/README.md) for the distinction between
  provisional diagnostics, historical results and completed targets.

Canonical equivalents are:

- preparation: [`scripts/data/prepare_segmentanytree_for_instance_training.py`](scripts/data/prepare_segmentanytree_for_instance_training.py);
- inference: [`scripts/runtime/run_segmentanytree_for_instance.py`](scripts/runtime/run_segmentanytree_for_instance.py);
- prediction adaptation: [`scripts/runtime/normalise_segmentanytree_predictions.py`](scripts/runtime/normalise_segmentanytree_predictions.py);
- summarisation: [`scripts/evaluation/summarise_for_instance_segmentanytree_benchmark.py`](scripts/evaluation/summarise_for_instance_segmentanytree_benchmark.py);
- training Slurm entrypoint: [`slurm/training/train_segmentanytree_for_instance_full.sbatch`](slurm/training/train_segmentanytree_for_instance_full.sbatch);
- inference Slurm entrypoint: [`slurm/inference/run_segmentanytree_for_instance_test_from_checkpoint.sbatch`](slurm/inference/run_segmentanytree_for_instance_test_from_checkpoint.sbatch); and
- evaluation Slurm entrypoint: [`slurm/evaluation/evaluate_segmentanytree_for_instance_test_from_checkpoint.sbatch`](slurm/evaluation/evaluate_segmentanytree_for_instance_test_from_checkpoint.sbatch).

## Evaluation Route

Both target results use aligned internal prediction arrays and the shared
one-to-one point-wise evaluator. The provisional coordinate-rematched
released-checkpoint tables remain diagnostic evidence only and do not stand in
for the completed aligned pretrained result.

The completed `fine_tuned_on_dev` primary result has held-out mean plot F1
`0.5447`, micro F1 `0.5320`, mean precision `0.4297` and mean recall `0.7806`
(TP=237, FP=331, FN=86). The completed `published_pretrained` baseline has mean
plot F1 `0.4534` and micro F1 `0.4442` (TP=247, FP=542, FN=76). The full values
and provenance are recorded in
[`docs/final_results_20260711.md`](docs/final_results_20260711.md).
The committed site table is
[`examples/sat_completed_target_site_results_20260711.csv`](examples/sat_completed_target_site_results_20260711.csv).
The committed 22-row per-plot source table is
[`examples/sat_completed_target_plot_results_20260711.csv`](examples/sat_completed_target_plot_results_20260711.csv);
each row records the SHA-256 of its frozen Barkla metrics JSON.

On the 11 held-out plots, the retained historical from-scratch checkpoint has
mean plot F1 `0.4825`
and micro F1 `0.4692` (TP=202, FP=336, FN=121). The
[`final aggregate`](examples/sat_final_test_aligned_summary_sat_for_quicktune_to49_20260706_140730.csv)
and [`provenance manifest`](examples/sat_final_test_aligned_provenance_sat_for_quicktune_to49_20260706_140730.json)
are authoritative for that historical run. The `0.4798` failure-mode tables
are an older diagnostic snapshot for the same checkpoint, not an alternate
result.

## Known Limitations

Failure-mode diagnostics show that the held-out errors are mainly caused by
over-segmentation and background-confusion false positives. For the completed
fine-tuned target, SCION is strongest at mean F1 `0.7206`, TUWIEN is weakest at
`0.3662`, and RMIT is the only site below its released-baseline result. NIBIO
improves to `0.5356` mainly through 136 fewer false positives, with a small
recall reduction. A validation-only post-processing sweep is retained as a
diagnostic ablation and does not replace any completed test result.

## Current Benchmark Status

The retained historical checkpoint is `sat_for_quicktune_to49_20260706_140730`. The
continuation `sat_for_quicktune_to55_20260707_214305` is rejected because
validation fell to `0.451`. The `fine_tuned_on_dev` follow-up
`segmentanytree_for-instance_fine_tuned_on_dev_20260708_215054_full` is also
rejected because its instance output was all background on the audited
held-out plots. The current status is recorded in
[`../../BENCHMARKS.md`](../../BENCHMARKS.md).

The released-pretrained and replacement fine-tuned target results are complete.
The fine-tuned run
`segmentanytree_for-instance_fine_tuned_on_dev_20260711_002931` is the primary
result. The released baseline is
`segmentanytree_for-instance_published_pretrained_20260710_231601`. Both used
frozen, one-time held-out evaluation routes and must not be rerun for setting
selection. Their aligned prediction files, per-plot metadata and summaries are
retained on Barkla for future metrics; use the verifier documented in
[`docs/final_results_20260711.md`](docs/final_results_20260711.md).
The rejected 8 July run remains historical evidence and is never used as an
initial checkpoint or target result.
