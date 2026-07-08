# SegmentAnyTree

SegmentAnyTree is the current labelled-accuracy method for FOR-instance.

## Completed experiment

The primary experiment trains from scratch on the supplied FOR-instance
development split. A fixed seed-42 partition assigns 16 plots to training and
five to internal validation. The 11 test plots were held out until checkpoint
selection and aligned-output checks were complete.

The accepted checkpoint is `sat_for_quicktune_to49_20260706_140730`. It
reports mean aligned F1 `0.537` across the five development validation plots
and `0.480` across the 11 held-out test plots. The continuation
`sat_for_quicktune_to55_20260707_214305` is rejected because validation fell to
`0.451`.

Failure-mode diagnostics show that the low held-out score is mainly caused by
over-segmentation and background-confusion false positives. TUWIEN and RMIT
are the weakest site-transfer cases; NIBIO has relatively high recall but low
precision. A validation-only post-processing sweep is retained as a diagnostic
ablation and does not replace the accepted unfiltered test result.

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

The released SegmentAnyTree repository and container are external
dependencies. They are not copied into this repository.
