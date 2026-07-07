# SegmentAnyTree

SegmentAnyTree is the current labelled-accuracy method for FOR-instance.

## Current experiment

The primary experiment trains from scratch on the supplied FOR-instance
development split. A fixed seed-42 partition assigns 16 plots to training and
five to internal validation. The 11 test plots are held out until model
selection and aligned-output checks are complete.

As of 6 July 2026, the epoch-47 checkpoint has a mean aligned F1 of `0.513`
across the five development validation plots, compared with `0.458` at epoch
45. A two-epoch continuation to epoch 49 is running. The 11 held-out test plots
remain untouched, and no final test accuracy is available.

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
