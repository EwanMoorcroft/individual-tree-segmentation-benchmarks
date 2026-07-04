# SegmentAnyTree

SegmentAnyTree is the current labelled-accuracy method for FOR-instance.

## Current experiment

The primary experiment trains from scratch on the supplied FOR-instance
development split. A fixed seed-42 partition assigns 16 plots to training and
five to internal validation. The 11 test plots are held out until model
selection and aligned-output checks are complete.

As of 4 July 2026, full training is running and the five-plot validation chain
is queued. No final held-out test accuracy is available.

Start with:

- [`docs/for_instance_benchmark.md`](docs/for_instance_benchmark.md) for the
  benchmark runbook;
- [`docs/running_full_training_20260704.md`](docs/running_full_training_20260704.md)
  for the current run provenance;
- [`slurm/README.md`](slurm/README.md) for the canonical submission sequence;
- [`configs/for_instance_training.yml`](configs/for_instance_training.yml) for
  the fixed training protocol; and
- [`examples/README.md`](examples/README.md) for the distinction between
  provisional diagnostics and accepted results.

The released SegmentAnyTree repository and container are external
dependencies. They are not copied into this repository.
