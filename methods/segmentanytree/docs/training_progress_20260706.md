# SegmentAnyTree training progress through 6 July 2026

This note records development-only training and validation progress for the
corrected FOR-instance experiment. It does not contain a held-out test result.

## Fixed protocol

- Training data: 16 FOR-instance development plots
- Checkpoint selection: five fixed development validation plots
- Held-out test data: 11 plots, untouched
- Internal validation seed: `42`
- Primary external metric: aligned point-wise F1 at IoU `0.5`
- Model: `PointGroup-PAPER`
- Scenario: ULS-only retraining from development data

## Training history

The first full training job, `9628896`, produced a valid epoch-30 checkpoint
but was cancelled after the next epoch entered a slow MeanShift clustering
path. Resume support and a bounded MeanShift worker configuration were then
added.

| Run | Job | Checkpoint | Training elapsed | Development validation |
| --- | ---: | ---: | ---: | --- |
| Initial full run | `9628896` | Epoch 30 | `10:27:34` before cancellation | Not selected |
| Resume to epoch 45 | `9651265` | Epoch 45 | About 19 hours | Mean F1 `0.4580` |
| Two-epoch continuation | `9664203` | Epoch 47 | `01:43:26` | Mean F1 `0.5127` |
| Two-epoch continuation | `9668753` | Target epoch 49 | Running when last recorded | Pending |

The epoch-47 run ID is
`sat_for_quicktune_to47_20260706_084834`.

## Epoch-47 validation

The five aligned development validation plots all improved relative to the
epoch-45 checkpoint:

| Plot | Epoch 45 F1 | Epoch 47 F1 | Change |
| --- | ---: | ---: | ---: |
| `CULS/plot_1_annotated` | 0.4444 | 0.6154 | +0.1709 |
| `NIBIO/plot_11_annotated` | 0.6582 | 0.6667 | +0.0084 |
| `NIBIO/plot_19_annotated` | 0.4731 | 0.4948 | +0.0217 |
| `NIBIO/plot_2_annotated` | 0.5714 | 0.6067 | +0.0353 |
| `TUWIEN/train` | 0.1429 | 0.1797 | +0.0368 |

The epoch-47 mean, minimum and maximum F1 are `0.5127`, `0.1797` and
`0.6667`. These values are development-validation metrics and must not be
reported as final test performance.

## Evaluation correction

The first aligned evaluation incorrectly reported F1 `0.0`. The internal
semantic reference array was degenerate, containing only class `0`, while the
aligned instance reference array retained background label `1` and valid tree
IDs. The evaluator now derives the reference tree mask from non-background
instance IDs when this specific degenerate semantic condition is present.

The correction changes reference-mask interpretation only. It does not modify
predictions, matching policy or the IoU threshold. Synthetic regression tests
cover the fallback, and both paper-compatible and harmonised evaluations
produce the same five-plot F1 values for these checkpoints.

## Resource profile

The current two-epoch continuation uses one L40S GPU, 16 CPU cores, 64 GB RAM,
batch size `8`, one MeanShift process and one OpenMP thread. During job
`9668753`, the GPU was observed at 100% utilization with approximately
23.9 GB of 46.1 GB memory used; maximum resident CPU memory was approximately
26.8 GB. This profile keeps the GPU busy without oversubscribing the
CPU-intensive MeanShift stage.

Additional nodes or GPUs cannot accelerate the current upstream trainer
without distributed-training changes. Short continuation runs provide the
lowest-risk checkpoint search while keeping each train-and-validation cycle
within a practical daily budget.

## Decision gate

After the epoch-49 validation finishes:

1. compare all five plot-level F1 values with epoch 47;
2. select a checkpoint using development validation only;
3. record the selected checkpoint SHA-256 and complete run metadata;
4. freeze inference and evaluation settings; and
5. run the 11 held-out test plots once.

No test submission should occur while checkpoint selection remains active.
