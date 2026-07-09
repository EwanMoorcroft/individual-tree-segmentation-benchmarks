# SegmentAnyTree training progress through 8 July 2026

This note records development-only training and validation progress for the
corrected FOR-instance experiment and the final held-out test evaluation for
the selected checkpoint.

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
| Two-epoch continuation | `9668753` | Epoch 49 | Completed | Mean F1 `0.5371` |
| Six-epoch continuation | `9696965` | Epoch 55 | `05:14:54` | Rejected; mean F1 `0.4505` |

The selected run ID is `sat_for_quicktune_to49_20260706_140730`. The later
`sat_for_quicktune_to55_20260707_214305` continuation is rejected because the
development validation score regressed.

## Selected checkpoint

The epoch-49 checkpoint is selected using the five aligned development
validation plots:

| Plot | Epoch 49 F1 |
| --- | ---: |
| `CULS/plot_1_annotated` | 0.8571 |
| `NIBIO/plot_11_annotated` | 0.6279 |
| `NIBIO/plot_19_annotated` | 0.4583 |
| `NIBIO/plot_2_annotated` | 0.5747 |
| `TUWIEN/train` | 0.1674 |

The selected checkpoint has validation mean F1 `0.5371`, minimum F1 `0.1674`
and maximum F1 `0.8571`. Held-out test evaluation was then run for the
selected checkpoint using aligned instance and semantic outputs, giving
11-plot mean F1 `0.4825`, mean precision `0.3807` and mean recall `0.6954`.
The evaluated checkpoint SHA-256 is
`9b871b15ac61589ea27c507e054ee66d3f543caa01fed9a5b790e4ce97bcecea`.

The validation-only post-processing sweep selected
`min_predicted_instance_points=5000`, with validation mean F1 `0.5514`.
Because this is a diagnostic threshold and did not provide a robust
replacement for the unfiltered result, the headline SAT result remains the
unfiltered epoch-49 held-out test score.

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

## Failure-mode interpretation

The accepted checkpoint is not limited by point correspondence or a broken
metric. Matched instances have high IoU where a match succeeds. The main
failure mode is over-segmentation and background-confusion false positives.
NIBIO has relatively high recall with low precision, while TUWIEN and RMIT are
weak site-transfer cases with many missed trees. The epoch-55 continuation
worsened validation, so further blind training is not a supported improvement
path.

The later `fine_tuned_on_dev` follow-up
`segmentanytree_for-instance_fine_tuned_on_dev_20260708_215054_full` is not an
accepted result. A held-out smoke audit found millions of tree-semantic points
but zero accepted instance predictions in the aligned instance output. The
accepted SAT result therefore remains the unfiltered
`sat_for_quicktune_to49_20260706_140730` checkpoint.
