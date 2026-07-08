# Benchmark Registry

This repository is structured to hold multiple individual tree segmentation
benchmarks. Each benchmark should provide a reproducible configuration, input
adapter, method runner, scheduler workflow, metadata outputs and focused tests.

## Benchmark Registry

| Dataset | Method | Status | Configuration or note |
| --- | --- | --- | --- |
| FRDR treeiso TLS | TLS2trees | Prediction benchmark completed | [`frdr_benchmark.yml`](methods/tls2trees/configs/frdr_benchmark.yml) |
| FOR-instance | SegmentAnyTree released checkpoint | Provisional inference-only run completed; export audit failed | [`for_instance_benchmark.yml`](methods/segmentanytree/configs/for_instance_benchmark.yml) |
| FOR-instance | SegmentAnyTree retrained from development split | Completed; accepted checkpoint `sat_for_quicktune_to49_20260706_140730`, test mean F1 0.480 | [`training_progress_20260706.md`](methods/segmentanytree/docs/training_progress_20260706.md) |
| FOR-instance | TreeX (`pointtree`) | Completed on exact-path local subset; strict test F1 0.402, labelled-mask test F1 0.522 | [`for_instance_benchmark.yml`](methods/treex/configs/for_instance_benchmark.yml) |
| FOR-instance | TLS2trees | Candidate compatibility test | [`for_instance_accuracy.yml`](methods/tls2trees/configs/for_instance_accuracy.yml) |
| FOR-instance | TreeLearn or another deep learning method | Candidate accuracy benchmark | Respect the supplied development/test split |
| Wytham Woods | TLS2trees | Candidate TLS accuracy benchmark | [`benchmark.yml`](datasets/wytham-woods/benchmark.yml) |
| Wytham Woods | SegmentAnyTree | Candidate accuracy benchmark | Plot-level input reconstruction required |
| Wytham Woods | Traditional TLS method | Candidate baseline | Plot-level input reconstruction required |
| NEWFOR | SegmentAnyTree | External comparison dataset; not implemented here | Add only through a separate documented dataset config |

The FRDR LAZ files do not contain individual-tree reference instance labels.
The TLS2trees workflow therefore preserves predictions and operational metadata
but does not report IoU/F1 without an external instance reference.

No candidate accuracy row indicates a completed method run or a reported
accuracy result. Dataset readiness and remaining preprocessing are documented
in [`docs/dataset_feasibility.md`](docs/dataset_feasibility.md).

The first SegmentAnyTree workflow completed inference for all 32 FOR-instance
LAS files with the released checkpoint. Its coordinate-rematched metrics are
provisional because they neither preserve point alignment nor represent a
model trained under the local development/test protocol. The corrected primary
experiment trains from scratch on FOR-instance development data, selects the
checkpoint on an internal development validation split and evaluates the
held-out test split once. The accepted checkpoint is
`sat_for_quicktune_to49_20260706_140730`, with mean aligned F1 `0.537` on the
five development validation plots and `0.480` on the 11 held-out test plots.
The later `to55` continuation is rejected because validation fell to `0.451`.
See the
[`shared protocol`](docs/protocols/for-instance.md),
[`runbook`](methods/segmentanytree/docs/for_instance_benchmark.md) and
[`training progress`](methods/segmentanytree/docs/training_progress_20260706.md).

## Adding A Benchmark

Additions should include:

1. A config named for the dataset and method.
2. A dataset inspection or conversion adapter where required.
3. A wrapper around the upstream method rather than a reimplementation.
4. Slurm jobs for inspection, preparation, prediction and summarisation.
5. Metadata recording for inputs, versions, commands, runtime and outputs.
6. Evaluation only when suitable reference labels are available.
7. Synthetic tests that do not require private or large datasets.
8. A concise runbook documenting environment, assumptions and limitations.

Raw data, external repositories, predictions, logs and large derived files must
remain outside Git.
