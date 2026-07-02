# Benchmark Registry

This repository is structured to hold multiple individual tree segmentation
benchmarks. Each benchmark should provide a reproducible configuration, input
adapter, method runner, scheduler workflow, metadata outputs and focused tests.

## Benchmark Registry

| Dataset | Method | Status | Configuration or note |
| --- | --- | --- | --- |
| FRDR treeiso TLS | TLS2trees | Prediction benchmark completed | [`frdr_tls2trees_benchmark.yml`](configs/frdr_tls2trees_benchmark.yml) |
| FOR-instance | SegmentAnyTree | Full 32-plot prediction and evaluation benchmark completed | [`for_instance_segmentanytree_benchmark.yml`](configs/for_instance_segmentanytree_benchmark.yml) |
| FOR-instance | TLS2trees | Candidate compatibility test | [`for_instance_tls2trees_accuracy.yml`](configs/for_instance_tls2trees_accuracy.yml) |
| FOR-instance | TreeLearn or another deep learning method | Candidate accuracy benchmark | Respect the supplied development/test split |
| Wytham Woods | TLS2trees | Candidate TLS accuracy benchmark | [`wytham_accuracy_benchmark.yml`](configs/wytham_accuracy_benchmark.yml) |
| Wytham Woods | SegmentAnyTree | Candidate accuracy benchmark | Plot-level input reconstruction required |
| Wytham Woods | Traditional TLS method | Candidate baseline | Plot-level input reconstruction required |
| NEWFOR | SegmentAnyTree | External comparison dataset; not implemented here | Add only through a separate documented dataset config |

The FRDR LAZ files do not contain individual-tree reference instance labels.
The TLS2trees workflow therefore preserves predictions and operational metadata
but does not report IoU/F1 without an external instance reference.

No candidate accuracy row indicates a completed method run or a reported
accuracy result. Dataset readiness and remaining preprocessing are documented
in [`docs/dataset_feasibility.md`](docs/dataset_feasibility.md).

The primary labelled workflow has completed SegmentAnyTree prediction,
normalisation and evaluation for all 32 FOR-instance LAS files. See the
[`runbook`](docs/segmentanytree_for_instance_benchmark.md) and
[`results note`](docs/segmentanytree_for_instance_results.md). The earlier
TLS2trees pilot scaffolding remains available as a compatibility test.

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
