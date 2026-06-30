# Benchmark Registry

This repository is structured to hold multiple individual tree segmentation
benchmarks. Each benchmark should provide a reproducible configuration, input
adapter, method runner, scheduler workflow, metadata outputs and focused tests.

## Benchmark Registry

| Dataset | Method | Status | Configuration or note |
| --- | --- | --- | --- |
| FRDR treeiso TLS | TLS2trees | Prediction benchmark completed | [`frdr_tls2trees_benchmark.yml`](configs/frdr_tls2trees_benchmark.yml) |
| FOR-instance | SegmentAnyTree | Candidate accuracy benchmark | [`for_instance_accuracy_benchmark.yml`](configs/for_instance_accuracy_benchmark.yml) |
| FOR-instance | TLS2trees | Leaf-off accuracy pilot workflow implemented; not yet run | [`for_instance_tls2trees_accuracy.yml`](configs/for_instance_tls2trees_accuracy.yml) |
| FOR-instance | TreeLearn or another deep learning method | Candidate accuracy benchmark | Respect the supplied development/test split |
| Wytham Woods | TLS2trees | Candidate TLS accuracy benchmark | [`wytham_accuracy_benchmark.yml`](configs/wytham_accuracy_benchmark.yml) |
| Wytham Woods | SegmentAnyTree | Candidate accuracy benchmark | Plot-level input reconstruction required |
| Wytham Woods | Traditional TLS method | Candidate baseline | Plot-level input reconstruction required |

The FRDR LAZ files do not contain individual-tree reference instance labels.
The TLS2trees workflow therefore preserves predictions and operational metadata
but does not report IoU/F1 without an external instance reference.

No candidate accuracy row indicates a completed method run or a reported
accuracy result. Dataset readiness and remaining preprocessing are documented
in [`docs/dataset_feasibility.md`](docs/dataset_feasibility.md).

The first labelled workflow uses FOR-instance
`CULS/plot_1_annotated.las` with TLS2trees. See
[`docs/for_instance_tls2trees_pilot.md`](docs/for_instance_tls2trees_pilot.md).

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
