# Benchmark Registry

This repository is structured to hold multiple individual tree segmentation
benchmarks. Each benchmark should provide a reproducible configuration, input
adapter, method runner, scheduler workflow, metadata outputs and focused tests.

## Current Benchmarks

| Dataset | Method | Status | Configuration | Runbook | Results |
| --- | --- | --- | --- | --- | --- |
| FRDR treeiso TLS | TLS2trees | Prediction benchmark completed | `configs/frdr_tls2trees_benchmark.yml` | `docs/tls2trees_frdr_benchmark_runbook.md` | `docs/frdr_tls2trees_results.md` |

The FRDR LAZ files do not contain individual-tree reference instance labels.
The TLS2trees workflow therefore preserves predictions and operational metadata
but does not report IoU/F1 without an external instance reference.

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
