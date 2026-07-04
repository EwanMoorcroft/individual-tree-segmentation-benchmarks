# Individual Tree Segmentation Benchmarks

[![Tests](https://github.com/EwanMoorcroft/individual-tree-segmentation-benchmarks/actions/workflows/tests.yml/badge.svg)](https://github.com/EwanMoorcroft/individual-tree-segmentation-benchmarks/actions/workflows/tests.yml)

Reproducible workflows for benchmarking individual-tree segmentation methods
on LiDAR point clouds. The repository contains public configurations, adapters,
evaluation code, scheduler workflows and small result summaries. Raw datasets,
model files, predictions, logs and external repositories are not included.

## Current status

Status last updated: 4 July 2026.

- **TLS2trees on FRDR treeiso:** the 16-plot prediction and operational
  benchmark is complete. FRDR does not contain individual-tree reference
  labels, so this is not an accuracy benchmark and no precision, recall, F1 or
  IoU is reported.
- **SegmentAnyTree on FOR-instance:** the earlier 32-plot released-checkpoint
  run is retained only as a provisional engineering diagnostic because its
  final exports failed point-correspondence checks.
- **SegmentAnyTree retrained on FOR-instance:** full training on the fixed
  16-plot development training split is running on Barkla. Validation on the
  five fixed development validation plots is queued. The 11 held-out test
  plots remain untouched and final accuracy results are not yet available.
- **Other combinations:** TLS2trees on FOR-instance and the Wytham Woods
  benchmarks remain candidates rather than completed studies.

The [benchmark registry](BENCHMARKS.md) is the short status index. Detailed
method information is under [`methods/`](methods/), dataset contracts are under
[`datasets/`](datasets/), and cross-method rules are under [`docs/`](docs/).

## Repository layout

The public repository is organised primarily by method:

```text
.
├── methods/
│   ├── segmentanytree/
│   │   ├── configs/
│   │   ├── docs/
│   │   ├── examples/
│   │   ├── scripts/
│   │   └── slurm/
│   └── tls2trees/
│       ├── configs/
│       ├── docs/
│       ├── examples/
│       ├── scripts/
│       └── slurm/
├── datasets/
│   ├── for-instance/
│   └── wytham-woods/
├── shared/evaluation/
├── src/benchmark/
├── docs/
│   ├── plans/
│   └── protocols/
└── tests/
```

- [`methods/segmentanytree/README.md`](methods/segmentanytree/README.md):
  current FOR-instance training, inference and evaluation workflow.
- [`methods/tls2trees/README.md`](methods/tls2trees/README.md): completed FRDR
  workflow and candidate FOR-instance work.
- [`datasets/README.md`](datasets/README.md): dataset-level configuration and
  suitability.
- [`docs/protocols/for-instance.md`](docs/protocols/for-instance.md): fixed
  cross-method FOR-instance protocol.
- [`docs/evaluation_metrics.md`](docs/evaluation_metrics.md): operational and
  instance-accuracy definitions.

Generated output paths such as `data/`, `results/` and `logs/` remain at the
repository root at runtime. They are ignored by Git and are not part of the
source-code hierarchy.

## Accepted and provisional results

The completed FRDR/TLS2trees per-plot operational summary is
[`methods/tls2trees/examples/tls2trees_frdr_prediction_summary.csv`](methods/tls2trees/examples/tls2trees_frdr_prediction_summary.csv).
It records 205,602,855 input points, 2,036 predicted tree files and one dropped
unknown point across 16 plots.

Files beginning with `provisional_released_checkpoint_` under
[`methods/segmentanytree/examples/`](methods/segmentanytree/examples/) document
the rejected coordinate-rematching route. They are not final SegmentAnyTree
accuracy results and must not be compared directly with the paper.

The accepted SegmentAnyTree result will be published only after:

1. training completes on the 16 development training plots;
2. a checkpoint is selected using only the five development validation plots;
3. aligned point-wise validation outputs pass the integrity checks;
4. the frozen checkpoint is run once on all 11 held-out test plots; and
5. paper-compatible and harmonised one-to-one metrics are both reported.

## Barkla environment

The repository utilities use:

```bash
module purge
module load miniforge3/25.3.0-python3.12.10
source ~/fastscratch/venvs/treebench/bin/activate
```

SegmentAnyTree runs through Apptainer 1.3.6 on GPU nodes. TLS2trees and
SegmentAnyTree are external dependencies and must be checked out separately
under `external/`; they are not vendored here.

The expected Barkla repository root is `~/scratch/tree-seg-benchmark`. Dataset
and fast-scratch paths are documented in the method configurations rather than
hard-coded in Python modules.

## Local verification

Install the pinned public utility dependencies and run the synthetic suite:

```bash
python -m pip install -r requirements.txt
python -m pytest
```

The tests do not require Barkla, GPUs, Apptainer, external repositories or real
point-cloud data.

## Safety boundary

Do not commit source point clouds, converted data, model files, checkpoints,
containers, predictions, scheduler logs or full benchmark output. Small CSV,
JSON and workbook summaries may be committed only when they contain no point
coordinates, private paths or credentials.
