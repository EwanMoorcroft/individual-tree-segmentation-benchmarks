# Individual Tree Segmentation Benchmarks

[![Tests](https://github.com/EwanMoorcroft/individual-tree-segmentation-benchmarks/actions/workflows/tests.yml/badge.svg)](https://github.com/EwanMoorcroft/individual-tree-segmentation-benchmarks/actions/workflows/tests.yml)

Reproducible workflows for benchmarking individual-tree segmentation methods
on LiDAR point clouds. The repository contains public configurations, adapters,
evaluation code, scheduler workflows and small result summaries. Raw datasets,
model files, predictions, logs and external repositories are not included.

## Current status

Status last updated: 9 July 2026.

- **TLS2trees on FRDR treeiso:** the 16-plot prediction and operational
  benchmark is complete. FRDR does not contain individual-tree reference
  labels, so this is not an accuracy benchmark and no precision, recall, F1 or
  IoU is reported.
- **SegmentAnyTree on FOR-instance:** the earlier 32-plot released-checkpoint
  run is retained only as a provisional engineering diagnostic because its
  final exports failed point-correspondence checks.
- **SegmentAnyTree retrained on FOR-instance:** the development-selected
  checkpoint `sat_for_quicktune_to49_20260706_140730` is the accepted SAT run.
  It reports mean aligned F1 `0.537` on the five validation plots and `0.4825`
  on the 11 held-out test plots. The later `to55` continuation is rejected
  because validation fell to `0.451`.
- **TreeX on FOR-instance:** the unsupervised `pointtree` TreeX benchmark is
  complete on the exact-path local subset of 21 development and 11 test plots.
  The cautious headline test result is strict F1 `0.402`; labelled-mask test
  F1 is `0.522`.
- **TreeLearn on FOR-instance:** a guarded one-plot development smoke route is
  scaffolded but has not been run. It is not an accuracy benchmark.
- **Other combinations:** TLS2trees on FOR-instance, full TreeLearn evaluation
  and the Wytham Woods benchmarks remain candidates rather than completed
  studies.

The [benchmark registry](BENCHMARKS.md) is the short status index. Each
completed, provisional or candidate row records a dataset slug, method slug,
run label, training mode declaration, evaluation mode, status and evidence
file. Detailed method information is under [`methods/`](methods/), dataset
contracts are under [`datasets/`](datasets/), and cross-method rules are under
[`docs/`](docs/).

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
│   ├── tls2trees/
│   │   ├── configs/
│   │   ├── docs/
│   │   ├── examples/
│   │   ├── scripts/
│   │   └── slurm/
│   ├── treelearn/
│   │   ├── configs/
│   │   ├── docs/
│   │   ├── examples/
│   │   ├── scripts/
│   │   └── slurm/
│   └── treex/
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
- [`methods/treex/README.md`](methods/treex/README.md): completed FOR-instance
  TreeX benchmark using the `pointtree` API.
- [`methods/tls2trees/README.md`](methods/tls2trees/README.md): completed FRDR
  workflow and candidate FOR-instance work.
- [`methods/treelearn/README.md`](methods/treelearn/README.md): guarded
  one-plot FOR-instance smoke route.
- [`datasets/README.md`](datasets/README.md): dataset-level configuration and
  suitability.
- [`docs/protocols/for-instance.md`](docs/protocols/for-instance.md): fixed
  cross-method FOR-instance protocol.
- [`docs/evaluation_metrics.md`](docs/evaluation_metrics.md): operational and
  instance-accuracy definitions.

Generated output paths such as `data/`, `results/` and `logs/` remain at the
repository root at runtime. They are ignored by Git and are not part of the
source-code hierarchy.

New method folders should keep the same public shape before any Barkla run is
described as comparable:

```text
methods/<method_slug>/
├── README.md
├── configs/
├── docs/
├── examples/
├── scripts/
└── slurm/
```

Where an established method already uses more specific filenames, its README
must name the current equivalents for preparation, inference, prediction
adaptation, summarisation and evaluation.

## Accepted and provisional results

The completed FRDR/TLS2trees per-plot operational summary is
[`methods/tls2trees/examples/tls2trees_frdr_prediction_summary.csv`](methods/tls2trees/examples/tls2trees_frdr_prediction_summary.csv).
It records 205,602,855 input points, 2,036 predicted tree files and one dropped
unknown point across 16 plots.

Files beginning with `provisional_released_checkpoint_` under
[`methods/segmentanytree/examples/`](methods/segmentanytree/examples/) document
the rejected coordinate-rematching route. They are not final SegmentAnyTree
accuracy results and must not be compared directly with the paper.

The accepted SegmentAnyTree result is the aligned point-wise evaluation of
`sat_for_quicktune_to49_20260706_140730`. Failure-mode audits show that the
main limitation is over-segmentation and background-confusion false positives,
with TUWIEN and RMIT as the weakest domain-transfer cases. A validation-only
post-processing sweep is retained as a diagnostic ablation; it does not replace
the unfiltered held-out test score.

## Barkla environment

The repository utilities use:

```bash
module purge
module load miniforge3/25.3.0-python3.12.10
source ~/fastscratch/venvs/treebench/bin/activate
```

SegmentAnyTree runs through Apptainer 1.3.6 on GPU nodes. SegmentAnyTree,
TLS2trees, TreeLearn and TreeX/`pointtree` are external dependencies and are
not vendored here.

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
coordinates, private paths or credentials. Local prediction copies under
`local_outputs/` are backup artifacts and must be preserved until an explicit
archive or retention decision is made.
