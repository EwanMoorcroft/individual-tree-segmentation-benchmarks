# Individual Tree Segmentation Benchmarks

[![Tests](https://github.com/EwanMoorcroft/individual-tree-segmentation-benchmarks/actions/workflows/tests.yml/badge.svg)](https://github.com/EwanMoorcroft/individual-tree-segmentation-benchmarks/actions/workflows/tests.yml)

Reproducible workflows for benchmarking individual-tree segmentation methods
on LiDAR point clouds. The repository contains public configurations, adapters,
evaluation code, scheduler workflows and small result summaries. Raw datasets,
model files, predictions, logs and external repositories are not included.

## Current status

Status last updated: 14 July 2026.

Three methods have completed comparable FOR-instance accuracy evaluation. All
five headline rows use the same 11 held-out plots, 323 reference instances,
point-aligned union-mask evaluation, IoU `>= 0.5` and maximum-cardinality
one-to-one matching.

| Method | Variant | Mean plot F1 | Micro F1 |
| --- | --- | ---: | ---: |
| SegmentAnyTree | Published pretrained | 0.453409 | 0.444245 |
| SegmentAnyTree | Fine-tuned on development data | 0.544679 | 0.531987 |
| TreeX | Unsupervised parameterised | 0.383108 | 0.362705 |
| TreeLearn | Published pretrained | 0.078944 | 0.098694 |
| TreeLearn | Fine-tuned on development data | 0.364685 | 0.331924 |

TLS2trees has completed a 16-plot FRDR prediction and operational benchmark.
FRDR has no individual-tree reference labels, so that work is not a
FOR-instance accuracy result and reports no precision, recall, F1 or IoU.

Historical and diagnostic results remain available but are excluded from the
headline table: the provisional coordinate-rematched SegmentAnyTree run, the
historical SegmentAnyTree `0.4825` mean plot F1 result, the TreeX
reference-labelled-mask `0.5222` result, and all TreeLearn development smokes,
overlap-affected runs and rejected validation checkpoints. TLS2trees on
FOR-instance and the Wytham Woods benchmarks remain candidates.

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
├── outputs/
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
- [`methods/treelearn/README.md`](methods/treelearn/README.md): completed
  published-pretrained and development-fine-tuned FOR-instance workflows,
  together with their preserved development and recovery routes.
- [`datasets/README.md`](datasets/README.md): dataset-level configuration and
  suitability.
- [`docs/protocols/for-instance.md`](docs/protocols/for-instance.md): fixed
  cross-method FOR-instance protocol.
- [`docs/evaluation_metrics.md`](docs/evaluation_metrics.md): operational and
  instance-accuracy definitions.
- [`docs/README.md`](docs/README.md): documentation index.
- [`CITATION.cff`](CITATION.cff): repository citation metadata.

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

## Completed, diagnostic and provisional results

The completed FRDR/TLS2trees per-plot operational summary is
[`methods/tls2trees/examples/tls2trees_frdr_prediction_summary.csv`](methods/tls2trees/examples/tls2trees_frdr_prediction_summary.csv).
It records 205,602,855 input points, 2,036 predicted tree files and one dropped
unknown point across 16 plots.

Files beginning with `provisional_released_checkpoint_` under
[`methods/segmentanytree/examples/`](methods/segmentanytree/examples/) document
the rejected coordinate-rematching route. They are not final SegmentAnyTree
accuracy results and must not be compared directly with the paper.

The completed historical SegmentAnyTree result is the aligned point-wise evaluation of
`sat_for_quicktune_to49_20260706_140730`. The canonical public evidence is the
[`final aggregate`](methods/segmentanytree/examples/sat_final_test_aligned_summary_sat_for_quicktune_to49_20260706_140730.csv)
and its
[`provenance manifest`](methods/segmentanytree/examples/sat_final_test_aligned_provenance_sat_for_quicktune_to49_20260706_140730.json).
The transferred failure-mode tables predate the final evaluation ID and are
clearly retained as diagnostic snapshots, not as alternate final metrics.
They indicate over-segmentation and background-confusion false positives, with
TUWIEN and RMIT as the weakest domain-transfer cases.

The consolidated public workbook is
[`outputs/sat_treex_benchmark_metrics/for_instance_method_benchmark_tracker.xlsx`](outputs/sat_treex_benchmark_metrics/for_instance_method_benchmark_tracker.xlsx).
It includes the five completed held-out rows shown above. Public-safe per-plot,
site and overall source tables independently reproduce every aggregate; raw
predictions remain off Git under the paths and hashes recorded in the
prediction-retention registry.

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

GitHub Actions runs the same synthetic suite, Python compilation and shell
syntax checks. Dependabot checks pinned Python packages and GitHub Actions
monthly; dependency changes remain reviewable pull requests.

## Safety boundary

Do not commit source point clouds, converted data, model files, checkpoints,
containers, predictions, scheduler logs or full benchmark output. Small CSV,
JSON and workbook summaries may be committed only when they contain no point
coordinates, private paths or credentials. Local prediction copies under
`local_outputs/` are backup artifacts and must be preserved until an explicit
archive or retention decision is made.
