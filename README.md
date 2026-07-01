# Individual Tree Segmentation Benchmarks

[![Tests](https://github.com/EwanMoorcroft/individual-tree-segmentation-benchmarks/actions/workflows/tests.yml/badge.svg)](https://github.com/EwanMoorcroft/individual-tree-segmentation-benchmarks/actions/workflows/tests.yml)

## Purpose

This repository collects reproducible workflows for benchmarking multiple
individual tree segmentation methods across multiple LiDAR datasets using the
University of Liverpool Barkla2 HPC system.

The first completed prediction benchmark uses TLS2trees instance segmentation
with the FRDR treeiso terrestrial laser scanning dataset. The current priority
is a full SegmentAnyTree accuracy benchmark on the labelled FOR-instance
dataset. Future configs can add other methods and datasets without creating
separate repositories.

No source datasets, converted point clouds, predictions, scheduler logs or
external method repositories are included.

## Current Status

- FRDR/TLS2trees: completed prediction and operational benchmark across 16
  plots; no reference instance accuracy is reported.
- FOR-instance/SegmentAnyTree: full 32-plot accuracy workflow prepared,
  beginning with `CULS/plot_1_annotated.las`; no prediction or accuracy result
  is reported yet. Split labels from `data_split_metadata.csv` are preserved.
- FOR-instance/TLS2trees: retained as a candidate compatibility test.
- Wytham Woods: downloaded and inspected; retained as a strong TLS reference
  dataset after plot-level reference reconstruction from per-tree files.

See the [benchmark registry](BENCHMARKS.md),
[dataset feasibility assessment](docs/dataset_feasibility.md), and
[labelled accuracy preparation plan](docs/labelled_accuracy_benchmark_plan.md)
for current and candidate dataset-method combinations.

The [SegmentAnyTree/FOR-instance runbook](docs/segmentanytree_for_instance_benchmark.md)
defines the next labelled benchmark. It evaluates semantic classes `4`, `5`
and `6` against positive `treeID` references. The earlier
[FOR-instance TLS2trees pilot](docs/for_instance_tls2trees_pilot.md) remains
available and uses its separate leaf-off class definition.

## Important Limitation

The FRDR LAZ files contain a `woods` scalar field for wood/non-wood
classification, not individual-tree reference instance labels. IoU, precision,
recall and F1 therefore cannot be computed from the FRDR LAZ files alone.

The workflow preserves predicted instances so these metrics can be calculated
later if suitable reference tree instance labels are supplied. The evaluator
exits with a clear error when no reference labels are provided.

## Known Limitations

- FRDR `woods` labels are semantic wood/non-wood labels, not tree-instance
  labels.
- `n_z` uses plot-local minimum Z, not terrain-normalised height.
- The patched TLS2trees instance script is a compatibility adaptation for newer
  pandas `groupby.apply` behaviour.
- The configured parameters reproduce a successful feasibility run and require
  visual validation before being treated as final benchmark parameters.
- Runtime and predicted tree count depend on plot size and point density.

## Completed FRDR/TLS2trees Run

All 16 configured FRDR plots completed on Barkla2. The run processed
205,602,855 input points and produced 2,036 predicted tree files containing
27,131,496 points. One `woods = 0.0` point in `NSpruce_plot2` was dropped under
the configured unknown-value policy.

`Mixed_plot1` was killed for exceeding a 32 GiB Slurm allocation, then completed
when rerun with 96 GiB; its recorded peak usage was 49.602968 GiB.

- [Completed benchmark results note](docs/frdr_tls2trees_results.md)
- [Per-plot prediction summary CSV](examples/tls2trees_frdr_prediction_summary.csv)
- [Evaluation metrics and reference-label requirements](docs/evaluation_metrics.md)
- [FOR-instance inventory example](examples/for_instance_inventory_summary.csv)

## FRDR Label Mapping

| FRDR value | Class | TLS2trees label |
| --- | --- | --- |
| `woods = 1` | wood | `3` |
| `woods = 2` | non-wood | `1` |

Unknown `woods` values are handled by `conversion.unknown_policy` in
[`configs/frdr_tls2trees_benchmark.yml`](configs/frdr_tls2trees_benchmark.yml).
The full benchmark uses `drop` because one plot, `NSpruce_plot2`, contains
`woods = 0.0`. Conversion metadata records all unknown and dropped point counts.

The current conversion calculates `n_z` by subtracting the retained
plot-local minimum Z. This is a documented feasibility approximation rather
than terrain normalisation.

The configured tile name, `001`, must remain numeric because TLS2trees parses
tile identifiers as integers.

## Barkla Environment

The tested Barkla environment used Python 3.12.10 with:

```bash
module purge
module load miniforge3/25.3.0-python3.12.10
source ~/fastscratch/venvs/treebench/bin/activate
```

The environment at `~/fastscratch/venvs/treebench` is a Python venv. Do not use
`conda activate` for it.

Install the protocol dependencies where required:

```bash
python -m pip install -r requirements.txt
```

## Expected Barkla Paths

| Purpose | Path |
| --- | --- |
| Project root | `~/scratch/tree-seg-benchmark` |
| FRDR dataset root | `~/data/datasets/frdr_treeiso` |
| FOR-instance dataset root | `~/data/datasets/for_instance/FORinstance_dataset` |
| TLS2trees checkout | `external/TLS2trees` |
| SegmentAnyTree checkout | `external/SegmentAnyTree` |
| Converted inputs | `data/interim/tls2trees/frdr_full/<plot_name>/` |
| SegmentAnyTree predictions | `data/predictions/segmentanytree/for_instance/<collection>/<plot_name>/` |
| SegmentAnyTree normalised predictions | `data/interim/segmentanytree/for_instance/<collection>/<plot_name>/normalised_predictions/` |
| Logs | `logs/<benchmark>/` |
| Metadata | `results/metadata/` |
| Tables | `results/tables/` |

The configured dataset path may resolve to a mounted Barkla2 filesystem path.
The inventory and conversion scripts do not modify source LAZ files.

## TLS2trees

- Repository: <https://github.com/tls-tools-ucl/TLS2trees>
- Tested commit: `ca12cb73b2c736d80b020e8025f8d975d42e6f01`
- Local checkout: `external/TLS2trees`

TLS2trees is not vendored in this repository. Clone and pin it separately:

```bash
mkdir -p external
git clone https://github.com/tls-tools-ucl/TLS2trees.git external/TLS2trees
git -C external/TLS2trees checkout ca12cb73b2c736d80b020e8025f8d975d42e6f01
```

The included
[`instance_patched.py`](scripts/methods/tls2trees_patched/instance_patched.py)
applies the documented `clstr` compatibility correction for newer pandas
`groupby.apply` behaviour without changing the external checkout.

## Repository Layout

The repository uses dataset-method configuration files and keeps reusable
responsibilities separate:

- `configs/`: dataset-method configurations and public-safe planning stubs;
- `scripts/data/`: dataset inspection, conversion and preparation utilities;
- `scripts/methods/`: wrappers around external segmentation methods;
- `scripts/evaluation/`: reusable metric and evaluator scripts;
- `scripts/slurm/`: scheduler workflows;
- `docs/`: benchmark results, feasibility notes, runbooks and metric definitions;
- `examples/`: small summaries and synthetic metadata only.

```text
.
├── README.md
├── BENCHMARKS.md
├── requirements.txt
├── configs/
│   ├── for_instance_accuracy_benchmark.yml
│   ├── for_instance_segmentanytree_benchmark.yml
│   ├── for_instance_tls2trees_accuracy.yml
│   ├── frdr_tls2trees_benchmark.yml
│   └── wytham_accuracy_benchmark.yml
├── docs/
│   ├── dataset_feasibility.md
│   ├── evaluation_metrics.md
│   ├── for_instance_tls2trees_pilot.md
│   ├── frdr_tls2trees_results.md
│   ├── labelled_accuracy_benchmark_plan.md
│   ├── segmentanytree_for_instance_benchmark.md
│   └── tls2trees_frdr_benchmark_runbook.md
├── examples/
│   ├── README.md
│   ├── for_instance_inventory_summary.csv
│   ├── frdr_dataset_inventory_example.csv
│   ├── segmentanytree_for_instance_plot_metrics_example.csv
│   ├── segmentanytree_for_instance_summary_example.csv
│   ├── tls2trees_conversion_metadata_example.json
│   ├── tls2trees_frdr_prediction_summary.csv
│   ├── tls2trees_prediction_summary_example.csv
│   └── tls2trees_run_metadata_example.json
├── scripts/
│   ├── data/
│   │   ├── convert_frdr_woods_to_tls2trees_ply.py
│   │   ├── convert_for_instance_to_tls2trees_ply.py
│   │   ├── inspect_for_instance_inventory.py
│   │   ├── inspect_frdr_dataset_inventory.py
│   │   └── select_for_instance_plot.py
│   ├── evaluation/
│   │   ├── instance_iou_f1.py
│   │   └── summarise_for_instance_segmentanytree_benchmark.py
│   ├── methods/
│   │   ├── normalise_segmentanytree_predictions.py
│   │   ├── run_segmentanytree_for_instance.py
│   │   ├── run_tls2trees_instance_for_plot.py
│   │   ├── run_tls2trees_for_instance_plot.py
│   │   ├── summarise_tls2trees_outputs.py
│   │   └── tls2trees_patched/
│   │       └── instance_patched.py
│   └── slurm/
│       ├── convert_frdr_to_tls2trees_array.sbatch
│       ├── convert_for_instance_tls2trees_pilot.sbatch
│       ├── evaluate_for_instance_tls2trees_pilot.sbatch
│       ├── inspect_for_instance_inventory.sbatch
│       ├── inspect_frdr_inventory.sbatch
│       ├── normalise_segmentanytree_for_instance_array.sbatch
│       ├── evaluate_segmentanytree_for_instance_array.sbatch
│       ├── run_segmentanytree_for_instance_array.sbatch
│       ├── run_segmentanytree_for_instance_pilot.sbatch
│       ├── run_tls2trees_for_instance_pilot.sbatch
│       ├── run_tls2trees_frdr_array.sbatch
│       ├── summarise_segmentanytree_for_instance.sbatch
│       └── summarise_tls2trees_frdr_outputs.sbatch
├── src/
│   └── benchmark/
│       ├── __init__.py
│       └── ply_io.py
└── tests/
    ├── test_segmentanytree_for_instance_workflow.py
    ├── test_for_instance_tls2trees_workflow.py
    └── test_frdr_tls2trees_workflow.py
```

## Public-Safe Results And Examples

The [`examples/`](examples/) directory contains the completed FRDR per-plot
summary, a small FOR-instance inventory extract, and synthetic schema examples.
No coordinates, point clouds, prediction files or logs are included.

## Recommended SegmentAnyTree Pilot

Inspect the external SegmentAnyTree checkout and configure its command before
submitting work. The pilot script performs a dry-run by default:

```bash
cd ~/scratch/tree-seg-benchmark
mkdir -p logs/segmentanytree_for_instance
git -C external/SegmentAnyTree rev-parse HEAD
sed -n '1,240p' external/SegmentAnyTree/README.md
sbatch scripts/slurm/inspect_for_instance_inventory.sbatch
sbatch scripts/slurm/run_segmentanytree_for_instance_pilot.sbatch
```

The pilot fails safely while `method.command_template` is unset. After
inspection, follow the prediction, normalisation, evaluation and
dependency-chained full-array commands in the
[SegmentAnyTree/FOR-instance runbook](docs/segmentanytree_for_instance_benchmark.md).
GPU resources must be added only when the installed method and Barkla
configuration confirm they are required.

## Recommended Staged Execution

1. Inspect the installed SegmentAnyTree version, command, dependencies and
   output schema.
2. Run and review the FOR-instance inventory and split assignments.
3. Dry-run and then execute `CULS/plot_1_annotated.las`.
4. Normalise and evaluate the pilot, then fix the final evaluation settings.
5. Submit prediction, normalisation and evaluation arrays with dependencies.
6. Summarise plot, collection and split accuracy only after all jobs finish.

Do not train or tune on the test split. External NEWFOR results should be
compared only when metrics, class filters, IoU thresholds and coordinate
tolerances are compatible.

## Outputs

The SegmentAnyTree/FOR-instance workflow writes:

- method-specific predictions outside Git;
- one normalised XYZ PLY per predicted tree;
- per-plot run metadata JSON, including runtime and peak memory when available;
- per-plot evaluation, matched-pair and unmatched-instance tables;
- plot, collection and split benchmark summary tables.

These outputs are intentionally excluded from Git.

## Version-Control Exclusions

The `.gitignore` excludes:

- raw and derived data under `Datasets/` and `data/`;
- the external TLS2trees checkout;
- scheduler and method logs;
- result metadata, tables and prediction artefacts;
- LAS, LAZ, PLY, NumPy and image files;
- virtual environments, logs, system files and Python/test caches.

No raw data or predictions are included in this repository.

## Tests

The test suite uses small synthetic point clouds and does not require any source
dataset or external method checkout:

```bash
python -m pip install pytest
python -m pytest
```

## Data And Citation

Each dataset must be obtained from its original source and must not be
redistributed through this repository. Follow each dataset's licence and
citation requirements. Cite every benchmarked method and its associated
research according to the upstream project guidance.

## Licence

No licence has been selected yet; contact the repository owner before reuse.
