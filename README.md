# Individual Tree Segmentation Benchmarks

[![Tests](https://github.com/EwanMoorcroft/individual-tree-segmentation-benchmarks/actions/workflows/tests.yml/badge.svg)](https://github.com/EwanMoorcroft/individual-tree-segmentation-benchmarks/actions/workflows/tests.yml)

## Purpose

This repository collects reproducible workflows for benchmarking multiple
individual tree segmentation methods across multiple LiDAR datasets using the
University of Liverpool Barkla2 HPC system.

The completed workflows cover a TLS2trees prediction benchmark on the FRDR
treeiso terrestrial laser scanning dataset and a labelled SegmentAnyTree
accuracy benchmark on FOR-instance. Future configs can add other methods and
datasets without creating separate repositories.

No source datasets, converted point clouds, predictions, scheduler logs or
external method repositories are included.

## Current Status

- FRDR/TLS2trees: completed prediction and operational benchmark across 16
  plots; no reference instance accuracy is reported.
- FOR-instance/SegmentAnyTree: prediction, normalisation and labelled
  evaluation completed for all 32 annotated LAS files. The results cover all
  five collections and preserve the supplied development/test split labels.
- FOR-instance/TLS2trees: retained as a candidate compatibility test.
- Wytham Woods: downloaded and inspected; retained as a strong TLS reference
  dataset after plot-level reference reconstruction from per-tree files.

See the [benchmark registry](BENCHMARKS.md),
[SegmentAnyTree/FOR-instance runbook](docs/segmentanytree_for_instance_benchmark.md),
[full SegmentAnyTree results](docs/segmentanytree_for_instance_results.md),
[evaluation definitions](docs/evaluation_metrics.md),
[dataset feasibility assessment](docs/dataset_feasibility.md), and
[labelled accuracy plan](docs/labelled_accuracy_benchmark_plan.md) for current
and candidate dataset-method combinations.

The full SegmentAnyTree benchmark evaluates semantic classes `4`, `5` and `6`
against positive `treeID` references. Across 32 plots it evaluated 1,130
reference trees and 2,532 predictions, with 376 true positives, 2,156 false
positives and 754 false negatives. Micro precision was 0.148499, micro recall
0.332743 and micro F1 0.205352 at a 0.5 IoU threshold. The mean IoU across the
376 matched pairs was 0.726375. Collection-level performance varied strongly;
the NIBIO results require further investigation before drawing method-level
conclusions. The earlier
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

## Completed FOR-instance/SegmentAnyTree Run

All 32 annotated LAS files completed prediction, normalisation and one-to-one
instance evaluation. The benchmark processed the supplied 21 development and
11 test plots without using the test split for parameter selection. Cumulative
per-plot runtime was 13,430 seconds and the maximum recorded task memory was
9.608 GiB. Runtime is cumulative across array tasks rather than elapsed wall
time.

- [Results and interpretation](docs/segmentanytree_for_instance_results.md)
- [Supervisor workbook](examples/segmentanytree_for_instance_full_results.xlsx)
- [Per-plot metrics](examples/segmentanytree_for_instance_full_plot_metrics.csv)
- [Overall summary](examples/segmentanytree_for_instance_full_summary.csv)
- [Collection summaries](examples/segmentanytree_for_instance_full_summary_by_collection.csv)
- [Split summaries](examples/segmentanytree_for_instance_full_summary_by_split.csv)
- [Matched instance pairs](examples/segmentanytree_for_instance_full_matches.csv)

These public-safe files contain aggregate measurements and identifiers only.
Raw point clouds, predictions, full metadata and scheduler logs remain ignored.

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
| SegmentAnyTree SIF | `~/scratch/containers/segment-any-tree_latest.sif` |
| SegmentAnyTree repaired userbase | `~/fastscratch/segmentanytree_pyuser_v1` |
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
- `examples/`: small public-safe summaries, pilot aggregates and synthetic
  metadata.

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
│   ├── segmentanytree_barkla_debug_log.md
│   ├── segmentanytree_for_instance_benchmark.md
│   ├── segmentanytree_for_instance_results.md
│   └── tls2trees_frdr_benchmark_runbook.md
├── examples/
│   ├── README.md
│   ├── for_instance_inventory_summary.csv
│   ├── frdr_dataset_inventory_example.csv
│   ├── segmentanytree_for_instance_plot_metrics_example.csv
│   ├── segmentanytree_for_instance_pilot_metrics.csv
│   ├── segmentanytree_for_instance_pilot_status.json
│   ├── segmentanytree_for_instance_full_results.xlsx
│   ├── segmentanytree_for_instance_full_plot_metrics.csv
│   ├── segmentanytree_for_instance_full_summary.csv
│   ├── segmentanytree_for_instance_full_summary_by_collection.csv
│   ├── segmentanytree_for_instance_full_summary_by_split.csv
│   ├── segmentanytree_for_instance_full_matches.csv
│   ├── segmentanytree_for_instance_full_inventory.csv
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
│   │   ├── record_segmentanytree_run.py
│   │   ├── run_segmentanytree_for_instance.py
│   │   ├── segmentanytree_runtime_patches/
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
│       ├── install_segmentanytree_python_stack.sbatch
│       ├── run_segmentanytree_for_instance_pilot_apptainer.sbatch
│       ├── test_segmentanytree_apptainer.sbatch
│       ├── test_segmentanytree_python_stack_repair.sbatch
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
summary, the public-safe full SegmentAnyTree result tables and workbook, the
historical pilot record, and synthetic schema examples. No coordinates, point
clouds, prediction files or logs are included.

## SegmentAnyTree Pilot And Full Run

The supported Barkla route uses Apptainer 1.3.6 on `gpu-l40s`. The SIF is
created from `docker://maciekwielgosz/segment-any-tree:latest` outside Git.
Install and validate the repaired container userbase before inference:

```bash
cd ~/scratch/tree-seg-benchmark
mkdir -p logs/segmentanytree_for_instance
SAT_CONTAINER=$(sbatch --parsable \
  scripts/slurm/test_segmentanytree_apptainer.sbatch)
SAT_STACK=$(sbatch --parsable \
  --dependency=afterok:${SAT_CONTAINER} \
  scripts/slurm/install_segmentanytree_python_stack.sbatch)
SAT_STACK_TEST=$(sbatch --parsable \
  --dependency=afterok:${SAT_STACK} \
  scripts/slurm/test_segmentanytree_python_stack_repair.sbatch)
```

The consolidated pilot requires an explicit execution flag:

```bash
SAT_PILOT=$(sbatch --parsable \
  --dependency=afterok:${SAT_STACK_TEST} \
  --export=ALL,SEGMENTANYTREE_EXECUTE=1 \
  scripts/slurm/run_segmentanytree_for_instance_pilot_apptainer.sbatch)
```

The original successful investigation completed inference and instance
prediction, then needed a 35-second repaired final-export job. The consolidated
script applies that export correction during inference but still needs a clean
reproduction on Barkla before the full array is submitted.

For the full benchmark, use the dependency-chained prediction, normalisation,
evaluation and summary commands in the
[runbook](docs/segmentanytree_for_instance_benchmark.md). Do not train or tune
on the evaluation split. Do not report an overall FOR-instance result until all
32 tasks and their summaries have been checked.

External NEWFOR results should be compared only when metrics, class filters,
IoU thresholds and coordinate tolerances are compatible.

## Outputs

The SegmentAnyTree/FOR-instance workflow writes:

- method-specific predictions outside Git;
- a labelled LAZ containing `PredInstance`, followed by one normalised XYZ PLY
  per positive predicted tree;
- per-plot run metadata JSON, including runtime and peak memory when available;
- per-plot evaluation, matched-pair and unmatched-instance tables;
- plot, collection and split benchmark summary tables.

These outputs are intentionally excluded from Git.

## Version-Control Exclusions

The `.gitignore` excludes:

- raw and derived data under `Datasets/` and `data/`;
- external method checkouts;
- scheduler and method logs;
- result metadata, tables and prediction artefacts;
- LAS, LAZ, PLY, NumPy, container, checkpoint and image files;
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
