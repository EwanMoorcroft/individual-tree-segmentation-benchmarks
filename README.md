# Individual Tree Segmentation Benchmarks

[![Tests](https://github.com/EwanMoorcroft/individual-tree-segmentation-benchmarks/actions/workflows/tests.yml/badge.svg)](https://github.com/EwanMoorcroft/individual-tree-segmentation-benchmarks/actions/workflows/tests.yml)

## Purpose

This repository collects reproducible workflows for benchmarking multiple
individual tree segmentation methods across multiple LiDAR datasets using the
University of Liverpool Barkla2 HPC system.

The first completed prediction benchmark uses TLS2trees instance segmentation
with the FRDR treeiso terrestrial laser scanning dataset. Future method
wrappers may cover SegmentAnyTree, TreeLearn and other deep learning methods,
and traditional segmentation baselines without creating separate repositories.

No source datasets, converted point clouds, predictions, scheduler logs or
external method repositories are included.

## Current Status

- FRDR/TLS2trees: completed prediction and operational benchmark across 16
  plots; no reference instance accuracy is reported.
- FOR-instance: downloaded and inspected; selected as the immediate next
  accuracy dataset because its annotated point clouds include `treeID`.
- Wytham Woods: downloaded and inspected; retained as a strong TLS reference
  dataset after plot-level reference reconstruction from per-tree files.

See the [benchmark registry](BENCHMARKS.md),
[dataset feasibility assessment](docs/dataset_feasibility.md), and
[labelled accuracy preparation plan](docs/labelled_accuracy_benchmark_plan.md)
for current and candidate dataset-method combinations.

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
| Dataset root | `~/data/datasets/frdr_treeiso` |
| TLS2trees checkout | `external/TLS2trees` |
| Converted inputs | `data/interim/tls2trees/frdr_full/<plot_name>/` |
| Predictions | `data/predictions/tls2trees/frdr_full/<plot_name>/` |
| Logs | `logs/tls2trees_frdr_full/` |
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
│   ├── frdr_tls2trees_benchmark.yml
│   └── wytham_accuracy_benchmark.yml
├── docs/
│   ├── dataset_feasibility.md
│   ├── evaluation_metrics.md
│   ├── frdr_tls2trees_results.md
│   ├── labelled_accuracy_benchmark_plan.md
│   └── tls2trees_frdr_benchmark_runbook.md
├── examples/
│   ├── README.md
│   ├── for_instance_inventory_summary.csv
│   ├── frdr_dataset_inventory_example.csv
│   ├── tls2trees_conversion_metadata_example.json
│   ├── tls2trees_frdr_prediction_summary.csv
│   ├── tls2trees_prediction_summary_example.csv
│   └── tls2trees_run_metadata_example.json
├── scripts/
│   ├── data/
│   │   ├── convert_frdr_woods_to_tls2trees_ply.py
│   │   └── inspect_frdr_dataset_inventory.py
│   ├── evaluation/
│   │   └── instance_iou_f1.py
│   ├── methods/
│   │   ├── run_tls2trees_instance_for_plot.py
│   │   ├── summarise_tls2trees_outputs.py
│   │   └── tls2trees_patched/
│   │       └── instance_patched.py
│   └── slurm/
│       ├── convert_frdr_to_tls2trees_array.sbatch
│       ├── inspect_frdr_inventory.sbatch
│       ├── run_tls2trees_frdr_array.sbatch
│       └── summarise_tls2trees_frdr_outputs.sbatch
├── src/
│   └── benchmark/
│       ├── __init__.py
│       └── ply_io.py
└── tests/
    └── test_frdr_tls2trees_workflow.py
```

## Public-Safe Results And Examples

The [`examples/`](examples/) directory contains the completed FRDR per-plot
summary, a small FOR-instance inventory extract, and synthetic schema examples.
No coordinates, point clouds, prediction files or logs are included.

## Recommended Pilot First

Create the log directory and check project storage before submitting jobs.
Slurm opens its output files before the job script starts:

```bash
cd ~/scratch/tree-seg-benchmark
mkdir -p logs/tls2trees_frdr_full

df -h ~/scratch/tree-seg-benchmark
du -h --max-depth=2 ~/scratch/tree-seg-benchmark/data 2>/dev/null | sort -h

sbatch --array=0-0 scripts/slurm/convert_frdr_to_tls2trees_array.sbatch
```

Wait for conversion task `0` to finish successfully, inspect its metadata, and
then submit only the matching instance task:

```bash
sbatch --array=0-0 scripts/slurm/run_tls2trees_frdr_array.sbatch
```

After the instance task finishes successfully:

```bash
python scripts/methods/summarise_tls2trees_outputs.py \
  --plot-name LPine_plot1 \
  --output-dir data/predictions/tls2trees/frdr_full/LPine_plot1
```

Do not submit the full arrays as independent jobs. Follow the dependency-chained
commands in
[`docs/tls2trees_frdr_benchmark_runbook.md`](docs/tls2trees_frdr_benchmark_runbook.md)
for the complete preflight, inventory and remaining plots.

## Recommended Staged Execution

1. Run and review the dataset inventory.
2. Run conversion for one small plot selected from the inventory.
3. Dry-run and execute the instance stage for that plot.
4. Repeat conversion and execution for one large plot.
5. Submit the remaining Slurm array tasks only after both pilots succeed.
6. Summarise all completed predictions.

The conversion and prediction arrays check available project-filesystem space
before heavy work. Set `TLS2TREES_MIN_FREE_GB` to change the configured reserve.

## Outputs

The workflow writes:

- `*.leafoff.ply` files containing individual predicted trees;
- per-plot conversion metadata JSON;
- per-plot run metadata JSON, including runtime and peak memory when available;
- per-plot output summary JSON;
- per-tree CSV summaries;
- `results/tables/tls2trees_frdr_prediction_summary.csv`.

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
