# Individual Tree Segmentation Benchmarks

[![Tests](https://github.com/EwanMoorcroft/individual-tree-segmentation-benchmarks/actions/workflows/tests.yml/badge.svg)](https://github.com/EwanMoorcroft/individual-tree-segmentation-benchmarks/actions/workflows/tests.yml)

## Purpose

This repository collects reproducible workflows for benchmarking individual
tree segmentation methods across LiDAR datasets using the University of
Liverpool Barkla2 HPC system.

The first implemented benchmark uses TLS2trees instance segmentation with the
FRDR treeiso terrestrial laser scanning dataset. Additional dataset and method
combinations can be added through new configs, data adapters, method wrappers,
Slurm jobs and evaluation modules without creating separate repositories.

No source datasets, converted point clouds, predictions, scheduler logs or
external method repositories are included.

## Scope

The current FRDR/TLS2trees protocol covers:

- FRDR dataset inventory and field inspection;
- conversion of the FRDR `woods` field to TLS2trees semantic labels;
- patched TLS2trees instance-stage execution;
- Slurm array conversion and prediction workflows;
- preservation of per-tree predictions;
- runtime, memory and output metadata collection;
- output validation and per-tree summary tables;
- a reusable IoU/F1 evaluator for future reference instance labels.

See [`BENCHMARKS.md`](BENCHMARKS.md) for the benchmark registry and extension
requirements.

## Important Limitation

The FRDR LAZ files contain a `woods` scalar field for wood/non-wood
classification, not individual-tree reference instance labels. IoU, precision,
recall and F1 therefore cannot be computed from the FRDR LAZ files alone.

The workflow preserves predicted instances so these metrics can be calculated
later if suitable reference tree instance labels are supplied. The evaluator
exits with a clear error when no reference labels are provided.

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

```text
.
├── README.md
├── BENCHMARKS.md
├── requirements.txt
├── configs/
│   └── frdr_tls2trees_benchmark.yml
├── docs/
│   └── tls2trees_frdr_benchmark_runbook.md
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

## Run Order

Create the log directory before submission because Slurm opens its output files
before the job script starts:

```bash
cd ~/scratch/tree-seg-benchmark
mkdir -p logs/tls2trees_frdr_full

sbatch scripts/slurm/inspect_frdr_inventory.sbatch
sbatch scripts/slurm/convert_frdr_to_tls2trees_array.sbatch
sbatch scripts/slurm/run_tls2trees_frdr_array.sbatch
sbatch scripts/slurm/summarise_tls2trees_frdr_outputs.sbatch
```

See
[`docs/tls2trees_frdr_benchmark_runbook.md`](docs/tls2trees_frdr_benchmark_runbook.md)
for preflight checks, dependencies and job sequencing.

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
- virtual environments and Python/test caches.

No raw data or predictions are included in this repository.

## Tests

The test suite uses small synthetic point clouds and does not require the FRDR
dataset or a TLS2trees checkout:

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
