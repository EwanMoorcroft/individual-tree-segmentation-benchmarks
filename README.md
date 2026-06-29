# FRDR TLS2trees Benchmark Protocol

This repository contains scripts and Slurm workflows for running TLS2trees
instance segmentation on the FRDR treeiso terrestrial laser scanning dataset
on the University of Liverpool Barkla2 HPC.

## Scope

The protocol covers:

- inventorying FRDR LAZ files;
- converting the FRDR `woods` field to TLS2trees semantic labels;
- running per-plot predictions through a Slurm array;
- validating and summarising predicted tree files;
- a reusable IoU/F1 evaluator for future use when reference instance labels
  become available.

This is a prediction benchmark workflow. It does not redistribute the FRDR
dataset or vendor the TLS2trees source repository.

## Accuracy Limitation

FRDR LAZ files contain a `woods` field for wood/non-wood classification, not
individual-tree reference instance labels. Instance IoU, precision, recall and
F1 therefore cannot be reported from these files alone. The evaluator refuses
to calculate these metrics unless an external instance reference is supplied.

## FRDR Mapping

| FRDR value | Meaning | TLS2trees label |
| --- | --- | --- |
| `woods = 1` | wood | `3` |
| `woods = 2` | non-wood | `1` |

Unknown values are handled according to `conversion.unknown_policy` in the
configuration. The full benchmark uses `drop` because `NSpruce_plot2` contains
`woods = 0.0`; dropped points are counted in conversion metadata.

Each converted plot is written as one binary PLY tile containing exactly:
`x`, `y`, `z`, `n_z`, and `label`. The current `n_z` calculation subtracts the
plot-local minimum Z and is recorded as a feasibility approximation rather
than terrain normalisation.

## Barkla Environment

The tested environment used Python 3.12.10 in the existing `treebench` venv.

```bash
module purge
module load miniforge3/25.3.0-python3.12.10
source ~/fastscratch/venvs/treebench/bin/activate
```

Do not use `conda activate` for this venv.

Install the protocol dependencies where required:

```bash
python -m pip install -r requirements.txt
```

## Expected Barkla Paths

| Purpose | Path |
| --- | --- |
| Project root | `~/scratch/tree-seg-benchmark` |
| FRDR dataset | `~/data/datasets/frdr_treeiso` |
| TLS2trees checkout | `external/TLS2trees` |
| Converted inputs | `data/interim/tls2trees/frdr_full/<plot_name>/` |
| Predictions | `data/predictions/tls2trees/frdr_full/<plot_name>/` |
| Logs | `logs/tls2trees_frdr_full/` |
| Metadata | `results/metadata/` |
| Tables | `results/tables/` |

The dataset path may resolve to a mounted filesystem path on Barkla2. Scripts
use the configured path and do not modify source LAZ files.

## TLS2trees Checkout

TLS2trees is not included in this repository. Clone and pin it separately:

```bash
mkdir -p external
git clone https://github.com/tls-tools-ucl/TLS2trees.git external/TLS2trees
git -C external/TLS2trees checkout ca12cb73b2c736d80b020e8025f8d975d42e6f01
```

The tested commit is:
`ca12cb73b2c736d80b020e8025f8d975d42e6f01`.

The included `instance_patched.py` wrapper applies the documented pandas
compatibility fix at runtime without modifying the external checkout.

## Run Order

Create the log directory before submission because Slurm opens log files
before job scripts start:

```bash
cd ~/scratch/tree-seg-benchmark
mkdir -p logs/tls2trees_frdr_full

sbatch scripts/slurm/inspect_frdr_inventory.sbatch
sbatch scripts/slurm/convert_frdr_to_tls2trees_array.sbatch
sbatch scripts/slurm/run_tls2trees_frdr_array.sbatch
sbatch scripts/slurm/summarise_tls2trees_frdr_outputs.sbatch
```

Use the staged procedure in
[`docs/tls2trees_frdr_benchmark_runbook.md`](docs/tls2trees_frdr_benchmark_runbook.md):

1. Run and review the inventory.
2. Convert one small plot selected from the inventory.
3. Dry-run and execute that plot.
4. Repeat with one large plot.
5. Submit the remaining conversion and prediction array tasks only after both
   pilots succeed.

The array scripts check available disk space before heavy work. The threshold
can be overridden with `TLS2TREES_MIN_FREE_GB`.

## Outputs

Generated outputs are excluded from version control:

- per-plot `*.leafoff.ply` tree predictions;
- conversion and run metadata JSON;
- per-plot output summary JSON;
- per-tree CSV summaries;
- `results/tables/tls2trees_frdr_prediction_summary.csv`.

The combined table records input and retained point counts, dropped unknown
points, predicted tree counts and points, runtime, peak memory when available,
return code and status.

## Tests

Tests use small synthetic LAS/PLY data and do not require the FRDR dataset:

```bash
python -m pip install pytest
python -m pytest
```

## Data And Citation

Do not redistribute FRDR data through this repository. Obtain the dataset from
its original FRDR source and follow the dataset's licence and citation
requirements. Cite TLS2trees and its associated research according to the
upstream project guidance.
