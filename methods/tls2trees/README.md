# TLS2trees

## Method Summary

TLS2trees is a terrestrial-laser-scanning tree extraction workflow used here
for one completed FRDR operational prediction benchmark and one FOR-instance
compatibility pilot.

## Upstream Repository And Citation

TLS2trees is an external dependency and is not vendored here. The tested
repository URL and commit are recorded in
[`configs/frdr_benchmark.yml`](configs/frdr_benchmark.yml) and
[`configs/for_instance_accuracy.yml`](configs/for_instance_accuracy.yml).

## Training Mode Support

The committed FRDR and FOR-instance TLS2trees rows are declared as
`external_training_only` because no local FOR-instance or FRDR weight fitting
is performed in this repository. Method parameters and patches are documented
in the configs and runbooks.

## Input Requirements

The FRDR route consumes LAZ files with the `woods` semantic field. The
FOR-instance pilot consumes annotated LAS files with `treeID` and
`classification` fields and currently tests a leaf-off subset using classes
`4` and `6`.

## Output Contract

TLS2trees produces one file per predicted tree. Instance accuracy therefore
uses coordinate-based matching unless a future adapter records stable
source-point identifiers. Operational summaries may be reported without
individual-tree reference labels.

## FOR-instance Compatibility

TLS2trees on FOR-instance remains a compatibility experiment, not the current
primary benchmark. The pilot and its limitations are documented in
[`docs/for_instance_pilot.md`](docs/for_instance_pilot.md).

## Barkla Environment

The workflow uses the shared Barkla Python environment documented in the
method configs. External repositories, raw data, predictions, logs and large
outputs remain outside Git.

## Slurm Workflow

Use:

- [`docs/frdr_runbook.md`](docs/frdr_runbook.md) for reproduction;
- [`docs/frdr_results.md`](docs/frdr_results.md) for the completed operational
  result; and
- [`configs/frdr_benchmark.yml`](configs/frdr_benchmark.yml) for parameters and
  paths.

Current canonical equivalents are:

- preparation: [`scripts/data/convert_for_instance_to_tls2trees_ply.py`](scripts/data/convert_for_instance_to_tls2trees_ply.py) and [`scripts/data/convert_frdr_woods_to_tls2trees_ply.py`](scripts/data/convert_frdr_woods_to_tls2trees_ply.py);
- inference: [`scripts/runtime/run_tls2trees_for_instance_plot.py`](scripts/runtime/run_tls2trees_for_instance_plot.py) and [`scripts/runtime/run_tls2trees_instance_for_plot.py`](scripts/runtime/run_tls2trees_instance_for_plot.py);
- prediction adaptation: coordinate-based tree-file outputs are consumed directly by the shared evaluator;
- summarisation: [`scripts/runtime/summarise_tls2trees_outputs.py`](scripts/runtime/summarise_tls2trees_outputs.py);
- FOR-instance Slurm entrypoints: [`slurm/for_instance/convert_for_instance_tls2trees_pilot.sbatch`](slurm/for_instance/convert_for_instance_tls2trees_pilot.sbatch), [`slurm/for_instance/run_tls2trees_for_instance_pilot.sbatch`](slurm/for_instance/run_tls2trees_for_instance_pilot.sbatch) and [`slurm/for_instance/evaluate_for_instance_tls2trees_pilot.sbatch`](slurm/for_instance/evaluate_for_instance_tls2trees_pilot.sbatch); and
- FRDR Slurm entrypoints: [`slurm/frdr/convert_frdr_to_tls2trees_array.sbatch`](slurm/frdr/convert_frdr_to_tls2trees_array.sbatch), [`slurm/frdr/run_tls2trees_frdr_array.sbatch`](slurm/frdr/run_tls2trees_frdr_array.sbatch) and [`slurm/frdr/summarise_tls2trees_frdr_outputs.sbatch`](slurm/frdr/summarise_tls2trees_frdr_outputs.sbatch).

## Evaluation Route

The completed FRDR run is an operational prediction benchmark only. FRDR has
semantic `woods` labels but no individual-tree reference labels, so the run
does not report instance precision, recall, F1 or IoU. The FOR-instance pilot
uses coordinate-based one-to-one matching for the leaf-off reference classes.

## Known Limitations

TLS2trees expects TLS-style inputs. FOR-instance is UAV laser scanning data, so
the pilot is a compatibility test rather than evidence that the method is
fully tuned for the dataset.

## Current Benchmark Status

The FRDR treeiso prediction and operational benchmark completed for all 16
plots. Its public-safe per-plot summary is
[`examples/tls2trees_frdr_prediction_summary.csv`](examples/tls2trees_frdr_prediction_summary.csv).
TLS2trees on FOR-instance remains a candidate compatibility row in
[`../../BENCHMARKS.md`](../../BENCHMARKS.md).
