# TLS2trees

## Method Summary

TLS2trees is a terrestrial-laser-scanning tree extraction workflow used here
for a completed FRDR operational prediction benchmark, a completed
development-tuned FOR-instance benchmark, a frozen published-default
FOR-instance workflow and one legacy instance-stage pilot.

## Upstream Repository And Citation

TLS2trees is an external dependency and is not vendored here. The tested
repository URL and commit are recorded in
[`configs/frdr_benchmark.yml`](configs/frdr_benchmark.yml) and
the legacy [`configs/for_instance_accuracy.yml`](configs/for_instance_accuracy.yml).
The publication-derived method parameters and executable provenance are
recorded in
[`configs/for_instance_published_default.yml`](configs/for_instance_published_default.yml).
The independently frozen held-out route, exact split and target contract are
recorded in
[`configs/for_instance_published_default_test.yml`](configs/for_instance_published_default_test.yml).
Cite the [`TLS2trees` paper](https://doi.org/10.1111/2041-210X.14233) when using
the method.

## Training Mode Support

The committed FRDR and FOR-instance TLS2trees rows are declared as
`external_training_only` because no local FOR-instance or FRDR weight fitting
is performed in this repository. FOR-instance method variants are named
`published_default` and `development_tuned`. Method parameters and patches are
documented in the configs and runbooks.

## Input Requirements

The FRDR route consumes LAZ files with the `woods` semantic field. The legacy
FOR-instance pilot consumes annotated LAS files with `treeID` and
`classification` fields and tests a leaf-off subset using classes `4` and `6`.
It supplies those reference semantics to the instance stage and is not a full
published-method benchmark. The retained development-tuned execution ran the
bundled semantic stage on label-stripped geometry.

## Output Contract

TLS2trees produces one file per predicted tree. The published/default adapter
matches those raw coordinates uniquely to deterministic 0.02 m representatives
and projects their labels through a saved map to every source row. The primary
evaluation artefact is source-row aligned; the legacy pilot remains a
coordinate-only diagnostic. Operational summaries may be reported without
individual-tree reference labels.

## FOR-instance Compatibility

TLS2trees on FOR-instance is a domain-compatibility benchmark because
FOR-instance is UAV laser-scanning data. The complete experiment design is
documented in [`docs/for_instance_benchmark.md`](docs/for_instance_benchmark.md).
The runnable development-smoke boundary is documented in
[`docs/for_instance_published_default_smoke.md`](docs/for_instance_published_default_smoke.md).
The earlier pilot and its limitations remain in
[`docs/for_instance_pilot.md`](docs/for_instance_pilot.md).

## Barkla Environment

Manifest, conversion and evaluation stages use the shared `treebench`
environment. The published semantic and instance stages require the isolated,
version-pinned `~/fastscratch/venvs/tls2trees` compatibility environment; the
guarded Slurm setup and GPU validator are documented in the development-smoke
runbook. Neither the TreeLearn environment nor `treebench` is modified.
External repositories, environments, raw data, predictions, logs and large
outputs remain outside Git.

## Slurm Workflow

Use:

- [`docs/frdr_runbook.md`](docs/frdr_runbook.md) for reproduction;
- [`docs/frdr_results.md`](docs/frdr_results.md) for the completed operational
  result;
- [`configs/frdr_benchmark.yml`](configs/frdr_benchmark.yml) for parameters and
  paths; and
- [`examples/README.md`](examples/README.md) for the completed FRDR and
  FOR-instance results, retained prediction evidence, development diagnostics,
  historical candidate evidence and fabricated schema examples.

The files below implement only the legacy pilot and are not the completed
source-row-aligned route:

- legacy preparation: [`scripts/data/convert_for_instance_to_tls2trees_ply.py`](scripts/data/convert_for_instance_to_tls2trees_ply.py);
- legacy instance stage: [`scripts/runtime/run_tls2trees_for_instance_plot.py`](scripts/runtime/run_tls2trees_for_instance_plot.py);
- legacy summarisation: [`scripts/runtime/summarise_tls2trees_outputs.py`](scripts/runtime/summarise_tls2trees_outputs.py); and
- legacy Slurm entrypoints: [`slurm/for_instance/convert_for_instance_tls2trees_pilot.sbatch`](slurm/for_instance/convert_for_instance_tls2trees_pilot.sbatch), [`slurm/for_instance/run_tls2trees_for_instance_pilot.sbatch`](slurm/for_instance/run_tls2trees_for_instance_pilot.sbatch) and [`slurm/for_instance/evaluate_for_instance_tls2trees_pilot.sbatch`](slurm/for_instance/evaluate_for_instance_tls2trees_pilot.sbatch).

The target-explicit source-row alignment, development selection, held-out test
execution, immutable retained-prediction evaluation and prediction retention
are implemented and validated on Barkla. The evaluation route excludes
FOR-instance class-3 out-points from the scoring domain. The separate
development leaf screen is diagnostic and cannot alter the tested
configuration. The fixed full published-default test is submitted with
[`slurm/for_instance/submit_published_default_held_out_test.sh`](slurm/for_instance/submit_published_default_held_out_test.sh)
and monitored with
[`slurm/for_instance/monitor_published_default_held_out_test.sh`](slurm/for_instance/monitor_published_default_held_out_test.sh).
It uses the reviewed published configuration without development-metric
selection. The completed FRDR route uses:

- preparation: [`scripts/data/convert_frdr_woods_to_tls2trees_ply.py`](scripts/data/convert_frdr_woods_to_tls2trees_ply.py);
- instance inference: [`scripts/runtime/run_tls2trees_instance_for_plot.py`](scripts/runtime/run_tls2trees_instance_for_plot.py); and
- FRDR Slurm entrypoints: [`slurm/frdr/convert_frdr_to_tls2trees_array.sbatch`](slurm/frdr/convert_frdr_to_tls2trees_array.sbatch), [`slurm/frdr/run_tls2trees_frdr_array.sbatch`](slurm/frdr/run_tls2trees_frdr_array.sbatch) and [`slurm/frdr/summarise_tls2trees_frdr_outputs.sbatch`](slurm/frdr/summarise_tls2trees_frdr_outputs.sbatch).

## Evaluation Route

The completed FRDR run is an operational prediction benchmark only. FRDR has
semantic `woods` labels but no individual-tree reference labels, so the run
does not report instance precision, recall, F1 or IoU. The legacy FOR-instance
pilot uses coordinate-based one-to-one matching for the leaf-off reference
classes and is not a headline result. The completed development-tuned run uses
source-row predictions and the final class-3-ignore evaluation protocol.

## Known Limitations

TLS2trees expects TLS-style inputs, whereas FOR-instance is UAV laser scanning
data. The low development-tuned test score is therefore interpreted as poor
cross-modality transfer for this frozen pipeline, not as evidence that
TLS2trees cannot segment trees in its intended terrestrial-scanning domain.

## Current Benchmark Status

The FRDR treeiso prediction and operational benchmark completed for all 16
plots. Its public-safe per-plot summary is
[`examples/tls2trees_frdr_prediction_summary.csv`](examples/tls2trees_frdr_prediction_summary.csv).
The development-tuned leaf-on result has mean plot F1 `0.015023` and micro F1
`0.016620` on all 11 held-out plots. Its leaf-off target is reported separately
as a zero-match diagnostic. The 22 retained source-row prediction files, the
development leaf screen, the published-default workflow and the legacy pilot
are indexed in [`../../BENCHMARKS.md`](../../BENCHMARKS.md).
