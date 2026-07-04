# TLS2trees

## Completed work

The FRDR treeiso prediction and operational benchmark completed for all 16
plots. Its public-safe per-plot summary is
[`examples/tls2trees_frdr_prediction_summary.csv`](examples/tls2trees_frdr_prediction_summary.csv).
FRDR has semantic `woods` labels but no individual-tree reference labels, so
the run does not report instance precision, recall, F1 or IoU.

Use:

- [`docs/frdr_runbook.md`](docs/frdr_runbook.md) for reproduction;
- [`docs/frdr_results.md`](docs/frdr_results.md) for the completed operational
  result; and
- [`configs/frdr_benchmark.yml`](configs/frdr_benchmark.yml) for parameters and
  paths.

## Candidate work

TLS2trees on FOR-instance remains a compatibility experiment, not the current
primary benchmark. The pilot and its limitations are documented in
[`docs/for_instance_pilot.md`](docs/for_instance_pilot.md).

TLS2trees is an external dependency and is not vendored here.
