# Public-Safe Results And Examples

This directory contains one completed public-safe benchmark summary and four
fabricated schema examples.

`tls2trees_frdr_prediction_summary.csv` contains the completed FRDR/TLS2trees
per-plot prediction and operational summary. It does not contain coordinates,
point clouds, logs or accuracy metrics.

Files ending in `_example` contain fabricated values for schema illustration.
They are not derived from FRDR data, TLS2trees predictions or a Barkla run and
must not be used as benchmark results.

- `frdr_dataset_inventory_example.csv` shows two inventory rows, including an
  explicit unknown `woods` value.
- `tls2trees_conversion_metadata_example.json` shows conversion metadata after
  that unknown value is dropped.
- `tls2trees_run_metadata_example.json` shows a successful instance-stage run
  record.
- `tls2trees_prediction_summary_example.csv` shows the corresponding combined
  plot summary.

Paths beginning with `/path/to/` are placeholders. Counts, coordinates,
versions, timings and memory values are synthetic.
