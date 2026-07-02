# Public-Safe Results And Examples

This directory contains public-safe benchmark summaries, one real
SegmentAnyTree pilot record and fabricated schema examples.

`tls2trees_frdr_prediction_summary.csv` contains the completed FRDR/TLS2trees
per-plot prediction and operational summary. It does not contain coordinates,
point clouds, logs or accuracy metrics.

`for_instance_inventory_summary.csv` contains ten observed inventory rows from
the FOR-instance inspection. It includes relative paths and aggregate counts,
not point coordinates. `has_treeSP` is marked `not_confirmed` because that field
was not established by the retained inspection summary. Split values are
copied from the dataset's `data_split_metadata.csv`.

`segmentanytree_for_instance_pilot_metrics.csv` and
`segmentanytree_for_instance_pilot_status.json` contain the completed
FOR-instance development-pilot record. They contain aggregate counts and
metrics only. They are not full benchmark results.

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
- `segmentanytree_for_instance_plot_metrics_example.csv` shows one synthetic
  labelled evaluation row.
- `segmentanytree_for_instance_summary_example.csv` shows its synthetic
  aggregate schema.

Paths beginning with `/path/to/` are placeholders. Counts, coordinates,
versions, timings and memory values are synthetic.
