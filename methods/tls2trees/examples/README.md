# TLS2trees Public-Safe Examples

This directory contains small result summaries and fabricated schema examples.
It contains no point coordinates, prediction files, model files, logs or
machine-specific paths.

## Completed operational result

[`tls2trees_frdr_prediction_summary.csv`](tls2trees_frdr_prediction_summary.csv)
is the authoritative 16-plot FRDR operational summary. It reports input and
retained point counts, predicted-tree counts, runtime, memory and completion
status. FRDR has no individual-tree reference labels, so this file is not an
instance-accuracy result.

## FOR-instance candidate evidence

[`for_instance_inventory_summary.csv`](for_instance_inventory_summary.csv)
records the input inventory used by the TLS2trees compatibility pilot. It does
not establish a completed FOR-instance accuracy benchmark.

## Fabricated schema examples

The following files use synthetic values to document public metadata and table
schemas; they are not benchmark results:

- [`frdr_dataset_inventory_example.csv`](frdr_dataset_inventory_example.csv);
- [`tls2trees_prediction_summary_example.csv`](tls2trees_prediction_summary_example.csv);
- [`tls2trees_conversion_metadata_example.json`](tls2trees_conversion_metadata_example.json); and
- [`tls2trees_run_metadata_example.json`](tls2trees_run_metadata_example.json).

Full converted inputs and predictions remain outside Git under ignored runtime
paths.
