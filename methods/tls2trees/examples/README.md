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

## Completed development-tuned FOR-instance result

The frozen development-tuned held-out execution is recorded by:

- [`tls2trees_development_tuned_test_plot_results.csv`](tls2trees_development_tuned_test_plot_results.csv);
- [`tls2trees_development_tuned_test_site_results.csv`](tls2trees_development_tuned_test_site_results.csv);
- [`tls2trees_development_tuned_test_results.csv`](tls2trees_development_tuned_test_results.csv);
- [`tls2trees_development_tuned_test_provenance.json`](tls2trees_development_tuned_test_provenance.json); and
- [`tls2trees_development_tuned_prediction_retention_manifest.json`](tls2trees_development_tuned_prediction_retention_manifest.json).

The leaf-on result is the canonical TLS2trees accuracy result. It evaluates 38
predicted instances against 323 references on 11 held-out plots, with 3 TP,
35 FP, 320 FN, mean plot F1 `0.015023` and micro F1 `0.016620`.

The target-specific leaf-off evidence is stored in the corresponding
`tls2trees_development_tuned_leaf_off_test_*` files. It has 22 predicted
instances and no matches, and is a diagnostic rather than a headline row. Both
targets exclude class-3 out-points from the scoring domain. All 22 source-row
prediction files are hash-verified in the retention manifest.

## Development leaf-attachment screen

The guarded finalisation route will write public leaf-screen plot, candidate
and provenance files for the completed development-only nine-setting screen
across five sites. All 45 metrics were valid and no held-out data were
accessed. Publication verifies the retained Barkla summary before exporting
the files.

## Published-default FOR-instance result

The frozen published-default workflow writes its held-out plot, site, overall,
provenance and retention files here after the separate Barkla run completes.
It does not reuse development-selected instance parameters.

## Historical FOR-instance candidate evidence

[`for_instance_inventory_summary.csv`](for_instance_inventory_summary.csv)
records the input inventory used by the TLS2trees compatibility pilot. It does
not establish a valid source-row-aligned FOR-instance accuracy result.

## Fabricated schema examples

The following files use synthetic values to document public metadata and table
schemas; they are not benchmark results:

- [`frdr_dataset_inventory_example.csv`](frdr_dataset_inventory_example.csv);
- [`tls2trees_prediction_summary_example.csv`](tls2trees_prediction_summary_example.csv);
- [`tls2trees_conversion_metadata_example.json`](tls2trees_conversion_metadata_example.json); and
- [`tls2trees_run_metadata_example.json`](tls2trees_run_metadata_example.json).

Full converted inputs and predictions remain outside Git under ignored runtime
paths.
