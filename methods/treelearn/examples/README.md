# TreeLearn Public-Safe Examples

This directory contains public-safe result tables and provenance records. It
contains no point coordinates, predictions, checkpoints, logs or private paths.

## Headline held-out results

The two comparable 11-plot test variants each retain per-plot, site, aggregate
and provenance evidence:

- published pretrained:
  [`treelearn_pretrained_test_plot_results_20260714.csv`](treelearn_pretrained_test_plot_results_20260714.csv),
  [`treelearn_pretrained_test_site_results_20260714.csv`](treelearn_pretrained_test_site_results_20260714.csv),
  [`treelearn_pretrained_test_results_20260714.csv`](treelearn_pretrained_test_results_20260714.csv) and
  [`treelearn_pretrained_test_provenance_20260714.json`](treelearn_pretrained_test_provenance_20260714.json);
- development fine-tuned:
  [`treelearn_finetuned_test_plot_results_20260713.csv`](treelearn_finetuned_test_plot_results_20260713.csv),
  [`treelearn_finetuned_test_site_results_20260713.csv`](treelearn_finetuned_test_site_results_20260713.csv),
  [`treelearn_finetuned_test_results_20260713.csv`](treelearn_finetuned_test_results_20260713.csv) and
  [`treelearn_finetuned_test_provenance_20260713.json`](treelearn_finetuned_test_provenance_20260713.json).

## Development diagnostics

The accepted one-plot smoke, 21-plot overlap-affected development result and
rejected five-plot fine-tuning validation sweep remain diagnostic evidence, not
additional headline rows:

- [`accepted_development_smoke_20260712.json`](accepted_development_smoke_20260712.json);
- [`treelearn_completed_development_results_20260712.csv`](treelearn_completed_development_results_20260712.csv),
  [`treelearn_completed_development_site_results_20260712.csv`](treelearn_completed_development_site_results_20260712.csv) and
  [`treelearn_completed_development_provenance_20260712.json`](treelearn_completed_development_provenance_20260712.json); and
- [`treelearn_finetune_validation_results_20260712.csv`](treelearn_finetune_validation_results_20260712.csv)
  and
  [`treelearn_finetune_validation_provenance_20260712.json`](treelearn_finetune_validation_provenance_20260712.json).

## Fabricated schema example

[`metadata_example.json`](metadata_example.json) uses synthetic values to
illustrate metadata shape. It is not benchmark evidence.

Raw and aligned prediction artifacts remain outside Git under ignored runtime
paths; their retained hashes and locations are recorded in the provenance and
repository-wide retention registry.
