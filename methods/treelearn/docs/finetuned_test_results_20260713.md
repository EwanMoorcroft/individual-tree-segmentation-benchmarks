# TreeLearn fine-tuned held-out test result

## Result identity

Run `treelearn_for-instance_fine_tuned_on_dev_long_20260712_233227` completed
the authorised one-time evaluation of its preregistered seed-42 epoch-35
checkpoint on 13 July 2026. All 11 locally available official FOR-instance
test plots completed, the final gate passed, and no task failed.

The checkpoint was trained on the frozen 16-plot development-training subset
and selected before test access. It started from the authors-released
`model_weights_finetuned.pth` checkpoint, whose stated training data excludes
FOR-instance. The frozen checkpoint SHA-256 is
`dcc02bb9fdd81cfbdb94454bb7a744c17eee7fa2c4a53096d529b21eb64fc590`.

## Overall test result

| Plots | Predictions | References | TP | FP | FN | Mean plot precision | Mean plot recall | Mean plot F1 | Micro precision | Micro recall | Micro F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | 623 | 323 | 157 | 466 | 166 | 0.282575 | 0.565696 | 0.364685 | 0.252006 | 0.486068 | 0.331924 |

The test plots contain 49,709,922 source points; 40,976,651 points are in the
union evaluation mask. Mean plot matched IoU is `0.751563`, mean IoU across
all matched pairs is `0.765048`, mean unweighted coverage is `0.542848`, and
mean weighted coverage is `0.652171`.

## Site results

| Site | Plots | Predictions | References | TP | FP | FN | Mean plot F1 | Micro precision | Micro recall | Micro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CULS | 1 | 55 | 20 | 20 | 35 | 0 | 0.533333 | 0.363636 | 1.000000 | 0.533333 |
| NIBIO | 6 | 412 | 161 | 84 | 328 | 77 | 0.297592 | 0.203883 | 0.521739 | 0.293194 |
| RMIT | 1 | 47 | 64 | 9 | 38 | 55 | 0.162162 | 0.191489 | 0.140625 | 0.162162 |
| SCION | 2 | 61 | 43 | 32 | 29 | 11 | 0.620663 | 0.524590 | 0.744186 | 0.615385 |
| TUWIEN | 1 | 48 | 35 | 12 | 36 | 23 | 0.289157 | 0.250000 | 0.342857 | 0.289157 |

SCION has the highest mean plot F1 (`0.620663`) and RMIT the lowest
(`0.162162`). These values are descriptive frozen test results and were not
used for checkpoint, threshold or post-processing selection.

## Comparability and retention

This is a primary cross-method row: it uses the same 11-plot test subset, 323
references, `for_instance_pointwise_v1` union mask, IoU `>= 0.5`, and
maximum-cardinality one-to-one matching as the completed SegmentAnyTree and
TreeX rows. It is separate from the published TreeLearn development diagnostic,
whose December 2024 weights have documented FOR-instance training overlap.

The final gate re-hashed all five raw or aligned prediction artefacts for each
plot. All 55 files remain in the recorded run-scoped Barkla roots, and the
retention manifest SHA-256 is
`972ad17ba103b151095d2925862e76e7186594b549d659cea2ca781d62600b0b`.
Future metrics can therefore be calculated without repeating inference. The
test route refuses a repeated submission, so this result cannot become a
model-selection signal.

Public-safe aggregate evidence is stored in:

- [`treelearn_finetuned_test_results_20260713.csv`](../examples/treelearn_finetuned_test_results_20260713.csv);
- [`treelearn_finetuned_test_site_results_20260713.csv`](../examples/treelearn_finetuned_test_site_results_20260713.csv); and
- [`treelearn_finetuned_test_provenance_20260713.json`](../examples/treelearn_finetuned_test_provenance_20260713.json).
