# TreeLearn completed FOR-instance development result

## Result identity

Run `treelearn_for-instance_published_pretrained_development_20260712_150030`
completed the frozen 21-plot FOR-instance development route on 12 July 2026.
All 21 plots completed fixed aligned point-wise evaluation, the final gate
passed, and there were no documented failures. This is a development-only
diagnostic result, not a held-out test result.

Subsequent provenance review found that the authors' December 2024 checkpoint
descends from a model fine-tuned with FOR-instance validation and test data.
This result remains valid as a published-checkpoint reproduction, but is
excluded from leakage-free cross-method ranking.

The run used:

- training mode `published_pretrained`;
- TreeLearn commit `fd240ce7caa4c444fe3418aca454dc578bc557d4`;
- benchmark commit `1a66696be033816d7f776d7e1293340356b34452`;
- checkpoint MD5 `56a3d78f689ae7f1190906b975700311`;
- checkpoint SHA-256
  `5df2f92828f92755bc12e114eaebe83f7ecea94a74c25a6170b68844cc5e19bb`;
- protocol `for_instance_pointwise_v1`; and
- maximum-cardinality one-to-one matching at IoU `>= 0.5` with no tuned
  prediction filtering.

## Overall development result

| Plots | Predictions | References | TP | FP | FN | Mean plot precision | Mean plot recall | Mean plot F1 | Micro precision | Micro recall | Micro F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 21 | 1,284 | 807 | 534 | 750 | 273 | 0.417910 | 0.732412 | 0.515571 | 0.415888 | 0.661710 | 0.510760 |

The 21 plots contain 101,769,037 source points; 84,136,822 points are in the
union of reference-tree and predicted-tree points used by the fixed evaluator.
Mean plot matched IoU is `0.830485`, the mean IoU across all matched pairs is
`0.827749`, mean unweighted coverage is `0.654057`, and mean weighted coverage
is `0.835697`.

## Site results

| Site | Plots | Predictions | References | TP | FP | FN | Mean plot F1 | Micro precision | Micro recall | Micro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CULS | 2 | 48 | 27 | 27 | 21 | 0 | 0.715010 | 0.562500 | 1.000000 | 0.720000 |
| NIBIO | 14 | 889 | 414 | 291 | 598 | 123 | 0.446965 | 0.327334 | 0.702899 | 0.446662 |
| RMIT | 1 | 126 | 159 | 84 | 42 | 75 | 0.589474 | 0.666667 | 0.528302 | 0.589474 |
| SCION | 3 | 106 | 92 | 64 | 42 | 28 | 0.652892 | 0.603774 | 0.695652 | 0.646465 |
| TUWIEN | 1 | 115 | 115 | 68 | 47 | 47 | 0.591304 | 0.591304 | 0.591304 | 0.591304 |

CULS has the highest mean plot F1 (`0.715010`) and NIBIO the lowest
(`0.446965`). NIBIO contains 14 of the 21 plots and 414 of the 807 reference
trees, so it has the largest influence on the overall count-aggregated result.
Overall recall exceeds precision, consistent with 750 false positives versus
273 false negatives. These differences are descriptive development results;
they were not used to change the checkpoint, threshold or post-processing.

## Retention

The retention gate verified all five prediction artefacts for every plot:
upstream full-forest LAZ and NPZ, upstream pointwise NPZ, aligned prediction
NPZ and aligned prediction LAS. In total, 105 files and 9,645,423,654 bytes
(about 9.65 GB or 8.98 GiB) remain in the run-scoped Barkla roots. Their sizes
and SHA-256 values remain in the uncommitted run retention manifest so future
metrics can be calculated without repeating inference.

Public-safe aggregate evidence is stored in:

- [`treelearn_completed_development_results_20260712.csv`](../examples/treelearn_completed_development_results_20260712.csv);
- [`treelearn_completed_development_site_results_20260712.csv`](../examples/treelearn_completed_development_site_results_20260712.csv); and
- [`treelearn_completed_development_provenance_20260712.json`](../examples/treelearn_completed_development_provenance_20260712.json).

No held-out test data were accessed by this published-checkpoint development
run, and it has no leakage-free test result. A later, separately frozen
fine-tuned checkpoint starting from the clean authors-released L1W weights has
completed an authorised one-time test; see the
[`fine-tuned test result`](finetuned_test_results_20260713.md). The two rows
remain separate because this December 2024 checkpoint has documented
FOR-instance training overlap.
