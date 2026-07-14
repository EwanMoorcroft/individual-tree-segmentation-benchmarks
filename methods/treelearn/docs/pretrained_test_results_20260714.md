# TreeLearn published-pretrained held-out test result

Run `treelearn_for-instance_published_pretrained_20260714_134109` completed the
authorised evaluation of the unchanged authors-released checkpoint on 14 July
2026. All 11 FOR-instance test plots completed and the retention gate verified
55 raw or aligned prediction files.

## Overall result

| Plots | Predictions | References | TP | FP | FN | Mean plot F1 | Micro precision | Micro recall | Micro F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | 366 | 323 | 34 | 332 | 289 | 0.078944 | 0.092896 | 0.105263 | 0.098694 |

The run used `for_instance_pointwise_v1`, the union evaluation mask, IoU
`>= 0.5` and maximum-cardinality one-to-one matching. It is directly
comparable with the other four headline rows.

## Site results

| Site | Plots | Predictions | References | TP | FP | FN | Mean plot F1 | Micro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CULS | 1 | 53 | 20 | 20 | 33 | 0 | 0.547945 | 0.547945 |
| NIBIO | 6 | 263 | 161 | 8 | 255 | 153 | 0.031183 | 0.037736 |
| RMIT | 1 | 26 | 64 | 6 | 20 | 58 | 0.133333 | 0.133333 |
| SCION | 2 | 3 | 43 | 0 | 3 | 43 | 0.000000 | 0.000000 |
| TUWIEN | 1 | 21 | 35 | 0 | 21 | 35 | 0.000000 | 0.000000 |

## Execution recovery and retention

The pinned upstream pipeline failed when `SCION/plot_31_annotated.las`
produced no initial clusters. The benchmark-owned execution recovery mapped
only unresolved labels to TreeLearn background `0` and skipped two empty
optional diagnostic visualizations. It did not change weights, thresholds,
grouping parameters or evaluator settings and performed no model selection.
The recovered plot therefore contributed zero predicted instances and 25
false negatives.

## Independent audit and interpretation

Audit job `9775494` re-hashed all 55 retained prediction files, recomputed the
per-plot and aggregate metrics and returned `AUDIT_PASS`. It reproduced TP
`34`, FP `332`, FN `289` and mean plot F1 `0.078944`. Reference-tree recall was
`0/74` below 10 m and `34/249` at or above 10 m. Of 289 unmatched references,
71 had zero best IoU and only eight had best IoU between `0.40` and `0.50`, so
the result is not explained by widespread matches narrowly missing the
threshold.

The authors' checkpoint documentation describes
`model_weights_finetuned.pth` as detecting trees above 10 m. The result is
therefore genuine for this exact clean checkpoint and pinned pipeline, but it
must not be interpreted as TreeLearn's best achievable performance on smaller
trees. The pinned upstream repository also does not package a complete
historical post-processing configuration specifically for this checkpoint;
alternative settings may be investigated on development data only and cannot
replace or tune this frozen test result.

Public-safe evidence:

- [`treelearn_pretrained_test_results_20260714.csv`](../examples/treelearn_pretrained_test_results_20260714.csv)
- [`treelearn_pretrained_test_site_results_20260714.csv`](../examples/treelearn_pretrained_test_site_results_20260714.csv)
- [`treelearn_pretrained_test_provenance_20260714.json`](../examples/treelearn_pretrained_test_provenance_20260714.json)
