# Evaluation Metrics

Metrics are grouped by the evidence they require. Operational and semantic
checks do not substitute for instance accuracy.

## Prediction And Operational Metrics

These metrics can be recorded without individual-tree reference labels:

- input, retained and dropped point counts;
- predicted tree count and points assigned to predictions;
- minimum, maximum and mean points per predicted tree;
- output-file validity and completion status;
- command, method version and return code;
- runtime and peak memory.

The completed FRDR/TLS2trees benchmark reports these measures. They describe
execution and predicted outputs, not segmentation accuracy.

## Semantic Consistency Checks

Semantic checks validate dataset preparation and method inputs:

- required field presence and data type;
- observed values and counts for each semantic class;
- mapping from source classes to method classes;
- unknown, ignored and dropped point counts;
- consistency between positive instance IDs and included tree-material classes;
- preservation of coordinates and point counts through conversion.

For FRDR, `woods = 1` means wood and `woods = 2` means non-wood. These values
cannot identify individual trees. FOR-instance uses `classification` values
`4`, `5` and `6` for stem, live branches and woody branches respectively; the
initial tree-only evaluation ignores classes `0`, `1`, `2` and `3`.

The first FOR-instance/TLS2trees pilot is narrower: its leaf-off reference uses
classes `4` and `6` only. Class `5` is excluded to avoid penalising leaf-off
predictions for missing live-branch points. A future leaf-on experiment must be
configured and reported separately.

The SegmentAnyTree/FOR-instance workflow uses classes `4`, `5` and `6`.
Inference produces internal aligned prediction arrays and a final labelled LAZ
with `PredInstance`. The final export must pass row-preservation checks before
it can be used for point-wise accuracy. The internal aligned arrays are the
preferred reproduction source. Every completed audit in the first test-split
snapshot failed row-preservation checks, so the accepted evaluation must use
aligned arrays written before final export merging.

## Instance Accuracy Readiness

| Dataset | Reference representation | Instance accuracy status |
| --- | --- | --- |
| FRDR treeiso TLS | No tree IDs in the benchmark LAZ inputs | F1, precision, recall and IoU unavailable from `woods` |
| FOR-instance | Positive `treeID` values in annotated LAS files | F1 and matched IoU feasible |
| Wytham Woods | One segmented reference tree per PLY filename | F1 and matched IoU feasible after scene reconstruction |

The first SegmentAnyTree metrics are stored outside Git under
`results/tables/segmentanytree_for_instance/per_plot/`, with evaluation
metadata under `results/metadata/segmentanytree_for_instance/`. Public-safe
per-plot, collection, split, matched-pair and inventory tables are retained in
[`SegmentAnyTree examples`](../methods/segmentanytree/examples/), and the interpretation is documented in
[`provisional_released_checkpoint_results.md`](../methods/segmentanytree/docs/provisional_released_checkpoint_results.md).
These coordinate-rematched values are provisional and are not the accepted
paper reproduction or trained-model result. The corrected
`retrained_from_dev` experiment writes separate training metadata,
development-validation metrics and held-out test metrics so results from the
released checkpoint cannot be mixed with results from the new checkpoint.

Summary tables distinguish two IoU aggregations:

- the mean of per-plot matched-IoU means, where plots without accepted matches
  contribute zero in the recorded plot table; and
- pooled matched IoU, calculated directly across all accepted matched pairs.

The pooled value describes match quality only. It must be reported with
precision, recall and F1 because unmatched predictions and references do not
contribute an IoU value.

The coordinate evaluator in
[`shared/evaluation/instance_iou_f1.py`](../shared/evaluation/instance_iou_f1.py)
intentionally refuses to run without a reference instance source:

```text
No reference instance labels supplied; IoU/F1 cannot be computed.
```

## Aligned Point-Wise Evaluation

The FOR-instance primary protocol requires one predicted instance label and
one reference `treeID` for each evaluated point. Point correspondence must
come from source row order or a stable point identifier, not rounded
coordinates.

[`pointwise_instance_metrics.py`](../methods/segmentanytree/scripts/evaluation/pointwise_instance_metrics.py)
reports:

- the released SegmentAnyTree per-prediction-best-IoU policy for paper
  reproduction;
- a strict one-to-one assignment for cross-method comparison;
- mean unweighted and point-weighted coverage; and
- accepted match identifiers and IoUs.

The evaluation mask is the union of predicted-tree and reference-tree points.
The accepted threshold is IoU greater than or equal to 0.5, matching the
released SegmentAnyTree evaluator. Primary comparable results use only the
supplied test split. Development plots are retained for diagnostics.

The full protocol is
[`for-instance.md`](protocols/for-instance.md).

## Coordinate-Based One-To-One Evaluation

The evaluator accepts:

1. a prediction directory containing one LAS, LAZ or PLY file per predicted
   tree; and
2. either a reference directory containing one file per tree or one labelled
   point cloud containing an individual-tree ID field.

This fallback is intended for methods that provide separate tree point clouds
without stable source-point identifiers. Predicted and reference coordinates
are quantised using the configured coordinate tolerance. Point-set IoU for
each candidate pair is:

```text
IoU = intersection point count / union point count
```

Predicted and reference trees are matched one-to-one only when their IoU meets
the configured threshold. With `TP` matched predictions, `FP` unmatched
predictions and `FN` unmatched references:

```text
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1 = 2 * precision * recall / (precision + recall)
```

F1 must be calculated only after this instance matching. Semantic wood/non-wood
agreement is not an instance F1 score. Coordinate-based and aligned point-wise
results must use different evaluation-mode labels and must not be combined.

## Required Accuracy Result Fields

Every labelled accuracy run must record:

- reference tree count;
- predicted tree count;
- true positives, false positives and false negatives;
- precision, recall and F1;
- mean and median matched IoU;
- IoU threshold, comparison operator and point-correspondence mode;
- coordinate tolerance when coordinate matching is used;
- reference instance field or filename rule;
- ignored semantic classes and instance labels;
- dataset split and reference provenance;
- runtime and peak memory;
- method version, command and parameter configuration;
- checkpoint checksum, training mode and training-data declaration.

No FRDR instance accuracy value should be reported unless an external,
documented individual-tree reference is supplied and evaluated.

The evaluator also writes matched-pair, unmatched-prediction and
unmatched-reference tables so TP, FP and FN assignments can be reviewed.
Over-segmentation and under-segmentation counts are not reported because the
current evaluator has no fixed many-to-one or one-to-many definition.
