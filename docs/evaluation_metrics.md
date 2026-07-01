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

The SegmentAnyTree/FOR-instance benchmark uses classes `4`, `5` and `6`. Its
adapter must first normalise a verified method output to one PLY per predicted
tree; it does not assume an undocumented SegmentAnyTree instance field.

## Instance Accuracy Readiness

| Dataset | Reference representation | Instance accuracy status |
| --- | --- | --- |
| FRDR treeiso TLS | No tree IDs in the benchmark LAZ inputs | F1, precision, recall and IoU unavailable from `woods` |
| FOR-instance | Positive `treeID` values in annotated LAS files | F1 and matched IoU feasible |
| Wytham Woods | One segmented reference tree per PLY filename | F1 and matched IoU feasible after scene reconstruction |

The evaluator in
[`scripts/evaluation/instance_iou_f1.py`](../scripts/evaluation/instance_iou_f1.py)
intentionally refuses to run without a reference instance source:

```text
No reference instance labels supplied; IoU/F1 cannot be computed.
```

## One-To-One Instance Evaluation

The evaluator accepts:

1. a prediction directory containing one LAS, LAZ or PLY file per predicted
   tree; and
2. either a reference directory containing one file per tree or one labelled
   point cloud containing an individual-tree ID field.

Predicted and reference coordinates are quantised using the configured
coordinate tolerance. Point-set IoU for each candidate pair is:

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
agreement is not an instance F1 score.

## Required Accuracy Result Fields

Every labelled accuracy run must record:

- reference tree count;
- predicted tree count;
- true positives, false positives and false negatives;
- precision, recall and F1;
- mean and median matched IoU;
- IoU threshold and coordinate tolerance;
- reference instance field or filename rule;
- ignored semantic classes and instance labels;
- dataset split and reference provenance;
- runtime and peak memory;
- method version, command and parameter configuration.

No FRDR instance accuracy value should be reported unless an external,
documented individual-tree reference is supplied and evaluated.

The evaluator also writes matched-pair, unmatched-prediction and
unmatched-reference tables so TP, FP and FN assignments can be reviewed.
Over-segmentation and under-segmentation counts are not reported because the
current evaluator has no fixed many-to-one or one-to-many definition.
