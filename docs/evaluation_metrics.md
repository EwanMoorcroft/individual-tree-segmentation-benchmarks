# Evaluation Metrics

## Metrics Available For FRDR

The completed FRDR/TLS2trees run supports prediction and operational reporting:

- input, retained and dropped point counts;
- wood and non-wood point counts used during conversion;
- predicted tree count;
- points assigned to predicted trees;
- minimum, maximum and mean points per predicted tree;
- process runtime and peak memory;
- return code, completion status and output validation.

These measures describe execution and predicted outputs. They are not
segmentation accuracy measures.

## Why Instance Accuracy Is Not Reported

The FRDR `woods` field is a semantic binary label:

- `woods = 1`: wood;
- `woods = 2`: non-wood.

It does not assign a unique tree identifier to each point. A wood point cannot
therefore be matched to a reference tree instance using `woods` alone. F1,
precision, recall and intersection over union (IoU) are not computed for this
benchmark.

The evaluator in
[`scripts/evaluation/instance_iou_f1.py`](../scripts/evaluation/instance_iou_f1.py)
returns an error when no reference instance source is supplied:

```text
No reference instance labels supplied; IoU/F1 cannot be computed.
```

## Evaluation With Reference Instances

If suitable reference labels become available, the evaluator accepts either:

1. a directory containing one LAS, LAZ or PLY file per reference tree; or
2. one labelled LAS, LAZ or PLY point cloud containing an individual-tree ID
   field.

Predicted and reference coordinates are quantised using the configured
coordinate tolerance. For each predicted-reference pair, point-set IoU is:

```text
IoU = intersection point count / union point count
```

The evaluator performs one-to-one matching for pairs meeting the configured
IoU threshold. With `TP` matched predictions, `FP` unmatched predictions and
`FN` unmatched references:

```text
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1 = 2 * precision * recall / (precision + recall)
```

Reference provenance, instance field, ignored labels, coordinate tolerance and
IoU threshold must be recorded with any reported accuracy result.
