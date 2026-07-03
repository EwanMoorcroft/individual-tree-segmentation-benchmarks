# SegmentAnyTree FOR-instance Results

## Status

Inference completed for all 32 files, but the accuracy values below are
**provisional and must not be cited as the final SegmentAnyTree benchmark**.
They were calculated by splitting the final exported LAZ into predicted trees
and matching rounded coordinates back to the source LAS.

This route does not reproduce the published aligned point-wise evaluation.
The NIBIO F1 differs from the published result by approximately 0.87, and the
initial CULS export contained more rows than its source file. The accuracy
evaluation is therefore being repeated using
[`for_instance_pointwise_v1`](for_instance_cross_method_protocol.md).

## Completed Workflow Scope

SegmentAnyTree prediction, normalisation and labelled instance evaluation
completed for all 32 annotated FOR-instance LAS files. The run includes five
collections, 21 development plots and 11 test plots. Every plot has a completed
prediction and normalisation record. The provisional coordinate evaluator also
produced one row per plot; no failed or missing plot is included in that
diagnostic summary.

The benchmark evaluated 151,478,959 input points and 1,130 positive reference
trees. Reference instances use `treeID`. Reference points are restricted to
semantic classes `4`, `5` and `6`; classes `0`, `1`, `2` and `3`, and
non-positive tree IDs, are ignored. Predictions and references were matched
one-to-one at an IoU threshold of 0.5 after coordinate quantisation at 0.02 m.

## Provisional Coordinate-Rematched Results

| Metric | Result |
| --- | ---: |
| Evaluated plots | 32 |
| Reference trees | 1,130 |
| Predicted trees | 2,532 |
| True positives | 376 |
| False positives | 2,156 |
| False negatives | 754 |
| Micro precision | 0.148499 |
| Micro recall | 0.332743 |
| Micro F1 | 0.205352 |
| Mean plot F1 | 0.195955 |
| Matched pairs | 376 |
| Pooled mean matched IoU | 0.726375 |
| Pooled median matched IoU | 0.744533 |
| Cumulative per-plot runtime | 13,430 s |
| Mean per-plot runtime | 419.688 s |
| Maximum recorded task memory | 9.608 GiB |

These values describe the coordinate-rematched route, not the accepted
paper-aligned result. The pooled IoU values describe the 376 accepted matches
only. They therefore
measure overlap quality after a prediction has been matched and must be read
alongside the low detection precision and recall. The cumulative runtime is
the sum of per-plot task runtimes, not the wall-clock duration of the Slurm
array.

## Provisional Results By Collection

| Collection | Plots | References | Predictions | TP | FP | FN | Precision | Recall | F1 | Pooled matched IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CULS | 3 | 47 | 142 | 47 | 95 | 0 | 0.330986 | 1.000000 | 0.497354 | 0.841255 |
| NIBIO | 20 | 575 | 1,591 | 12 | 1,579 | 563 | 0.007542 | 0.020870 | 0.011080 | 0.515160 |
| RMIT | 2 | 223 | 288 | 145 | 143 | 78 | 0.503472 | 0.650224 | 0.567515 | 0.716819 |
| SCION | 5 | 135 | 224 | 107 | 117 | 28 | 0.477679 | 0.792593 | 0.596100 | 0.745585 |
| TUWIEN | 2 | 150 | 287 | 65 | 222 | 85 | 0.226481 | 0.433333 | 0.297483 | 0.671995 |

Collection-level performance is highly uneven. NIBIO accounts for 575
reference trees but only 12 accepted matches; 16 of its 20 plots have no
accepted match. The SegmentAnyTree paper reports NIBIO F1 near 0.88 under its
aligned evaluation, compared with 0.011 here. The discrepancy is treated as an
evaluation failure until the internal prediction arrays and final export have
been audited.

## Provisional Results By Supplied Split

| Split | Plots | References | Predictions | TP | FP | FN | Precision | Recall | F1 | Pooled matched IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Development | 21 | 807 | 1,743 | 265 | 1,478 | 542 | 0.152037 | 0.328377 | 0.207843 | 0.720764 |
| Test | 11 | 323 | 789 | 111 | 678 | 212 | 0.140684 | 0.343653 | 0.199640 | 0.739771 |

The split labels are reported for transparency. These diagnostic values include
development and test data. The final comparable headline result will use the
test split only.

## Provisional Public-Safe Files

- [`segmentanytree_for_instance_full_results.xlsx`](../examples/segmentanytree_for_instance_full_results.xlsx):
  formatted workbook containing the overall, per-plot, collection, split,
  matched-pair and inventory tables, together with the completed FRDR
  operational benchmark.
- [`segmentanytree_for_instance_full_plot_metrics.csv`](../examples/segmentanytree_for_instance_full_plot_metrics.csv):
  one row per FOR-instance plot.
- [`segmentanytree_for_instance_full_summary.csv`](../examples/segmentanytree_for_instance_full_summary.csv):
  overall counts, metrics and resources.
- [`segmentanytree_for_instance_full_summary_by_collection.csv`](../examples/segmentanytree_for_instance_full_summary_by_collection.csv):
  collection-level results.
- [`segmentanytree_for_instance_full_summary_by_split.csv`](../examples/segmentanytree_for_instance_full_summary_by_split.csv):
  development/test summaries.
- [`segmentanytree_for_instance_full_matches.csv`](../examples/segmentanytree_for_instance_full_matches.csv):
  all 376 accepted prediction-reference matches and their IoUs.
- [`segmentanytree_for_instance_full_inventory.csv`](../examples/segmentanytree_for_instance_full_inventory.csv):
  public-safe plot inventory fields used by the report.

These files exclude coordinates, point clouds, predictions, checkpoints,
absolute machine paths and scheduler logs. They are retained to document the
failed coordinate-rematching route and will be replaced after revalidation.
The ignored working results remain under `results/`, `data/` and `logs/` on
Barkla.

The CSV publication step is reproducible with
[`build_segmentanytree_public_results.py`](../scripts/reporting/build_segmentanytree_public_results.py),
using a transferred copy of the ignored full result tables as its input.

## Interpretation And Remaining Validation

The run establishes that full inference is operational. Accuracy acceptance
requires:

1. inventorying the internal aligned semantic and instance PLY files;
2. auditing source and final-export point counts and coordinate multiplicity;
3. evaluating aligned point labels with the released matching policy;
4. reporting a separate harmonised one-to-one test-split result;
5. recording the checkpoint checksum and training-data provenance; and
6. replacing these provisional tables only after the reproduction gates pass.

No FRDR accuracy values are combined with these results because the FRDR inputs
do not contain individual-tree reference IDs.
