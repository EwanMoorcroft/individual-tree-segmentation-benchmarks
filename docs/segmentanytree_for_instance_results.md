# SegmentAnyTree FOR-instance Results

## Scope And Completion

SegmentAnyTree prediction, normalisation and labelled instance evaluation
completed for all 32 annotated FOR-instance LAS files. The run includes five
collections, 21 development plots and 11 test plots. Every plot has a completed
prediction, normalisation record and evaluation row; no failed or missing plot
is included in the summary.

The benchmark evaluated 151,478,959 input points and 1,130 positive reference
trees. Reference instances use `treeID`. Reference points are restricted to
semantic classes `4`, `5` and `6`; classes `0`, `1`, `2` and `3`, and
non-positive tree IDs, are ignored. Predictions and references were matched
one-to-one at an IoU threshold of 0.5 after coordinate quantisation at 0.02 m.

## Overall Results

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

The pooled IoU values describe the 376 accepted matches only. They therefore
measure overlap quality after a prediction has been matched and must be read
alongside the low detection precision and recall. The cumulative runtime is
the sum of per-plot task runtimes, not the wall-clock duration of the Slurm
array.

## Results By Collection

| Collection | Plots | References | Predictions | TP | FP | FN | Precision | Recall | F1 | Pooled matched IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CULS | 3 | 47 | 142 | 47 | 95 | 0 | 0.330986 | 1.000000 | 0.497354 | 0.841255 |
| NIBIO | 20 | 575 | 1,591 | 12 | 1,579 | 563 | 0.007542 | 0.020870 | 0.011080 | 0.515160 |
| RMIT | 2 | 223 | 288 | 145 | 143 | 78 | 0.503472 | 0.650224 | 0.567515 | 0.716819 |
| SCION | 5 | 135 | 224 | 107 | 117 | 28 | 0.477679 | 0.792593 | 0.596100 | 0.745585 |
| TUWIEN | 2 | 150 | 287 | 65 | 222 | 85 | 0.226481 | 0.433333 | 0.297483 | 0.671995 |

Collection-level performance is highly uneven. NIBIO accounts for 575
reference trees but only 12 accepted matches; 16 of its 20 plots have no
accepted match. Four NIBIO plots do produce matches, and the other collections
complete under the same coordinate and evaluation workflow. This pattern
requires collection-specific validation of segmentation behaviour, coordinate
alignment and reference compatibility before it is attributed solely to model
performance.

## Results By Supplied Split

| Split | Plots | References | Predictions | TP | FP | FN | Precision | Recall | F1 | Pooled matched IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Development | 21 | 807 | 1,743 | 265 | 1,478 | 542 | 0.152037 | 0.328377 | 0.207843 | 0.720764 |
| Test | 11 | 323 | 789 | 111 | 678 | 212 | 0.140684 | 0.343653 | 0.199640 | 0.739771 |

The split labels are reported for transparency. Test plots were not used to
select the semantic filter, IoU threshold or coordinate tolerance.

## Public-Safe Result Files

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
absolute machine paths and scheduler logs. The ignored working results remain
under `results/`, `data/` and `logs/` on Barkla.

The CSV publication step is reproducible with
[`build_segmentanytree_public_results.py`](../scripts/reporting/build_segmentanytree_public_results.py),
using a transferred copy of the ignored full result tables as its input.

## Interpretation And Remaining Validation

The run establishes that the full deployment and evaluation workflow is
operational, but the aggregate F1 is not sufficient on its own to explain the
method's behaviour. Immediate follow-up work should:

1. inspect NIBIO prediction and reference overlays on development plots;
2. quantify the upstream output point-count differences for every collection;
3. review unmatched predictions for over-segmentation and duplicate crowns;
4. retain the fixed evaluation protocol while investigating collection
   compatibility; and
5. compare against a second method using the same references and split policy.

No FRDR accuracy values are combined with these results because the FRDR inputs
do not contain individual-tree reference IDs.
