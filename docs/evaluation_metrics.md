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

The TLS2trees leaf-off target uses reference classes `4` and `6`; class `5` is
background for that woody-only diagnostic. Its completed leaf-on target uses
classes `4`, `5` and `6` and a distinct leaf-attachment output. Both exclude
class-3 out-points before forming the reference/prediction union mask. They are
therefore reported in a separate protocol group from the shared pointwise
rows. Here “leaf-on”/“leaf-off” describes prediction and scoring material, not
acquisition season.

The SegmentAnyTree/FOR-instance workflow uses classes `4`, `5` and `6`.
Inference produces internal aligned prediction arrays and a final labelled LAZ
with `PredInstance`. The final export must pass row-preservation checks before
it can be used for point-wise accuracy. The internal aligned arrays are the
preferred reproduction source. Every completed audit in the first test-split
snapshot failed row-preservation checks, so comparable evaluation must use
aligned arrays written before final export merging.

## Instance Accuracy Readiness

| Dataset | Reference representation | Instance accuracy status |
| --- | --- | --- |
| FRDR treeiso TLS | No tree IDs in the benchmark LAZ inputs | F1, precision, recall and IoU unavailable from `woods` |
| FOR-instance | Positive `treeID` values in annotated LAS files | F1 and matched IoU feasible |
| Wytham Woods | One segmented reference tree per PLY filename | F1 and matched IoU feasible after scene reconstruction |

FOR-instance feasibility alone does not make every row ranking-comparable.
Prediction material, reference scoring mask and leaf-attachment state must be
recorded, and rows with different masks remain in separate comparison groups.

The legacy coordinate-rematched SegmentAnyTree values, rejected from accepted
accuracy reporting, are stored outside Git under
`results/tables/segmentanytree_for_instance/per_plot/`, with evaluation
metadata under `results/metadata/segmentanytree_for_instance/`. Public-safe
per-plot, collection, split, matched-pair and inventory diagnostics are
retained in [`SegmentAnyTree examples`](../methods/segmentanytree/examples/),
and the interpretation is documented in
[`provisional_released_checkpoint_results.md`](../methods/segmentanytree/docs/provisional_released_checkpoint_results.md).
These coordinate-rematched values are not an aligned paper reproduction and
remain rejected from accepted accuracy reporting. The current
`published_pretrained` and `fine_tuned_on_dev` targets write separate
predictions, metadata and held-out test metrics so their identities cannot be
mixed. The historical
`retrained_from_dev` result is represented publicly by its
[`final aggregate`](../methods/segmentanytree/examples/sat_final_test_aligned_summary_sat_for_quicktune_to49_20260706_140730.csv)
and [`provenance manifest`](../methods/segmentanytree/examples/sat_final_test_aligned_provenance_sat_for_quicktune_to49_20260706_140730.json).
The final per-plot table has not been transferred into the local checkout, so
older diagnostic rows must not be used to reconstruct it.

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
- a maximum-cardinality strict one-to-one assignment for cross-method
  comparison;
- mean unweighted and point-weighted coverage; and
- accepted match identifiers and IoUs.

The evaluation mask is the union of predicted-tree and reference-tree points.
The accepted threshold is IoU greater than or equal to 0.5, matching the
released SegmentAnyTree evaluator. Primary comparable results use only the
supplied test split. Development plots are retained for diagnostics.

The full protocol is
[`for-instance.md`](protocols/for-instance.md).

## Diagnostic Thresholds And Plot Summaries

The canonical protocol remains IoU `>= 0.5`. The shared diagnostic helper in
[`diagnostic_metrics.py`](../src/benchmark/diagnostic_metrics.py) can evaluate
the same aligned prediction/reference arrays at thresholds `0.25`, `0.50` and
`0.75`. It reports precision, recall and F1 at each threshold plus mean/median
matched IoU, unmatched counts and overlap-based split/merge indicators.

Across plots, diagnostic reporting may add median plot F1, plot-F1
interquartile range, number of zero-F1 plots and per-site macro summaries.
These diagnostics must not be used for checkpoint, threshold or parameter
selection and must not overwrite the canonical `0.50` point estimate.

Availability depends on retained evidence. Per-plot aggregate CSVs can support
plot-level robust summaries and some bootstrap calculations, but cannot
reconstruct alternate thresholds, matched pairs or pointwise semantic errors.
Those require accessible aligned predictions or equivalent retained overlap
evidence. Barkla-only retention is reported as unavailable locally rather than
treated as a zero. Current support is declared in
[`diagnostic_metric_availability.csv`](../outputs/for_instance_benchmark_metrics/diagnostic_metric_availability.csv).
Committed plot rows support aggregate distribution/bootstrap diagnostics.
The registry identifies point-aligned TreeX and TreeLearn retention evidence,
but records that raw inputs were not accessed during this cleanup.
SegmentAnyTree requires its retained Barkla artefacts, and TLS2trees requires
its Barkla NPZ plus the exact source LAS. This is an availability statement,
not permission to inspect or rerun held-out work.

## Plot-Level Bootstrap Intervals

[`result_statistics.py`](../src/benchmark/result_statistics.py) implements
bootstrap intervals by resampling whole plots. It never treats trees or points
from the same plot as independent samples. Defaults are seed `20260721`,
`10000` iterations and a `0.95` confidence level; iteration count and seed are
recorded/configurable. Point estimates remain in their canonical tables and
confidence intervals are written separately.

Optional `site` stratification resamples within sites. The test subset contains
only 11 plots and several sites have one plot, so a stratified interval can
have little or no within-site variation. It is a sensitivity description of
this benchmark subset, not a population-level guarantee.

## Semantic And Instance Error Decomposition

When independent prediction semantics and aligned instances are available,
diagnostics use these definitions:

- **semantic omission:** reference tree-material points predicted as non-tree;
- **semantic commission:** reference non-tree points predicted as tree;
- **unassigned reference-tree points:** reference tree-material points with no
  positive predicted instance;
- **semantic commission points:** points predicted semantically as tree outside
  the reference tree-material mask;
- **predicted-instance points on reference background:** points assigned a
  positive predicted instance outside the reference tree-material mask;
- **instance splitting:** one reference instance overlaps multiple predicted
  instances at or above the recorded minimum-intersection rule;
- **instance merging:** one predicted instance overlaps multiple reference
  instances at or above that rule;
- **unmatched predicted instances:** predictions without an accepted
  one-to-one match at the stated IoU threshold; and
- **unmatched reference instances:** references without an accepted match.

Semantic-omission rate uses reference-tree points as its denominator;
semantic-commission rate uses predicted-tree points. A zero denominator yields
a null rate, not zero.

The default split/merge diagnostic counts any positive point intersection
(`min_intersection_points=1`), so it is deliberately sensitive and must report
that setting. It does not redefine TP, FP or FN. A semantic field derived from
whether an instance ID is positive cannot identify whether an omission arose
in a semantic network or in grouping/post-processing; such method/stage fields
are `unsupported` or `not_recorded`.

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
- explicitly named mean plot and count-aggregated micro metrics;
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
Split/merge diagnostics use the explicit overlap rule above and remain
separate from the maximum-cardinality one-to-one primary assignment.
