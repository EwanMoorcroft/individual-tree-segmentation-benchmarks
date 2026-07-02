# Labelled Accuracy Benchmark Plan

## Completed Primary Benchmark

SegmentAnyTree on FOR-instance is the primary labelled accuracy benchmark.
FOR-instance supplies `treeID` reference instances and semantic
`classification` labels, allowing one-to-one precision, recall, F1 and matched
IoU evaluation.

The canonical pilot and the full 32-file benchmark have completed inference,
normalisation and evaluation. The workflow includes classes `4`, `5` and `6`,
ignores classes `0`, `1`, `2` and `3`, and ignores non-positive tree IDs.
Results are reported in
[`segmentanytree_for_instance_results.md`](segmentanytree_for_instance_results.md).

## Split Control

Read `data_split_metadata.csv` before selecting any plot and preserve its split
label in metadata and metric tables.

- Use development plots for adapter checks, parameter selection and error
  analysis.
- Keep evaluation plots isolated from training and parameter selection.
- Fix semantic filters, normalisation, coordinate tolerance and IoU threshold
  before evaluating the full set.
- Record the model checkpoint, external commit and container route.

## Completed Full-Run Sequence

1. The canonical Apptainer pilot produced its labelled LAZ without a separate
   repair job.
2. The labelled output was normalised to one XYZ PLY per positive predicted
   instance.
3. Prediction, normalisation and evaluation arrays completed for all 32 LAS
   files.
4. Collection, split and overall summaries were built after checking for
   missing or failed tasks.
5. Development and test split results were reported separately.

Every accuracy row records reference and prediction counts, TP, FP, FN,
precision, recall, F1, mean and median matched IoU, runtime, peak memory,
thresholds and ignored classes.

## Next Validation Priorities

1. Inspect NIBIO development plots, where 16 of 20 plots have no accepted
   match under the fixed evaluation protocol.
2. Quantify output point-count differences and coordinate coverage by
   collection.
3. Review unmatched predictions for over-segmentation.
4. Run a second method against the same references without changing the
   SegmentAnyTree result protocol.
5. Keep test plots out of any subsequent parameter adjustment.

## Other Candidate Work

TLS2trees on FOR-instance remains a compatibility experiment rather than the
current priority because the dataset is UAV laser scanning. Its leaf-off pilot
uses classes `4` and `6`, which must not replace the SegmentAnyTree class
definition.

Wytham Woods remains a future TLS accuracy candidate. Its 876 per-tree PLY files
must first be reconstructed into a documented plot scene with preserved
reference IDs and common method inputs. No Wytham benchmark has been run.

The detailed runbook is
[`segmentanytree_for_instance_benchmark.md`](segmentanytree_for_instance_benchmark.md).
