# Labelled Accuracy Benchmark Plan

## Current Priority

SegmentAnyTree on FOR-instance is the primary labelled accuracy benchmark.
FOR-instance supplies `treeID` reference instances and semantic
`classification` labels, allowing one-to-one precision, recall, F1 and matched
IoU evaluation.

The development pilot on `CULS/plot_1_annotated.las` has completed inference,
normalisation and evaluation. It contains 1,816,672 input points and six
positive reference trees. The workflow includes classes `4`, `5` and `6`,
ignores classes `0`, `1`, `2` and `3`, and ignores non-positive tree IDs.

The full 32-file benchmark remains pending. Before submission, the consolidated
Apptainer pilot must reproduce the repaired output, the 15 unmatched pilot
predictions should be inspected, and per-plot point-count and coordinate
matching checks should be retained.

## Split Control

Read `data_split_metadata.csv` before selecting any plot and preserve its split
label in metadata and metric tables.

- Use development plots for adapter checks, parameter selection and error
  analysis.
- Keep evaluation plots isolated from training and parameter selection.
- Fix semantic filters, normalisation, coordinate tolerance and IoU threshold
  before evaluating the full set.
- Record the model checkpoint, external commit and container route.

## Full-Run Sequence

1. Re-run the canonical Apptainer pilot and confirm it produces the labelled
   LAZ without a separate postprocessing repair.
2. Check required dimensions, positive `PredInstance` values, output point
   count and coordinate alignment.
3. Normalise the labelled point cloud to one XYZ PLY per positive predicted
   instance.
4. Re-evaluate the pilot against the unchanged source LAS with a 0.02 m
   coordinate tolerance and 0.5 IoU threshold.
5. Submit prediction, normalisation and evaluation arrays for all 32 LAS files.
6. Resolve failed or missing tasks before building collection, split and
   overall summaries.
7. Report development and evaluation splits separately and avoid parameter
   selection on evaluation results.

Every accuracy row must record reference and prediction counts, TP, FP, FN,
precision, recall, F1, mean and median matched IoU, runtime, peak memory,
thresholds and ignored classes.

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
