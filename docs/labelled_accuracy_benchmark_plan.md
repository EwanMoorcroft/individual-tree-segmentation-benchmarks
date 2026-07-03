# Labelled Accuracy Benchmark Plan

## Primary Benchmark

SegmentAnyTree on FOR-instance is the primary labelled accuracy benchmark.
FOR-instance supplies `treeID` reference instances and semantic
`classification` labels, allowing point-wise instance IoU, precision, recall,
F1 and coverage evaluation.

The canonical pilot and all 32 files completed inference. The earlier
coordinate-rematched evaluation is provisional because the final export has
not been shown to preserve one output row per source point. Its diagnostic
values are documented in
[`segmentanytree_for_instance_results.md`](segmentanytree_for_instance_results.md).
The accepted accuracy evaluation remains pending.

## Split Control

Read `data_split_metadata.csv` before selecting any plot and preserve its split
label in metadata and metric tables.

- Use development plots for adapter checks, parameter selection and error
  analysis.
- Keep evaluation plots isolated from training and parameter selection.
- Fix semantic filters, point correspondence, matching policy and IoU threshold
  before evaluating the test set.
- Record the model checkpoint, external commit and container route.

## Revalidation Sequence

1. Record the supplied checkpoint SHA-256 and confirm its upstream provenance.
2. Inventory the internal semantic and instance evaluation PLY files retained
   by the inference workflow.
3. Audit final LAZ point counts and coordinate multiplicity against every
   source plot.
4. Evaluate point-aligned labels using the released SegmentAnyTree matching
   policy.
5. Evaluate the same labels with a strict one-to-one policy for cross-method
   comparison.
6. Use development plots only for diagnostics and report the supplied test
   split as the primary paper comparison.
7. Rebuild the workbook and public-safe tables after all validation gates pass.

Every accepted accuracy row records reference and prediction counts, TP, FP,
FN, precision, recall, F1, coverage, matched IoU, matching policy, runtime,
peak memory, checkpoint identity, thresholds and semantic masks.

## Next Validation Priorities

1. Inspect one CULS and one NIBIO output to identify the exact internal aligned
   files before submitting a full CPU evaluation.
2. Quantify output point-count differences and duplicate-coordinate conflicts
   by collection.
3. Determine whether the NIBIO collapse originates in the final export,
   semantic mapping, checkpoint/configuration or the model prediction.
4. Freeze the evaluation configuration before reading new test results.
5. Run a second method against the same harmonized references without
   replacing its method-specific reproduction metrics.

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
