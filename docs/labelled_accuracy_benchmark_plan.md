# Labelled Accuracy Benchmark Plan

## Primary Benchmark

SegmentAnyTree on FOR-instance is the primary labelled accuracy benchmark.
FOR-instance supplies `treeID` reference instances and semantic
`classification` labels, allowing point-wise instance IoU, precision, recall,
F1 and coverage evaluation.

The canonical pilot and all 32 files completed inference with the released
checkpoint. The earlier coordinate-rematched evaluation is provisional
because the final export does not preserve one output row per source point and
the model was not trained under the local FOR-instance protocol. Its
diagnostic values are documented in
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

## Corrected SegmentAnyTree Sequence

1. Reproduce the upstream seed-42 split of development plots into 16 training
   and 5 validation plots.
2. Convert development LAS files to the upstream Treeins PLY schema and verify
   that no test file enters the training root.
3. Run the two-epoch, three-plot training preflight.
4. Load its checkpoint for inference on the selected development validation
   plot and evaluate aligned point-wise labels.
5. If the complete preflight succeeds, prepare all development plots and train
   the ULS-only model from scratch.
6. Select and freeze the checkpoint using only the five development validation
   plots.
7. Run inference once on all 11 held-out test plots.
8. Report the released matching policy and a strict one-to-one policy
   separately.
9. Rebuild the workbook and public-safe tables only after all gates pass.

Every accepted accuracy row records reference and prediction counts, TP, FP,
FN, precision, recall, F1, coverage, matched IoU, matching policy, runtime,
peak memory, checkpoint identity, thresholds and semantic masks.

## Next Validation Priorities

1. Submit the development-only preparation and training preflight.
2. Confirm a non-empty `PointGroup-PAPER.pt` checkpoint and recorded checksum.
3. Confirm equal lengths for semantic prediction, semantic reference, instance
   prediction and instance reference arrays on development validation data.
4. Submit full development training only after those checks pass.
5. Add the next method against the same fixed development/test boundary and
   harmonised evaluator.

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
