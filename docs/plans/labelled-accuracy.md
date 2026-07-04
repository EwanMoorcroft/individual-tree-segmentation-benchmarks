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
[`provisional_released_checkpoint_results.md`](../../methods/segmentanytree/docs/provisional_released_checkpoint_results.md).
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
3. Use the completed short training pilot to confirm the adapter and
   checkpoint route.
4. Train the ULS-only model from scratch on all 16 training plots.
5. Evaluate each candidate checkpoint on the five fixed development
   validation plots using aligned point-wise labels.
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

1. Let the active 16-plot full training run finish without changing its Barkla
   working tree.
2. Record the checkpoint checksum and complete run metadata.
3. Confirm equal lengths for semantic prediction, semantic reference, instance
   prediction and instance reference arrays on all five validation plots.
4. Select and freeze the checkpoint using development validation only.
5. Submit the 11 held-out test plots once, then add the next method against the
   same split boundary and harmonised evaluator.

## Other Candidate Work

TLS2trees on FOR-instance remains a compatibility experiment rather than the
current priority because the dataset is UAV laser scanning. Its leaf-off pilot
uses classes `4` and `6`, which must not replace the SegmentAnyTree class
definition.

Wytham Woods remains a future TLS accuracy candidate. Its 876 per-tree PLY files
must first be reconstructed into a documented plot scene with preserved
reference IDs and common method inputs. No Wytham benchmark has been run.

The detailed runbook is
[`for_instance_benchmark.md`](../../methods/segmentanytree/docs/for_instance_benchmark.md).
