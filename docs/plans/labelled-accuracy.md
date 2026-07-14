# Labelled Accuracy Benchmark Status

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
The aligned development-trained evaluation is complete and retained as
historical evidence. Run `sat_for_quicktune_to49_20260706_140730` has 11-plot
held-out mean plot F1 `0.4825` and micro F1 `0.4692`; it is not a target row.
The completed target comparison evaluated the released checkpoint unchanged,
then fine-tuned those released weights on development data and evaluated the
frozen development-selected checkpoint once on the held-out split.

## Split Control

Read `data_split_metadata.csv` before selecting any plot and preserve its split
label in metadata and metric tables.

- Use development plots for adapter checks, parameter selection and error
  analysis.
- Keep evaluation plots isolated from training and parameter selection.
- Fix semantic filters, point correspondence, matching policy and IoU threshold
  before evaluating the test set.
- Record the model checkpoint, external commit and container route.

## Completed SegmentAnyTree Sequence

1. Reproduce the upstream seed-42 split of development plots into 16 training
   and 5 validation plots.
2. Convert development LAS files to the upstream Treeins PLY schema and verify
   that no test file enters the training root.
3. Extract and hash-check the released checkpoint.
4. Evaluate that unchanged checkpoint on all 11 held-out test plots using
   aligned point-wise outputs.
5. Start a separate model from the released weights with fresh optimizer and
   epoch history, then fine-tune on all 16 training plots.
6. Evaluate the fine-tuned checkpoint on the five fixed development validation
   plots and freeze it without using test results for selection.
7. Run the selected fine-tuned checkpoint once on all 11 held-out test plots.
8. Report the released matching policy and a strict one-to-one policy
   separately.
9. Rebuild the workbook and public-safe tables only after all gates pass.

All nine stages completed. The published-pretrained target obtained mean plot
F1 `0.453409` and micro F1 `0.444245`; the development-fine-tuned target
obtained mean plot F1 `0.544679` and micro F1 `0.531987`. The public per-plot,
site and overall tables reconcile to 11 test plots and 323 references for each
variant. The historical from-scratch aggregate and provenance manifest remain
under [`methods/segmentanytree/examples/`](../../methods/segmentanytree/examples/)
and must not be used as either target result.

Every completed accuracy row records reference and prediction counts, TP, FP,
FN, precision, recall, F1, coverage, matched IoU, matching policy, runtime,
peak memory, checkpoint identity, thresholds and semantic masks.

## Required Validation Gates

1. The released checkpoint checksum must match the pinned value.
2. Both variants must emit aligned semantic and instance arrays.
3. Fine-tuning must load released weights only, not historical optimizer state.
4. Fine-tuned checkpoint selection must use the five development validation
   plots only.
5. Each target must retain all 11 test predictions and aligned metric files.

All five gates passed for both target variants. The shared off-Git retention
manifest and its SHA-256 are recorded in the public provenance and retention
registry.

The historical run has provenance gaps for the exact workflow commit,
container digest and final per-plot table transfer. Its retained aggregate is
valid historical evidence, but it does not close any current target gate.

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
