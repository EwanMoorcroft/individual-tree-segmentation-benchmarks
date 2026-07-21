# TreeLearn FOR-instance Validity Audit

Audit date: 21 July 2026.

This audit reviews the frozen TreeLearn FOR-instance evidence. It does not
change either accepted held-out score and does not authorise another test run.
The two accepted rows remain results for their exact checkpoints, pinned
upstream commit, generated pipeline configuration, source-row adapter and
shared evaluator.

## Audit Checklist

| Validity question | Status | Repository evidence | Remaining limitation |
| --- | --- | --- | --- |
| Coordinate units and scaling | Partial | The adapter compares source and original-resolution prediction XYZ row by row and uses a 0.005 m tolerance. The accepted smoke observed a maximum delta of `0.00027683842927217484` m. | Public run evidence does not independently record the LAS CRS/unit declaration or enumerate every upstream coordinate scaling operation. The metre interpretation and any upstream scale convention should be captured from the frozen runtime configuration. |
| Vertical normalisation | Open | The benchmark stages the source LAS unchanged for inference and retains original-resolution output. | The public configuration does not state whether, where or with what constants the pinned upstream modular pipeline normalises height. Add a runtime-exported upstream configuration record before interpreting height-specific failures. |
| Block/tile generation and reconstruction | Partial | The generated pipeline configuration sets `tile_generation: true`, retains upstream full-forest LAZ/NPZ, requires the full-forest point count to match the source, and evaluates only the aligned full-resolution result. | Intermediate tile membership, overlap and reconstruction decisions are not summarised publicly, so block-edge losses cannot yet be measured separately. |
| Minimum instance-size filtering | Partial | The shared evaluator applies `min_predicted_instance_points=0` and `min_predicted_tree_fraction=0.0`; no benchmark-side prediction-size filtering or test-time selection is permitted. | Any upstream grouping/minimum-size rule inherited through `configs/_modular/grouping.yaml` is not enumerated in public provenance for the clean checkpoint. |
| Semantic thresholds | Open | The adapter maps positive TreeLearn instance IDs to predicted semantic class `4` and labels `0`/`-1` to background for the shared evaluator. | This derived semantic field is not an independent record of the network's semantic probabilities or threshold decisions. Upstream semantic thresholds are not listed in the public result evidence. |
| Block-boundary duplicates | Partial | The final accepted artefact has exactly one `pred_tree_id` and one `source_row_index` per source row, preventing duplicate rows in the scored array. | The public summaries do not count duplicate or competing upstream instances before full-forest reconstruction, so they cannot quantify boundary-induced split/merge errors. |
| Expected input features | Partial | The benchmark requires LAS XYZ plus `classification` and `treeID`; reference fields are retained for evaluation, and the staged inference input is the unchanged LAS. The pinned pipeline loads upstream modular sample-generation, model, grouping and test-dataset configs. | The complete model feature vector, feature units and derived-feature normalisation are not copied into the public run record. `classification` and `treeID` are reference evidence; the audit has not established that they are model features. |
| Source-row alignment | Closed | The accepted smoke verified equal point counts, source order and coordinate tolerance. Held-out final gates retained raw and aligned artefacts for every plot; the aligned NPZ includes `source_row_index == arange(point_count)` and is the only primary evaluation input. | Alignment proves correspondence, not correctness of upstream semantic or grouping decisions. |
| Post-processing configuration | Partial | The generated config records the pinned modular config names plus benchmark overrides for spatial shape, tiling, full-forest return, shape settings and HDBSCAN use. The held-out routes forbid post-test changes. | The pinned upstream repository does not bundle a complete historical post-processing configuration specifically for `model_weights_finetuned.pth`; the result is for the exact recorded benchmark pipeline, not every plausible TreeLearn setting. |
| Checkpoint training-data overlap | Closed for classification | The December 2024 default checkpoint is documented as descending from training that included FOR-instance validation/test data and remains development-only/overlap-affected. The clean `model_weights_finetuned.pth` checkpoint is documented as externally pretrained then L1W-fine-tuned, with stated training data excluding FOR-instance. The local fine-tuned checkpoint used only the frozen 16-plot development-training subset. | The audit relies on the authors' stated checkpoint training datasets; an exact upstream plot manifest is not bundled for the December 2024 checkpoint. |

## Frozen Result Interpretation

The development-fine-tuned row remains a primary harmonised result: all 11
test plots, 323 references, source-row alignment, the union mask, IoU `>= 0.5`
and maximum-cardinality one-to-one matching are recorded. Its mean plot F1 is
`0.364685` and micro F1 is `0.331924`.

The unchanged clean published-checkpoint row remains a shared-protocol
baseline with mean plot F1 `0.078944` and micro F1 `0.098694`. The independent
retention audit reproduced its TP/FP/FN counts. The authors describe that
checkpoint as targeting trees above 10 m, and the audit observed no matched
references below 10 m. That is a limitation of interpreting the frozen result,
not evidence that the evaluator or row-alignment adapter failed.

One plot in the published-checkpoint run required the documented empty-group
execution recovery. It mapped unresolved labels to background and skipped
empty optional visualisations without changing weights, thresholds, grouping
settings or evaluator behaviour. It is retained as execution provenance, not
as tuning.

The evidence therefore supports “valid result for the exact frozen TreeLearn
pipeline”. It does not support a claim that the score is TreeLearn's best
possible performance, that every historical upstream post-processing value is
known, or that low F1 has been isolated to one internal stage.

## Error Attribution Limits

The scored `pred_classification` is derived from positive instance IDs. It
cannot by itself distinguish a network semantic omission from a grouping-stage
omission. Final one-label-per-row arrays support unmatched-instance and
overlap-based split/merge diagnostics, but intermediate semantic
probabilities, initial clusters and block ownership are not in the committed
public summaries. Report unavailable decomposition fields as `unsupported` or
`not_recorded`.

## Required Follow-up

Any future TreeLearn investigation must use development data only until a new
protocol and test authorisation exist. Before proposing another held-out row:

- export the fully resolved pinned upstream modular configuration;
- record CRS/coordinate units and all coordinate/height transforms;
- enumerate input and derived model features;
- record upstream semantic and instance-filter thresholds;
- inventory tile overlap, reconstruction and pre-reconstruction duplicate
  candidates; and
- publish development-only semantic/grouping error decomposition.

These follow-ups must not modify or replace the two frozen accepted rows. The
source runbooks and result evidence are:

- [`one_plot_smoke.md`](one_plot_smoke.md);
- [`long_finetune_protocol_20260712.md`](long_finetune_protocol_20260712.md);
- [`finetuned_test_results_20260713.md`](finetuned_test_results_20260713.md);
- [`pretrained_test_results_20260714.md`](pretrained_test_results_20260714.md);
- [`for_instance_one_plot_smoke.yml`](../configs/for_instance_one_plot_smoke.yml);
  and
- [`run_for_instance_one_plot_smoke.py`](../scripts/run_for_instance_one_plot_smoke.py).
