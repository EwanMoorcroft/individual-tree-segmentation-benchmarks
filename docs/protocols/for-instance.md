# FOR-instance Cross-Method Benchmark Protocol

## Purpose And Version

This protocol defines how segmentation methods are trained, run and evaluated
on FOR-instance so results remain reproducible and comparable. It separates
reproduction of a method's published result from the harmonised comparison
used across methods.

Protocol identifier: `for_instance_pointwise_v1`.

Result roles, ranking eligibility, test-exposure records, development budgets
and future run identities are controlled by the companion
[`result-governance protocol`](result-governance.md). Governance metadata does
not alter this evaluation protocol or any frozen score.

Primary sources:

- FOR-instance dataset:
  <https://doi.org/10.5281/zenodo.8287792>
- FOR-instance data paper:
  <https://arxiv.org/abs/2309.01279>
- SegmentAnyTree paper:
  <https://doi.org/10.1016/j.rse.2024.114367>

## Two Evaluation Layers

Every method should produce two clearly labelled result sets where its
published implementation differs from the shared benchmark.

1. **Published-method reproduction.** Use the authors' checkpoint, input
   preparation, semantic mask, matching implementation and designated test
   split as closely as possible. This result checks that the installation and
   inference path reproduce the paper.
2. **Harmonised cross-method evaluation.** Convert predictions to one
   point-aligned instance label per source point and evaluate every method with
   the same one-to-one matching implementation.

The published-method result must not silently replace the harmonised result,
or vice versa. Differences between them are part of the reproducibility
record.

## Dataset Lock

- Use the 32 LAS files from Zenodo record 8287792.
- Store the archive checksum and an individual checksum for every LAS file in
  ignored run metadata.
- Preserve each `relative_path`, collection and supplied split from
  `data_split_metadata.csv`.
- The metadata catalogue contains 56 development and 26 test paths. The
  downloaded 32-LAS benchmark is the exact-path subset of 21 development and
  11 test plots; every method uses this same local subset.
- Do not edit the source LAS files.
- Record the observed point count, dimensions, semantic values and positive
  `treeID` count before running a method.
- Treat CULS, NIBIO, RMIT, SCION and TUWIEN as separate acquisition domains
  when reporting results.

## Training And Test Control

Every run must declare one of these training modes:

- `published_pretrained`: use an authors' checkpoint without updating its
  weights;
- `retrained_from_dev`: train from initial weights using only FOR-instance
  development data;
- `external_training_only`: fit no model weights on FOR-instance, whether the
  method was trained externally or has no fitted weights;
- `fine_tuned_on_dev`: start from a pretrained checkpoint and update weights
  using only FOR-instance development data.

For deterministic or rule-based methods without fitted weights, record
`external_training_only` to indicate that no FOR-instance development or test
data were used for model fitting, and record the method-specific mode
separately, for example `unsupervised_parameterised`.

New runs must also separate:

- `learning_regime`: `supervised`, `self_supervised`, `unsupervised`,
  `deterministic` or `rule_based`; and
- `dataset_exposure`: `published_checkpoint`, `external_only`,
  `development_tuned`, `development_trained` or `none`.

These fields do not replace the legacy training mode. They prevent
parameter-tuned unsupervised methods from being described as untouched
external baselines merely because no model weights were updated. The current
method mapping is maintained in
[`result-governance.md`](result-governance.md#learning-regime-and-dataset-exposure).

For every checkpoint, record:

- file name and SHA-256 checksum;
- source URL or generation command;
- upstream repository commit;
- stated training datasets;
- whether FOR-instance development data were included;
- random seed and training configuration where training is performed locally.

FOR-instance test files must not be used for training, early stopping,
threshold selection, visual parameter adjustment or repeated debugging.
Development results may be reported for diagnostics, but the primary
comparable accuracy result is calculated on the supplied test split.

Every known held-out exposure must be appended to the public-safe
[`test-exposure ledger`](../../outputs/for_instance_benchmark_metrics/test_exposure_ledger.csv).
The record distinguishes a test job executing, metrics being viewed,
predictions being visualised and any configuration change afterwards. Earlier
exploratory exposure remains visible; it is not erased by a later freeze.
Unknown dates, viewing events or decisions remain `unknown`; use
`not_applicable` only when an event field genuinely cannot apply. Values must
not be inferred.

The discarded SegmentAnyTree coordinate-rematching run included test outputs
before this protocol was frozen. Those values must not guide the corrected
training configuration. From this point, training failures and model selection
are handled only with development training and validation data. The held-out
test job is submitted only after the checkpoint and evaluation settings have
been frozen.

Where a method needs validation data, create it from the supplied development
split and record the selection algorithm and seed. SegmentAnyTree follows its
upstream conversion script: seed 42 and a fixed random 25% of development
plots for validation. The same FOR-instance development/test boundary and
harmonised evaluator apply to every later method. Method-specific
augmentations, architectures and optimisation schedules remain part of each
method's documented reproduction rather than being forced to be identical.

For the current SegmentAnyTree and TreeLearn development fine-tunes, the
declared schedule is 35 epochs. Each method must also record examples per
epoch, batch size, total examples and optimizer steps because an epoch does not
represent equal work across architectures. TreeX is deterministic and has no
optimizer or epoch count. Cross-method comparability is defined by the frozen
development/test boundary, validation-only selection, one-time 11-plot test,
point-aligned prediction contract and identical evaluator—not by hiding
method-specific training exposure.

The public shared-protocol ranking must contain only `held_out_test_primary`
comparison rows. Each ranked row must use the supplied 11-plot test split, all
323 reference instances and this shared protocol with maximum-cardinality
one-to-one matching. Within that group, development-selected target rows use
`result_status=primary` and declared comparison settings use
`result_status=baseline`. A baseline may be development-parameterised only
when that exposure is explicit. The current ranked group contains two primary
rows and three baselines. The two complete class-3-ignore TLS2trees rows remain
accepted held-out evidence in a separate protocol group; their shared-ranking
eligibility is false because the reference scoring mask differs. Development,
smoke, checkpoint-sweep, alternate-material and overlap-affected results must
be stored as diagnostics and must never occupy ranked rows. Missing method
variants are reported as gaps rather than filled with a differently scoped
result.

Method-development effort is recorded separately in
[`method_development_budget.csv`](../../outputs/for_instance_benchmark_metrics/method_development_budget.csv).
Where supported, record configurations, validation evaluations, checkpoints,
epochs, optimiser steps, GPU/CPU hours, manual inspection and hyperparameter
source. Do not infer compute hours from Slurm resource requests. Identical
accuracy evaluation does not imply equal optimisation effort.

## Reference Definition

The reference instance field is `treeID`.

- Include points whose `classification` is `4`, `5` or `6`.
- Ignore classes `0`, `1`, `2` and `3`.
- Ignore `treeID` values `0` and `-1`.
- Do not remove difficult trees after viewing predictions.
- Preserve the reference labels as one point-aligned array.

The tree-material definition matches the binary tree class used by
SegmentAnyTree: stems, live branches and woody branches are tree points;
terrain, low vegetation, out-points and unclassified points are non-tree.

Result metadata must distinguish prediction material, reference scoring mask
and leaf attachment. Shared rows use `prediction_material=full_tree_material`,
`reference_scoring_mask=classes_4_5_6` and
`leaf_attachment=not_applicable`. A woody-only target, a woody-plus-leaf
target and a distinct attachment stage must be named separately. These fields
describe processing/scoring material and must not overwrite genuine
acquisition-season metadata. Explicit included and ignored semantic classes
remain required when a short mask slug is insufficient.

## Prediction Contract

The preferred prediction artefact contains exactly one predicted semantic
label and one predicted instance label for every source point.

- Preserve source point order or a stable source-point identifier.
- Do not recover correspondence through rounded XYZ coordinates when stable
  point indices are available.
- A final exported point cloud must pass point-count and coordinate-multiset
  validation before being used for accuracy evaluation.
- If an exporter duplicates or removes rows, evaluate the aligned internal
  prediction arrays instead.
- Retain raw method outputs separately from harmonised prediction adapters.
- Retain every full prediction artefact on Barkla under the method's
  `data/predictions/<method>/for_instance...` root, or an explicitly recorded
  run-specific prediction root, so later metrics and adapter checks can reuse
  the same outputs.
- Record the Barkla-relative prediction directory and any harmonised prediction
  artefact paths in run metadata. Local copies under `local_outputs/` are
  backups only and must not be the only retained prediction source.
- A completed run must inventory retained prediction files with byte sizes and
  SHA-256 values where the method workflow supports hashing. The final gate
  must fail when an expected prediction artefact is missing or has changed.
- Add every accepted, rejected or diagnostic result used in reporting to
  `outputs/for_instance_benchmark_metrics/for_instance_prediction_retention_registry.csv`.
  Retries use new run-specific roots and must not delete earlier evidence.
- Record every ignored or unassigned prediction label.

Methods that naturally output one file per tree must map those predictions
back to source-point identifiers. Coordinate-tolerance matching may be used
only when the method provides no stable point correspondence, and must be
reported as a separate evaluation mode.

## Point-Wise IoU

For predicted instance \(P\) and reference instance \(R\):

```text
IoU(P, R) = points in both P and R / points in either P or R
```

SegmentAnyTree's published implementation accepts a match at IoU greater than
or equal to 0.5. Although the paper text uses `> 0.5`, the released evaluator
uses `>= 0.5`; the reproduction layer follows the released implementation and
records the operator explicitly.

The evaluation mask is the union of reference-tree and predicted-tree points,
matching the released SegmentAnyTree evaluator. A prediction on a reference
background point therefore contributes to the predicted instance and can
reduce its IoU.

## Matching And Metrics

The published SegmentAnyTree reproduction reports its released
per-prediction-best-IoU policy. The harmonised benchmark uses a strict
one-to-one assignment at IoU `>= 0.5`.

For the harmonised result:

```text
TP = accepted one-to-one matches
FP = predicted instances without an accepted match
FN = reference instances without an accepted match
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1 = 2 * precision * recall / (precision + recall)
```

Also report:

- mean and median IoU among accepted matches;
- mean unweighted coverage and mean point-weighted coverage;
- reference, prediction, TP, FP and FN counts;
- per-plot runtime and peak memory;
- semantic tree/non-tree accuracy where the method predicts semantics;
- all accepted and unmatched instance identifiers.

Diagnostic evaluation may additionally report precision, recall and F1 at IoU
thresholds `0.25`, `0.50` and `0.75`; median plot F1 and interquartile range;
zero-F1 plot count; per-site macro summaries; over-/under-segmentation
indicators; and unmatched counts. These values are never used for selection
and never replace the IoU `>= 0.5` point estimate.

Where method outputs contain independent semantic labels, error decomposition
may report semantic omission/commission, instance splitting/merging,
unassigned reference-tree points, semantic-commission points,
positive-instance points on reference background and unmatched instances. A
semantic label derived only from a positive instance ID cannot isolate
semantic-model errors from grouping errors. Unsupported fields must remain
explicit.

Bootstrap confidence intervals resample plots rather than individual trees,
use default seed `20260721` and `10000` configurable iterations, and are
emitted separately from canonical point estimates. Optional site-stratified
resampling must be labelled carefully: the test subset has only 11 plots and
several sites have one plot, so the interval is not a broad population
estimate.
Availability for each accepted row is recorded in
[`diagnostic_metric_availability.csv`](../../outputs/for_instance_benchmark_metrics/diagnostic_metric_availability.csv).

Aggregate counts before calculating micro precision, recall and F1. Report
per-plot values, collection summaries, development/test summaries and the
test-only primary result. Do not average collection F1 values to produce the
overall micro F1.

## Reproduction Gates

TreeLearn and later method adapters must complete the
[`method-adapter acceptance checklist`](method-adapter-acceptance-checklist.md)
before full FOR-instance evaluation is submitted.

A method result is not accepted as comparable until all gates pass:

1. dataset checksums and split labels are recorded;
2. source and output schemas are recorded;
3. repository commit, container digest and checkpoint checksum are recorded;
4. no test plot was used for tuning;
5. prediction rows remain aligned with source/reference rows;
6. raw and harmonised prediction artefacts are retained on Barkla under the
   recorded prediction roots;
7. all expected test plots have a result or a documented failure;
8. the evaluator and threshold operator are recorded;
9. one development plot is manually checked for label alignment;
10. results are compared with the paper by collection; and
11. an absolute F1 difference greater than 0.10 from a directly comparable
    published result triggers investigation before the result is described as
    a successful reproduction;
12. result role, completion state, ranking eligibility and any exclusion reason
    are recorded;
13. learning regime, dataset exposure and material/mask fields are recorded;
    and
14. method-specific upstream, environment and checkpoint provenance is
    recorded or explicitly unknown.

The comparison gate is a diagnostic, not a target to optimise against.
Published scores must never be used to tune test predictions.

## SegmentAnyTree Reproduction Route

The current Barkla inference uses the published container interface and
`PointGroup-PAPER.pt`. It runs `eval.py`; it does not train a new model.

For paper-aligned evaluation:

1. inventory the internal PLY outputs with
   `inspect_segmentanytree_internal_outputs.sbatch`;
2. identify the aligned semantic and instance evaluation PLY files;
3. run `pointwise_instance_metrics.py` on those aligned files;
4. record both `paper_compatible` and `harmonized` metrics;
5. run `audit_segmentanytree_for_instance_export_array.sbatch` separately to
   determine whether the final LAZ is row-preserving; and
6. do not use the final LAZ for accuracy when the export audit fails.

The earlier coordinate-rematched result set is retained as rejected workflow
evidence after its export audit failed. Its historically provisional values
are not an accepted SegmentAnyTree reproduction.

The current SegmentAnyTree comparison evaluates the released checkpoint as
`published_pretrained`, then starts a separate `fine_tuned_on_dev` run from
those released weights. FOR-instance development plots supply training and
internal validation data, and the supplied test split remains absent from the
training data root. The earlier `retrained_from_dev` run remains a completed
historical result; it is not a current target and its metrics must not be
attributed to either comparison variant.

## Required Public Documentation

For each dataset-method combination, publish:

- a configuration without private paths or credentials;
- source and method citations;
- exact training mode and split policy;
- pinned repository commit and checkpoint checksum;
- input adapter and prediction adapter;
- evaluator version and metric definition;
- synthetic tests;
- public-safe per-plot and aggregate tables after validation;
- a prediction-retention registry row that identifies the reusable off-Git
  prediction set;
- failures, deviations and known limitations.

Canonical benchmark data are CSV tables under
`outputs/for_instance_benchmark_metrics/`. The workbook is a generated review
artefact built deterministically with
`python scripts/reporting/build_for_instance_workbook.py`; it must not become
an independently edited source of truth. Derived site, plot-distribution and
bootstrap-CI diagnostic summaries are built with
`python scripts/reporting/build_for_instance_governance_outputs.py`; that
command does not author the evidence-led governance ledgers.

Future run IDs use
`<method>__<dataset>__<training-mode>__<selection-mode>__<split>__<YYYYMMDDTHHMMSS>`.
Historical run IDs and hash-linked paths remain unchanged; use a canonical
alias instead of renaming evidence.

Raw datasets, checkpoints, predictions, full logs and machine-specific
metadata remain outside Git.
