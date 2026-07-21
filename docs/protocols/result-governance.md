# Benchmark Result Governance

Status date: 21 July 2026.

This protocol controls how benchmark evidence is classified, displayed and
extended. It supplements the fixed
[`FOR-instance protocol`](for-instance.md); it does not change accepted scores,
the IoU `>= 0.5` threshold, maximum-cardinality one-to-one matching or any
frozen run artefact.

## Controlled Result Taxonomy

[`benchmark_result_registry.csv`](../../outputs/for_instance_benchmark_metrics/benchmark_result_registry.csv)
is the canonical governance registry. Every row must retain a useful
human-readable `status_description` and expose these machine-readable fields:

| Field | Allowed values | Meaning |
| --- | --- | --- |
| `result_status` | `primary`, `baseline`, `diagnostic`, `historical`, `rejected`, `operational_only`, `candidate`, `failed` | Scientific/reporting role of the result |
| `completion_state` | `completed`, `partial`, `pending` | Execution completeness, independent of reporting role |
| `ranking_eligible` | `true`, `false` | Whether the row may be ranked inside its named comparison group |
| `exclusion_reason` | lower-case underscore/hyphen slug; empty only when eligible | Why a row is excluded from ranking |

`result_status` is not an execution message. Existing values such as
`completed_aligned_pointwise_test` remain in their original metric/evidence
tables or human descriptions; they must not be treated as the new scientific
taxonomy. Governance overlays may retain optional `legacy_result_role` and
`legacy_result_status` fields, but the canonical registry does not duplicate
them or rewrite frozen evidence.

Use these roles consistently:

- `primary`: the preregistered or development-selected result for a method and
  protocol;
- `baseline`: a published/default or otherwise declared comparison setting;
  any development parameterisation must be disclosed through
  `dataset_exposure`;
- `diagnostic`: smoke, development, alternate-threshold, alternate-material or
  error-analysis evidence that cannot enter the primary ranking;
- `historical`: a completed result retained from an earlier workflow but not a
  current target;
- `rejected`: a configuration stopped by a validation or governance decision;
- `operational_only`: execution, resource or output-validity evidence without
  individual-tree accuracy labels;
- `candidate`: work not yet executed or accepted; and
- `failed`: an attempted run that did not produce an accepted result.

A completed diagnostic is `result_status=diagnostic` and
`completion_state=completed`; completion never promotes it to a primary row.
A rejected run can likewise be complete as evidence. Missing facts use
`unknown` or `not_recorded` where the table permits them; they are never
inferred from a score.

## Current FOR-instance Reporting Groups

All seven accepted held-out rows remain canonical evidence, but they occupy
three visibly separate reporting groups:

| Reporting group | Method variants | Taxonomy | Ranking |
| --- | --- | --- | --- |
| Primary harmonised results | SegmentAnyTree development fine-tuned; TreeLearn development fine-tuned | `primary`, `completed` | Eligible inside `for_instance_pointwise_v1` |
| Shared-protocol baselines | SegmentAnyTree published checkpoint; TreeX unsupervised parameterised; TreeLearn published checkpoint | `baseline`, `completed` | Eligible inside `for_instance_pointwise_v1` |
| TLS2trees held-out protocol | TLS2trees development tuned; TLS2trees published default | tuned row `primary`, default row `baseline`; both `completed` | Not eligible for the shared-protocol ranking; `exclusion_reason=different_reference_scoring_mask` |

The TLS2trees development-tuned result is the primary result **within the
TLS2trees class-3-ignore protocol**. Its `ranking_eligible=false` value records
the different scoring mask; it does not demote, invalidate or reinterpret the
result. The published-default TLS2trees row is its within-protocol baseline.
Neither may appear in a numerical ranking with the five shared-protocol rows.

Development smokes, checkpoint sweeps, alternate-material targets, additional
IoU thresholds, bootstrap intervals and error decomposition are diagnostics.
The TreeX reference-labelled-mask result is likewise a completed diagnostic,
not a second baseline, because it changes the scoring domain.
The earlier SegmentAnyTree aligned run is historical. Validation regressions
and configurations stopped before accepted test evaluation are rejected. The
coordinate-rematched export-audit failure is also rejected from accepted
accuracy reporting while its evidence remains preserved. FRDR without
instance labels is operational only. Unrun methods and Wytham routes are
candidates.

## Learning Regime And Dataset Exposure

Training history and FOR-instance exposure are separate concepts. New rows
must add:

| Field | Allowed values |
| --- | --- |
| `learning_regime` | `supervised`, `self_supervised`, `unsupervised`, `deterministic`, `rule_based`, `unknown` |
| `dataset_exposure` | `published_checkpoint`, `external_only`, `development_tuned`, `development_trained`, `none`, `unknown` |

The legacy `training_mode` remains for compatibility. It does not, by itself,
prove that a parameterised method was untouched by FOR-instance development
data. The current mapping is:

| Variant | Learning regime | Dataset exposure | Legacy training mode |
| --- | --- | --- | --- |
| SegmentAnyTree published checkpoint | `supervised` | `published_checkpoint` | `published_pretrained` |
| SegmentAnyTree development fine-tuned | `supervised` | `development_trained` | `fine_tuned_on_dev` |
| TreeLearn clean published checkpoint | `supervised` | `published_checkpoint` | `published_pretrained` |
| TreeLearn development fine-tuned | `supervised` | `development_trained` | `fine_tuned_on_dev` |
| TreeX unsupervised parameterised | `unsupervised` | `development_tuned` | `external_training_only` |
| TLS2trees published default | `unsupervised` | `published_checkpoint` | `external_training_only` |
| TLS2trees development tuned | `unsupervised` | `development_tuned` | `external_training_only` |

For TLS2trees, `learning_regime=unsupervised` describes the instance pipeline's
benchmark role; the bundled learned semantic component and its checkpoint must
still be disclosed in environment provenance. `external_training_only` is
retained for older consumers and must not be paraphrased as “no development
exposure” for TreeX or development-tuned TLS2trees.

## Prediction Material And Scoring Domain

New result rows must declare:

| Field | Allowed values | Scope |
| --- | --- | --- |
| `prediction_material` | `woody_only`, `woody_plus_leaf`, `full_tree_material`, `unknown` | Material represented by predictions |
| `reference_scoring_mask` | `classes_4_5_6`, `class_3_ignored`, `custom`, `unknown` | Reference/scoring domain |
| `leaf_attachment` | `enabled`, `disabled`, `not_applicable`, `unknown` | Whether a distinct attachment stage was used |

Also retain the explicit reference classes and ignored classes when the short
mask slug cannot fully describe the domain. Shared pointwise rows use full tree
material, reference classes `4,5,6`, and the established union mask. TLS2trees
leaf-on rows use woody-plus-leaf predictions, reference classes `4,5,6`, class
3 excluded before the union mask, and leaf attachment enabled. TLS2trees
leaf-off diagnostics use woody-only predictions, reference classes `4,6`,
class 3 excluded, and leaf attachment disabled.

“Leaf-on” and “leaf-off” in those TLS2trees target names describe prediction
and scoring material. They do not assert the acquisition season. “Leaf
attachment” describes a processing stage and is not interchangeable with
either acquisition season or the reference mask.

## Future Run Identifiers

Historical IDs are immutable evidence. Do not rename files, paths, provenance
records or hashed configurations to make them conform. A registry may add a
canonical alias while retaining the original `run_id`.

Future runs use:

```text
<method>__<dataset>__<training-mode>__<selection-mode>__<split>__<YYYYMMDDTHHMMSS>
```

Semantic components use lower-case ASCII letters, digits and hyphens; the
timestamp is UTC and contains no punctuation. Example:

```text
segmentanytree__for-instance__fine-tuned-dev__best-validation__test__20260711T002931
```

Do not introduce informal identity suffixes such as `long`, `full`,
`quicktune`, `final`, `best` or `new` outside the structured selection field.
The validator in `src/benchmark/governance.py` applies the strict rule only to
future IDs and keeps a permissive historical-ID path.

## Held-out Test Exposure

[`test_exposure_ledger.csv`](../../outputs/for_instance_benchmark_metrics/test_exposure_ledger.csv)
is the public-safe record of known held-out exposure. Each row distinguishes:

- `test_job_executed`: inference or evaluation touched a test plot;
- `metrics_viewed`: aggregate or plot metrics were inspected;
- `predictions_visualised`: qualitative predictions were inspected;
- `configuration_changed_afterwards`: method or post-processing configuration
  changed after that exposure; and
- `decision_or_change`: the documented consequence, including “none”.

These events are not interchangeable. A job can execute without a person
viewing metrics; aggregate metrics can be viewed without qualitative
visualisation; and exposure does not prove that a configuration changed.
Unknown dates or decisions remain explicitly `unknown`; use `not_applicable`
only when the field genuinely cannot apply.

Earlier exploratory exposure is preserved transparently. It is not erased by
a later protocol freeze. A later primary run is eligible only when its
checkpoint, parameters, scoring domain and adapter were frozen from
development evidence before its authorised test job, and the test result was
not then used to select a replacement setting.

## Development Budgets

[`method_development_budget.csv`](../../outputs/for_instance_benchmark_metrics/method_development_budget.csv)
records supported counts for configurations attempted, validation evaluations,
checkpoints evaluated, epochs, optimiser steps, GPU/CPU hours, manual
inspection and the hyperparameter source. Resource requests are not converted
into compute hours. Unknown wall time remains `unknown`.

Accuracy comparability means the split, alignment contract, scoring domain and
evaluator are comparable. It does not mean that methods received equal
optimisation effort. Budgets are therefore disclosed beside, not folded into,
accuracy scores.

## Diagnostic Metrics And Uncertainty

The canonical point estimate remains IoU `>= 0.5`. Additional IoU thresholds
`0.25`, `0.50` and `0.75`, median plot F1, plot-F1 interquartile range, zero-F1
plot count, site macro summaries, matched-IoU summaries, unmatched counts and
split/merge indicators are diagnostics only. They must not select a
checkpoint, parameter set or replacement primary result.

Plot-level bootstrap confidence intervals resample plots, never individual
trees. The implemented defaults are seed `20260721`, `10000` iterations and a
`0.95` confidence level; all remain recorded/configurable. Optional
site-stratified resampling is supported. Interval outputs remain separate from
canonical point estimates. With only 11 test plots and several sites
represented by one plot, intervals—especially site-stratified intervals—
describe uncertainty in this fixed benchmark subset and are not strong
population claims.

Availability is evidence-dependent:

- point-aligned retained predictions support deterministic threshold
  diagnostics when they are accessible;
- plot tables support plot bootstrap summaries but cannot recreate every
  matched-pair or semantic error field;
- semantic omission/commission requires method semantic predictions;
- splitting/merging requires aligned instance arrays or the full
  predicted-by-reference intersection matrix; accepted one-to-one match rows
  alone discard extra overlaps; and
- operational-only datasets without reference instances support none of the
  instance-accuracy diagnostics.

Use `unsupported`, `prediction_artifact_unavailable` or `not_recorded` rather
than fabricating values. Off-Git Barkla retention does not mean the artefacts
are available in a local checkout. No held-out inference or prediction
inspection is authorised merely to fill a diagnostic column.

The shared implementations are `src/benchmark/diagnostic_metrics.py` and
`src/benchmark/result_statistics.py`; per-result support is declared in
[`diagnostic_metric_availability.csv`](../../outputs/for_instance_benchmark_metrics/diagnostic_metric_availability.csv).
Committed plot-table diagnostics are published separately as the
[`plot-distribution table`](../../outputs/for_instance_benchmark_metrics/for_instance_plot_distribution_diagnostics.csv),
[`site table`](../../outputs/for_instance_benchmark_metrics/for_instance_method_site_results.csv)
and [`bootstrap-CI table`](../../outputs/for_instance_benchmark_metrics/for_instance_bootstrap_confidence_intervals.csv).

## Canonical Tables And Workbook

CSV tables under `outputs/for_instance_benchmark_metrics/` are canonical.
The Excel workbook is a deterministic generated review artefact: it must be
rebuilt from those tables, never edited as an independent source of truth.
Automated checks must reconcile workbook aggregates and row identities with
the CSV sources without requiring Microsoft Excel or LibreOffice. A generated
workbook still requires a public-safety and layout review before release.

Derived diagnostic-summary and workbook regeneration commands:

```bash
python scripts/reporting/build_for_instance_governance_outputs.py
python scripts/reporting/build_for_instance_workbook.py
```

The first command builds derived diagnostic summaries; it does not generate or
rewrite evidence-led exposure, budget or environment records.

## Environment Provenance

[`method_environment_provenance.csv`](../../outputs/for_instance_benchmark_metrics/method_environment_provenance.csv)
is the canonical public-safe environment table. Each implemented method/run
should provide the fields in
[`method_run_provenance.yml`](../templates/method_run_provenance.yml), including
the upstream repository and commit, dirty state, container identity/digest,
Python/CUDA/framework versions, checkpoint source/hash and stated checkpoint
training datasets. Use `not_applicable` for genuinely absent components and
`unknown`/`not_recorded` for missing evidence. Do not imply that method-specific
containers or environments are the shared lightweight utility environment.

## Change Control

Any governance-only change must preserve accepted numeric values and evidence
paths. Promotion, demotion, renaming or score replacement requires an explicit
audit finding, rationale and diff. If a deterministic recomputation is later
authorised from retained predictions, publish it as diagnostic evidence first
and document exactly why it does or does not affect the canonical score.
