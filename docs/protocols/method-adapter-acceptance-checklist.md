# Method Adapter Acceptance Checklist

This checklist applies to TreeLearn and every later FOR-instance method adapter
before full development or held-out test evaluation is submitted. A one-plot
development smoke run may be used to validate the adapter, but full evaluation
must wait until each required gate below has evidence.

Protocol identifier: `for_instance_method_adapter_acceptance_v1`.

Result roles and future run IDs also follow
[`result-governance.md`](result-governance.md).

## Acceptance Record

Create one acceptance record per method and run profile before full evaluation.
The record may be a method configuration, run metadata JSON, or implementation
note, but it must include:

- method slug, dataset slug and run ID;
- for future runs, a canonical ID in the form
  `<method>__<dataset>__<training-mode>__<selection-mode>__<split>__<YYYYMMDDTHHMMSS>`;
- upstream repository URL, citation and pinned commit or package version;
- training mode: `published_pretrained`, `fine_tuned_on_dev`,
  `retrained_from_dev` or `external_training_only`;
- controlled result status, completion state, ranking eligibility and any
  exclusion reason;
- learning regime and dataset exposure, recorded separately from the legacy
  training mode;
- prediction material, reference scoring mask, explicit reference classes and
  leaf-attachment state;
- checkpoint filename and SHA-256 checksum, or a clear statement that the
  method has no fitted checkpoint;
- upstream dirty state, container identity/digest, Python/CUDA/framework
  versions, checkpoint source and stated checkpoint training datasets, using
  [`the provenance template`](../templates/method_run_provenance.yml);
- method-specific environment, command entrypoints and Slurm resources;
- prediction root, metadata root and table root;
- evaluation mode, point-correspondence mode and IoU threshold operator.

Historical run IDs remain unchanged. If a historical ID is non-canonical, add
a canonical alias in governance metadata rather than renaming files or hashed
evidence.

## Required Gates

- [ ] **Repository scaffold.** The method has a README, config, runtime entrypoint
  and Slurm entrypoint under `methods/<method>/`. Public documentation uses
  repository-relative paths and does not include private paths, raw logs,
  checkpoints, raw predictions or raw datasets.
- [ ] **Dataset lock.** The adapter uses the supplied FOR-instance split metadata
  and preserves `relative_path`, collection and split labels. The source LAS
  files are not edited.
- [ ] **Test split control.** No held-out test plot is used for training, early
  stopping, threshold selection, visual tuning, checkpoint selection or repeated
  adapter debugging. Development validation and held-out test results remain
  separate.
- [ ] **Exposure record.** Any known held-out execution, metric viewing,
  prediction visualisation and subsequent configuration change is recorded as
  a separate evidence-backed field in the public-safe test-exposure ledger.
- [ ] **Reference contract.** The adapter records `treeID` as the reference
  instance field, `classification` as the semantic field, tree classes `4`, `5`
  and `6`, ignored semantic classes `0`, `1`, `2` and `3`, and ignored instance
  labels `0` and `-1`, unless a separately named protocol variant is justified.
- [ ] **Input conversion audit.** Any method-specific input conversion records
  retained points, dropped points, semantic remapping, coordinate preservation,
  source row identifiers and positive reference-tree counts.
- [ ] **Prediction retention on Barkla.** Every full prediction artefact is
  retained under `data/predictions/<method>/for_instance...` on Barkla, or under
  an explicitly recorded run-specific prediction root. Local copies under
  `local_outputs/` are backups only. The run gate inventories expected files,
  byte sizes and SHA-256 values where supported; retries use new roots rather
  than deleting earlier evidence. The public retention registry records whether
  future metrics can be calculated without inference.
- [ ] **Prediction metadata.** Run metadata records the Barkla-relative prediction
  directory, harmonised prediction artefact paths, command, return code, runtime,
  peak memory, method version, checkpoint checksum and status for every plot.
- [ ] **Environment provenance.** The exact upstream commit/dirty state,
  method-specific container or environment, available dependency versions and
  checkpoint training-data declaration are recorded. Missing evidence is
  explicit and is not copied from the shared utility environment.
- [ ] **Point correspondence.** The preferred adapter output has one predicted
  instance label and one reference `treeID` for each evaluated source point,
  using source row order or a stable source-point identifier. Coordinate-based
  matching is allowed only when no stable correspondence exists, and then it is
  labelled as a fallback mode.
- [ ] **Export validation.** If evaluation uses an exported point cloud, the
  export passes row-count and coordinate-multiset validation. If the export
  duplicates, drops or reorders points, aligned internal arrays or a harmonised
  row-preserving artefact are used instead.
- [ ] **Ignored predictions.** Background, invalid, empty and unassigned
  prediction labels are recorded and handled consistently before metrics are
  calculated.
- [ ] **Shared evaluation route.** The primary comparable result uses the shared
  harmonised point-wise evaluator, or a method wrapper that has synthetic tests
  proving equivalent strict one-to-one IoU matching.
- [ ] **Metric definition.** Primary comparable metrics use strict one-to-one
  matching at IoU `>= 0.5`, then calculate precision, recall and F1 from TP, FP
  and FN. Semantic tree/non-tree accuracy is not reported as instance F1.
- [ ] **Aggregation rule.** Split-level and collection-level summaries aggregate
  TP, FP and FN first, then calculate micro precision, recall and F1. Overall
  micro F1 is not calculated by averaging plot or collection F1 values.
- [ ] **Required result tables.** Full evaluation writes per-plot, per-collection,
  split-level, matched-pair, unmatched-prediction and unmatched-reference tables,
  plus run metadata for the same run ID.
- [ ] **Diagnostic availability.** The result records whether additional IoU
  thresholds, plot bootstrap intervals, semantic error decomposition and
  split/merge diagnostics are supported by its retained artefacts. Unsupported
  values remain explicit and do not block the canonical point estimate.
- [ ] **Development budget.** Configurations, validation evaluations,
  checkpoints, epochs, optimiser steps, manual inspection, hyperparameter
  source and evidence-backed compute hours are recorded without guessing.
- [ ] **Documented failures.** Every expected full-evaluation plot has either a
  completed result or a documented failure with status, error summary and output
  paths.
- [ ] **Manual alignment check.** At least one development plot is manually
  checked for label alignment before held-out test evaluation is submitted.
- [ ] **Public-safe summary.** Any committed examples contain only public-safe
  summaries. They exclude coordinates, raw point clouds, full prediction arrays,
  checkpoints, private paths and credentials.

## Synthetic Test Requirements

Each adapter must add synthetic tests for:

- row count preservation;
- coordinate preservation where coordinates are part of the adapter contract;
- source row identifier or stable point identifier preservation;
- reference field detection;
- prediction field detection;
- semantic class mapping and ignored class handling;
- ignored instance labels;
- empty predictions;
- all-background predictions;
- duplicate predicted instance IDs;
- missing required fields;
- mismatched prediction and reference lengths;
- exported point cloud validation failure, when exported clouds are supported;
- coordinate-fallback labelling, when coordinate matching is supported;
- result-table schema and metadata fields.

At least one lightweight integration test must use a generated LAS/LAZ fixture
with synthetic XYZ, `classification`, `treeID` and `source_index` fields. It
must exercise dataset adaptation, prediction normalisation, source-row
alignment, evaluation and aggregate output without Barkla, a GPU or private
data.

Synthetic fixtures must stay small and must not include raw benchmark LAS, LAZ,
PLY, NPZ or checkpoint files.

## Smoke Gate Before Full Evaluation

Before submitting a full FOR-instance array, run a development-only smoke gate:

- one small development plot completes conversion, inference and adapter
  normalisation;
- the full prediction is present under the recorded Barkla prediction root;
- metadata includes runtime, memory, command, status and prediction paths;
- the adapter output passes row-count and field checks;
- synthetic adapter tests pass;
- the evaluation command writes matched-pair and unmatched-instance tables;
- no held-out test plot is read by the workflow.

Full held-out test evaluation may start only after the smoke gate and required
gates are complete.

## Blocking Conditions

Do not submit full FOR-instance evaluation when any of the following are true:

- the adapter only produces local prediction copies and no retained Barkla
  prediction source;
- point correspondence is inferred from rounded coordinates despite stable row
  identifiers being available;
- development and held-out test outputs share a result table or metadata record;
- F1 is computed without individual-tree reference labels;
- coordinate-based and aligned point-wise results are combined;
- full evaluation would overwrite existing predictions without an explicit
  run-specific root or documented replacement decision;
- public documentation or committed examples include raw data, raw predictions,
  checkpoints, credentials, private paths or full logs.
