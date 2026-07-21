# Repository Governance Audit

Audit date: 21 July 2026.

Scope: public documentation, result registries, evaluation/reporting contracts,
method provenance and release controls. The audit preserved every accepted
score and all completed, diagnostic, historical and rejected evidence. No
held-out inference was run, no new held-out prediction was inspected and no
method was tuned.

## Findings And Actions

| Issue | Severity | Evidence | Action taken | Files changed | Unresolved risk | Recommended follow-up |
| --- | --- | --- | --- | --- | --- | --- |
| Public status described six results while seven completed held-out rows existed; another section already said seven. | High | Root README status/table versus canonical method-result CSV and workbook. | Dated status 21 July 2026 and separated two primary harmonised rows, three shared-protocol baselines and two differently scoped TLS2trees rows without changing values. | `README.md`, `BENCHMARKS.md`, `outputs/README.md` | Readers may retain older copies/screenshots. | Use canonical CSV-derived release notes. |
| TLS2trees rows used a different class-3-ignore scoring mask but were described as headline rows beside shared-mask results. | High | TLS2trees configs/provenance and canonical result table protocol/mask columns. | Kept both rows as accepted evidence; development-tuned is primary and published-default is baseline within the TLS2trees protocol, both excluded from the shared ranking with an explicit reason. | `README.md`, `BENCHMARKS.md`, `docs/protocols/result-governance.md`, TLS2trees validity audit | Cross-protocol numeric comparison remains possible if downstream users drop governance columns. | Require ranking eligibility and mask fields in exports/plots. |
| Free-text status mixed execution completion with scientific result role. | High | Values such as `completed_aligned_pointwise_test` in existing result/provenance files. | Added controlled `result_status`, `completion_state`, `ranking_eligible` and `exclusion_reason`; preserved older terms in original evidence/human descriptions, with optional legacy overlay fields. | `docs/protocols/result-governance.md`; canonical governance registry | Frozen method provenance still contains legacy field names. | Add overlays; do not edit hash-pinned evidence. |
| Historical run IDs mix `for-instance`/`for_instance`, abbreviations and informal suffixes. | Medium | Registry IDs including `sat`, `quicktune`, `long` and `full`. | Preserved historical IDs and defined a strict six-component future format plus canonical aliases. | `docs/protocols/result-governance.md`, `docs/protocols/for-instance.md` | External tooling may continue creating legacy-shaped IDs. | Validate every newly created run ID at submission. |
| `external_training_only` conflated learning regime with development exposure. | High | TreeX and TLS2trees rows share that legacy mode despite development parameterisation. | Added separate controlled learning-regime and dataset-exposure fields and documented the current mapping. | `docs/protocols/result-governance.md`, `docs/protocols/for-instance.md`, canonical governance registry | Historical configs retain the ambiguous legacy field for compatibility. | Make the two new fields mandatory for future runs. |
| Held-out execution, metric viewing, qualitative viewing and subsequent changes were not centrally distinguished. | High | Evidence was distributed across method runbooks/provenance. | Added a public-safe exposure ledger and protocol definitions; unsupported details remain `unknown`/`not_recorded`. | `outputs/for_instance_benchmark_metrics/test_exposure_ledger.csv`, `docs/protocols/result-governance.md` | Older visualisation/view dates are not recoverable from current public evidence. | Append evidence-backed events; never infer an event from a score. |
| Method tuning effort was not comparable from epoch counts alone. | Medium | SegmentAnyTree and TreeLearn both report 35 epochs but different examples/steps; TreeX has no optimiser; TLS2trees uses parameter screens. | Added a structured method-development budget and stated that accuracy comparability does not imply equal optimisation effort. | `outputs/for_instance_benchmark_metrics/method_development_budget.csv`, governance/protocol docs | Actual GPU/CPU hours are incomplete. | Import recorded accounting only; do not convert resource requests to hours. |
| “Leaf-on”, “leaf-off” and “leaf attachment” could be misread as acquisition season or a common scoring domain. | High | TLS2trees target definitions use different material classes and class-3 exclusion. | Defined prediction material, explicit reference classes/mask and attachment fields; retained acquisition metadata separately. | Governance protocol, FOR-instance protocol, evaluation metrics, TLS2trees validity audit | Older filenames necessarily retain concise target terms. | Require material/mask columns whenever those files are joined. |
| Extra IoU thresholds, robust plot summaries and uncertainty were absent from the primary reporting contract. | Medium | Existing canonical tables primarily report IoU 0.5 point estimates. | Added diagnostic-only thresholds, plot bootstrap support and availability metadata without changing canonical scores. | Evaluation modules/tests, `docs/evaluation_metrics.md`, governance protocol, diagnostic availability table | Most retained predictions are off Git and may not be locally accessible; only 11 test plots limit interval interpretation. | Run deterministic diagnostics from retained artefacts only after an explicit Barkla-safe plan. |
| Semantic and grouping errors could not always be separated. | Medium | Some methods lack independent semantic labels or intermediate clusters in public artefacts. | Defined semantic omission/commission and split/merge diagnostics; unsupported fields are explicit. | Evaluation modules/tests, `docs/evaluation_metrics.md`, diagnostic availability table | Method-specific intermediate evidence remains incomplete. | Add development-only stage counters before attributing low F1. |
| Lightweight unit fixtures did not provide one complete LAS adapter-to-aggregate contract. | Medium | Existing tests exercised components, but not one generated fixture with all FOR-instance dimensions through the full lightweight path. | Added a synthetic LAS fixture with XYZ, `classification`, `treeID` and `source_index`, then exercised normalisation, alignment, evaluation and aggregation without private data/GPU/Barkla. | `tests/test_generated_las_integration.py`, adapter/evaluator modules | A synthetic fixture cannot reproduce every upstream method runtime. | Keep it as the stable integration contract and retain method-specific smokes separately. |
| TreeLearn documentation did not consolidate all validity questions. | High | Coordinate alignment was documented, but units/scaling, vertical transforms, upstream thresholds/features and block-edge behaviour were distributed or missing. | Added an evidence-linked checklist with closed, partial and open items. | `methods/treelearn/docs/for_instance_validity_audit.md` | Fully resolved upstream runtime configuration is not public for every checkpoint. | Export the resolved config and stage diagnostics on development data. |
| TLS2trees final counts did not expose semantic/stem/cluster attrition. | High | Public tables give final predictions and matches but not intermediate stage counts. | Added an evidence-linked checklist and a qualified validity conclusion; no result was changed because it was low. | `methods/tls2trees/docs/for_instance_validity_audit.md` | Semantic, stem, clustering and filtering contributions cannot yet be isolated. | Publish development-only per-stage counters. |
| Updating stale terminology inside frozen TLS2trees configs would break provenance hashes. | Critical | Published provenance records hashes for the published, workflow and benchmark configs. | Left all frozen configs untouched and documented their exact hashes; governance is layered alongside them. | TLS2trees validity audit only | Future contributors may edit them accidentally. | Add/retain hash checks and use overlay metadata. |
| Workbook maintenance was described as manual although CSV data are canonical. | High | Workbook-sync tests read an existing workbook; prior method notes asked for manual rebuild/visual checks. | Declared CSV as source of truth and workbook as a generated review artefact; added a deterministic builder/check route. | `outputs/README.md`, governance/release docs, reporting script/tests | Visual layout still needs human review. | Run builder and workbook-sync tests before every release. |
| Environment/upstream provenance fields varied by method. | Medium | Method configs and JSON provenance record different subsets. | Added a machine-readable environment registry and a public-safe template using explicit unknown/not-applicable values. | `outputs/for_instance_benchmark_metrics/method_environment_provenance.csv`, `docs/templates/method_run_provenance.yml` | Historical CUDA/framework/container details are incomplete. | Backfill only from retained environment manifests. |
| Release evidence lacked one central readiness gate. | High | Documentation index had runbooks but no release checklist. | Added a checklist for protocol, tables, commit, environments, upstream/checkpoint identity, retention, exposure, limitations, workbook and tag planning. | `docs/release_readiness.md`, `docs/README.md` | A checklist cannot replace verification at a specific commit. | Complete and sign off the checklist for each release candidate. |
| The labelled-accuracy planning document read as current governance. | Medium | It still described TLS2trees as only a future compatibility experiment. | Marked it historical and redirected current status to canonical governance/output documents. | `docs/plans/labelled-accuracy.md` | Historical prose intentionally retains the decision context of its snapshot. | Do not use planning documents as current result registries. |
| Operational paths and scheduler examples can look machine-specific. | Low | Public runbooks use generic `$HOME`, `~/scratch`, `~/fastscratch` and upstream container paths. | Retained generic reproducibility examples; checked public text for personal `/Users/...` paths and credentials. | Audit/release documentation | Commands were not re-executed on Barkla in this branch. | Validate operational commands during the next authorised Barkla maintenance window. |
| Repository-relative Markdown links could drift as new governance files were added. | Low | Read-only link scan across Markdown after the documentation edits. | Confirmed every local Markdown target resolves; no link correction was required beyond links introduced by this change. | Documentation files in this change | The filesystem check does not validate external URLs or the semantic correctness of anchors. | Repeat the link scan before release and verify external sources when network access is authorised. |
| Multiple public tables and method summaries can drift. | Medium | README, registry, method evidence, canonical CSV and workbook repeat result identities. | Assigned canonical roles and documented deterministic governance/workbook generation and consistency tests. | `outputs/README.md`, governance/release docs, reporting scripts/tests | Narrative prose still needs review when new rows are added. | Generate tables first, then update prose from observed output. |

## Result-Integrity Conclusion

No demonstrable implementation or aggregation error was identified in the
accepted point estimates during this governance review. No accepted canonical
score was changed or replaced. The reporting generator re-derives existing
counts, mean plot F1 and micro metrics from committed per-plot CSVs only as a
fail-fast consistency check; it also computes separate plot-distribution and
bootstrap diagnostics. Those derived checks do not become replacement primary
scores. The governance correction is presentational and machine-readable: five
rows share one ranking protocol, while two complete TLS2trees rows use another
scoring mask and remain outside that ranking.

The new controlled roles make earlier prose explicit: the coordinate-rematched
SegmentAnyTree export-audit failure is rejected, the `0.4825` aligned
SegmentAnyTree row is historical, the TreeX reference-labelled-mask result is
diagnostic, the TLS2trees development-tuned leaf-on row is primary only within
its own protocol, and the published-default leaf-on row is that protocol's
baseline. These labels do not change a score or provenance identity.

Historical, diagnostic, rejected, operational and candidate records remain
available. Additional thresholds, uncertainty intervals and error
decomposition are explicitly diagnostic and cannot silently replace the IoU
`>= 0.5` canonical estimates.

## Deliberately Deferred

- No held-out inference, tuning or new qualitative prediction inspection.
- No attempt to fill missing runtime, environment or exposure facts by
  inference.
- No modification of hash-pinned TLS2trees configuration files.
- No new segmentation method.
- No Git tag, GitHub release or merge.
- No diagnostic values fabricated when retained Barkla predictions are absent
  locally.

Use [`release_readiness.md`](release_readiness.md) for the commit-specific
release gate and [`protocols/result-governance.md`](protocols/result-governance.md)
for future result additions.
