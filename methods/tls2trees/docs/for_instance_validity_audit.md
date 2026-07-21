# TLS2trees FOR-instance Validity Audit

Audit date: 21 July 2026.

This audit reviews the completed TLS2trees development-tuned and
published-default held-out workflows. It preserves their values and their
separate `for_instance_pointwise_class3_ignore` scoring domain. It does not
authorise another held-out run or any post-test parameter selection.

## Audit Checklist

| Validity question | Status | Repository evidence | Remaining limitation |
| --- | --- | --- | --- |
| Stem-detection counts | Open | Frozen instance parameters and final emitted tree counts are recorded. The development compatibility probe documents the observed `no_graph_connected_stem_bases` failure mode that motivated development-only search. | Public held-out evidence does not provide per-plot detected stem/base/cylinder counts before filtering. |
| Semantic and tree-point counts | Open | Label-stripped geometry is passed through the bundled FSCT semantic model; source point totals and final prediction counts are retained. Reference semantic labels are not supplied as model predictions. | Public held-out summaries do not record per-stage terrain/leaf/CWD/wood point counts or the number of source rows eligible for the instance stage. |
| Intermediate cluster counts | Open | The workflow retains run-scoped raw outputs off Git and records the instance configuration. | Public evidence does not tabulate slice clusters, accepted stem bases, graph components, rejected clusters or points dropped at each stage. |
| Final tree-output counts | Closed | The development-tuned leaf-on row has 38 predictions, 323 references and 3 matches; its leaf-off diagnostic has 22 predictions and no matches. The published-default leaf-on row has 6 predictions and no matches; its leaf-off diagnostic has 3 predictions and no matches. | Counts describe emitted/scored instances, not where earlier candidates were lost. |
| Leaf attachment | Closed for behaviour; partial for attribution | `instance.py` writes leaf-off trees and, with attachment enabled, separate leaf-on trees. Targets use separate aligned roots and summaries. A nine-setting, five-development-plot leaf screen completed 45/45 metrics and every setting had the same aggregate accuracy. | Intermediate leaf-candidate/graph/assigned-point counts are not public, so leaf-stage contribution cannot be decomposed plot by plot. The screen is diagnostic and cannot select a new test configuration. |
| Class-mask differences | Closed | Leaf-on references use classes `4,5,6`; leaf-off references use `4,6`. Class 3 is excluded before the union mask in both TLS2trees targets. Predictions on other included background rows remain false-positive material and are not silently removed. | This mask differs from the shared pointwise union domain; both TLS2trees held-out rows are therefore excluded from that ranking. |
| Source-row adapter | Closed | The converter retains a representative source row for each voxel, raw tree points are uniquely mapped to representatives, labels are projected back to source rows, and the primary artefact contains one label per source row. All 22 aligned target files per variant passed retention verification. | Alignment proves that the scored arrays correspond to source rows; it does not prove that semantic, stem or grouping predictions are accurate. |
| Zero/near-zero F1 interpretation | Partial | Both frozen pipelines completed their input, semantic, instance, alignment, retention and evaluation gates. The accepted counts are therefore not explained by a missing-reference evaluator or a failed final alignment step. | Current public stage summaries cannot isolate terrain normalisation, semantic transfer, stem detection, graph clustering, filtering and leaf attachment. The evidence supports a poor result for the exact UAV-to-terrestrial frozen pipeline, not a universal claim about TLS2trees or a single confirmed failing stage. |

## Material And Scoring Contract

The target names describe processing/scoring material, not acquisition season:

| Target | Prediction material | Reference material | Scoring mask | Leaf attachment |
| --- | --- | --- | --- | --- |
| leaf-on | `woody_plus_leaf` | classes `4,5,6`, positive `treeID` | class 3 excluded, then union of reference-target and predicted-target points | `enabled` |
| leaf-off | `woody_only` | classes `4,6`, positive `treeID` | class 3 excluded, then union of reference-target and predicted-target points | `disabled` |

The development-tuned leaf-on result is the primary result within this
TLS2trees protocol. It has mean plot F1 `0.015023` and micro F1 `0.016620`.
The published-default leaf-on baseline has mean plot and micro F1 both
`0.000000`. Both are complete, accepted evidence, but both use
`ranking_eligible=false` and
`exclusion_reason=different_reference_scoring_mask` for the shared-protocol
leaderboard. The two leaf-off rows remain target-specific diagnostics.

## Frozen Configuration Boundary

The published-default provenance pins these committed files by SHA-256:

| File | Frozen SHA-256 |
| --- | --- |
| `methods/tls2trees/configs/for_instance_published_default.yml` | `0b27478dcf31cc804755567661089c370d4f1ce2152bf1a0cda325627f37b3c9` |
| `methods/tls2trees/configs/for_instance_published_default_test.yml` | `5518ca3c26d469ddcbf43ed86b08d4fd3ba88425e5d2e715acd6031d8ea0ea03` |
| `methods/tls2trees/configs/for_instance_benchmark.yml` | `c2571212995831e2290addc87b1395d1f8e96cb953eb81b3bb5070dbafe40a67` |

Do not edit those frozen files to modernise terminology or governance fields.
Add governance records alongside them. Changing any byte would break the
published provenance hash even if the scientific parameters were unchanged.

## Failure Attribution Boundary

The completed gates rule out several narrow explanations:

- the evaluator had all 323 reference instances;
- the primary artefacts were source-row aligned rather than coordinate-fallback
  metrics;
- both targets completed all 11 test plots;
- prediction files passed size/hash retention checks; and
- the published-default configuration was neither selected from FOR-instance
  metrics nor changed after test execution.

They do **not** show how many potential trees were lost at semantic, stem,
slice-cluster, graph, filter or leaf stages. The small final prediction counts
make upstream stage attrition a central unresolved question. Until stage
counters are exported, describe the result as weak transfer of the exact
frozen pipeline from terrestrial-method assumptions to FOR-instance UAV laser
scanning. Do not label it simply “algorithm failure” or “adapter failure”.

## Required Follow-up

Any follow-up must begin on development data and must not replace the frozen
test rows. Add public-safe stage summaries that record, per plot and target:

- input/downsampled point counts and semantic counts by FSCT class;
- slice and stem-candidate counts before and after each acceptance filter;
- intermediate graph/cluster counts and rejection reasons;
- emitted raw tree-file counts, cross-tile ownership merges and aligned output
  counts;
- leaf candidates, graph components and assigned/unassigned leaf points; and
- point losses or conflicts at every source-row projection gate.

Those counters would permit semantic omission/commission and grouping failure
analysis without inspecting new held-out predictions. New parameter work, if
authorised later, must remain development-only until separately frozen.

Current evidence:

- [`for_instance_benchmark.md`](for_instance_benchmark.md);
- [`for_instance_benchmark.yml`](../configs/for_instance_benchmark.yml);
- [`for_instance_published_default.yml`](../configs/for_instance_published_default.yml);
- [`tls2trees_development_tuned_test_provenance.json`](../examples/tls2trees_development_tuned_test_provenance.json);
- [`tls2trees_published_default_test_provenance.json`](../examples/tls2trees_published_default_test_provenance.json); and
- [`tls2trees_development_leaf_screen_provenance.json`](../examples/tls2trees_development_leaf_screen_provenance.json).
