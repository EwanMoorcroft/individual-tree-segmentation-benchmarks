# TLS2trees FOR-instance Benchmark Design

## Status

This document defines the four target FOR-instance TLS2trees rows:

| Variant | Target |
| --- | --- |
| `published_default` | leaf-off |
| `development_tuned` | leaf-off |
| `published_default` | leaf-on |
| `development_tuned` | leaf-on |

TLS2trees is not trained on FOR-instance. All four rows use
`training_mode: external_training_only`; the separate variant field records
whether method parameters come from the publication or are selected on the
development split.

The exact manifest, published-default conversion, semantic and instance
runners, source-row adapter, target-aware evaluator and Barkla chains are
implemented and validated. Development selection was frozen before the
held-out test. The final evaluator excludes class-3 out-points from the scoring
domain and uses the retained source-row predictions directly. Both leaf-on
rows are complete; leaf-off is retained as a target-specific diagnostic for
each variant. The published-default configuration was evaluated without
FOR-instance metric selection and did not change after the test. The legacy
one-plot pilot is not one of the four target rows.

## Authoritative Sources

- FOR-instance data and split metadata:
  <https://doi.org/10.5281/zenodo.8287792>
- FOR-instance protocol:
  [`docs/protocols/for-instance.md`](../../../docs/protocols/for-instance.md)
- TLS2trees paper: Wilkes et al. (2023),
  <https://doi.org/10.1111/2041-210X.14233>
- TLS2trees publication archive:
  <https://doi.org/10.5281/zenodo.8406668>
- TLS2trees upstream repository:
  <https://github.com/tls-tools-ucl/TLS2trees>

The publication release is tag `tls2trees` at commit
`216100ed2dade15d1bd6f09c287787e55085102a`. The executable pin is current
official commit `ca12cb73b2c736d80b020e8025f8d975d42e6f01`, which includes
post-release global-shift, pandas and tile-index fixes. Both commits are
recorded so the publication identity and executable implementation cannot be
confused.

## Repository Audit

The repository already provides:

- the dataset-supplied development/test boundary and exact local subset;
- a cross-method maximum-cardinality one-to-one matcher;
- point-aligned SegmentAnyTree and TreeLearn evaluation patterns;
- a coordinate fallback evaluator for per-tree point-cloud outputs;
- a TLS2trees instance-stage runner, output summariser and Barkla pilot; and
- a completed FRDR operational benchmark with runtime and memory evidence.

The existing FOR-instance pilot is scientifically unsuitable for the four
target rows because it:

- uses one CULS development plot only;
- supplies reference semantic classes directly to `instance.py` instead of
  running the bundled semantic model;
- removes reference background before inference;
- does not implement the paper's 10 m tiling and 0.02 m downsampling;
- uses local-minimum height rather than a documented terrain model;
- reuses FRDR feasibility thresholds rather than published values;
- writes no stable source-row correspondence;
- cannot explicitly select leaf-on files during evaluation; and
- has no split, variant, target, run-ID, freeze or full-summary isolation.

It remains useful as a legacy instance-stage diagnostic and must retain that
label.

## Official Split

The source of truth is the dataset's `data_split_metadata.csv`. Its SHA-256 in
the inspected dataset is
`dd64aa338681f8f4166f8d175879a2b0b0158ecf222497ec6f7f0b23bc4fce94`.
The catalogue contains 56 development and 26 test paths. The downloaded
32-LAS benchmark is the exact existing-path subset:

| Split | Plots | Points | Reference trees | Site counts |
| --- | ---: | ---: | ---: | --- |
| Development | 21 | 101,769,037 | 807 | CULS 2; NIBIO 14; RMIT 1; SCION 3; TUWIEN 1 |
| Held-out test | 11 | 49,709,922 | 323 | CULS 1; NIBIO 6; RMIT 1; SCION 2; TUWIEN 1 |

The exact ordered paths are recorded in
[`for_instance_benchmark.yml`](../configs/for_instance_benchmark.yml). A future
manifest builder must intersect the dataset metadata with files by exact path,
record file and metadata hashes, and fail on any different count, path, split
or order. It must not infer aliases or rebalance sites.

The development split is available for compatibility work, parameter search,
error analysis and final selection. The held-out test split is unavailable to
all parameter generation and selection code.

## Reference Targets

The dataset README defines `classification` as:

- `0`: unclassified;
- `1`: low vegetation;
- `2`: terrain;
- `3`: out-points;
- `4`: stem;
- `5`: live branches or green crown; and
- `6`: woody branches.

Both targets use `treeID > 0`. Values `0`, `-1` and any other non-positive
value are background, not reference instances. No tree is removed after
viewing predictions.

### Leaf-off

- Reference material: classes `4` and `6`.
- Background: classes `0`, `1`, `2`, `3` and `5`.
- Prediction artifact: `*.leafoff.ply` only.
- Predicted points on class-5 or other background rows remain predicted
  positives and can reduce IoU; they are not silently removed.

### Leaf-on

- Reference material: classes `4`, `5` and `6`.
- Background: classes `0`, `1`, `2` and `3`.
- Prediction artifact: `*.leafon.ply` only.
- A reference tree participates if it has a positive `treeID` and at least one
  point in the target classes. No subjective per-tree foliage filter is used.

Leaf-on and leaf-off results use separate roots, manifests and summaries.

## Official Pipeline And I/O

The online method has three stages:

1. registered input is tiled and downsampled;
2. `semantic.py` applies the bundled FSCT model; and
3. `instance.py` segments wood and optionally attaches leaf points.

The published route uses 10 m tiles and 0.02 m voxel-centre nearest-neighbour
downsampling. `semantic.py` writes a segmented PLY containing at least
`x`, `y`, `z`, `n_z` and `label`, with method classes `0` terrain, `1` leaf,
`2` coarse woody debris and `3` wood. The complete benchmark must run this
stage on label-stripped geometry. FOR-instance `classification` values are
reference evidence only and are not passed to the method as predictions.
Upstream buffering requires nine indexed tiles. The adapter fails before
inference when a plot has fewer tiles; it does not reduce this neighbour count
or otherwise change the published/default algorithm.

`instance.py` always writes one `*.leafoff.ply` per predicted tree. With
`--add-leaves`, it additionally writes the corresponding `*.leafon.ply`; it
does not replace the leaf-off output. One published/default run can therefore
produce both target artifacts, which are evaluated separately.

## Published/default Reconstruction

Appendix A, Table A.1 of the paper is the primary parameter source. Upstream
CLI defaults are recorded separately because several differ materially.

| Parameter | Paper value | Upstream CLI default | Proposed benchmark value |
| --- | ---: | ---: | ---: |
| Tile edge | 10 m | external preprocessing | 10 m |
| Downsample voxel | 0.02 m | external preprocessing | 0.02 m |
| Semantic buffer | 5 m | 0 m | 5 m |
| Semantic model | bundled FSCT model | bundled FSCT model | bundled FSCT model |
| `n_tiles` | 3, 5 or 7 | 3 | 5 |
| Slice thickness | 0.5 m | 0.2 m | 0.5 m |
| Stem boundary | 2.0-2.5 m | 1.5-2.0 m | 2.0-2.5 m |
| Minimum stem radius | 0.025 m | 0.025 m | 0.025 m |
| Minimum stem points | 200 | 200 | 200 |
| Graph edge length | 2 m | 1 m | 2 m |
| Maximum cumulative gap | 3 m | infinite | 3 m |
| Minimum tree points | 200 | 0 | 200 |
| Add leaves | true | false | true |
| Leaf voxel length | 0.5 m | 0.5 m | 0.5 m |
| Leaf edge length | 1 m | 1 m | 1 m |

The paper does not supply one universal `n_tiles` value. It reports 3, 5 and 7
for different sites, while the official README example uses 5. The proposed
reconstruction therefore uses 5 and records the uncertainty. The paper prose
also mentions 0.2 m slicing, but its values-used appendix and official example
both use 0.5 m; the proposed configuration uses 0.5 m.

The machine-readable record, including implicit semantic and instance
constants, model hash and compatibility modifications, is
[`for_instance_published_default.yml`](../configs/for_instance_published_default.yml).

## Compatibility Changes Versus Tuning

The following are compatibility or reproducibility changes and do not select
FOR-instance parameters:

- conversion from LAS geometry to the published tiled PLY schema;
- removal of reference-only fields before semantic inference;
- exact field and tile-index mapping;
- the official post-release global-shift and pandas fixes;
- a reversible local coordinate shift followed by exact source-row projection;
- stable source-row tracking or a lossless coordinate sidecar;
- proposed Python, NumPy and PyTorch seeds of 42, recorded as a reproducibility
  patch that can change the stochastic realization and still requires
  implementation validation;
- explicit target-file selection;
- run-specific output paths, hashes and provenance; and
- replacing the hard-coded leaf graph distance with the parsed argument while
  retaining its published value of 1 m.

FOR-instance coordinates are commonly hundreds of thousands to millions of
metres. Upstream `instance.py` casts XYZ to float32. Processing unshifted
coordinates would lose centimetre-scale distinctions, so local processing and
the source-row projection gate are required for valid evaluation. Raw method
PLYs remain in the recorded local frame and are never sent to the coordinate
fallback evaluator. The gate must prove point-count and source-row identity
before accuracy metrics are accepted.

Changes to voxel scale, semantic threshold, spatial context, slicing, stem
detection, graph thresholds, minimum tree size or leaf attachment geometry are
parameter tuning and are forbidden in the published/default rows.

## Parameter Audit And Search Scope

| Group | Parameter | Published value | Candidate values | Effect and expected runtime | Decision |
| --- | --- | ---: | --- | --- | --- |
| Preprocessing | Tile edge | 10 m | fixed | Scheduling/context; more tiles add overhead | Fixed |
| Preprocessing | Downsample voxel | 0.02 m | 0.02, 0.04, 0.08 | Density, small structure and memory | Conditional |
| Semantic | Model weights | bundled | fixed | Defines semantic predictions | Fixed |
| Semantic | Buffer | 5 m | fixed initially | Edge context and memory | Fixed initially |
| Semantic | `is_wood` threshold | 1.0 | 0.5, 0.75, 1.0 | Wood recall/precision; semantic rerun | Conditional |
| Context | `n_tiles` | 5 reconstruction | 3, 5, 7 | Physical context and memory | Conditional |
| Skeleton | Slice thickness | 0.5 m | 0.25, 0.5, 1.0 | Vertical connectivity; thinner is slower | Search |
| Stem finding | Boundary | 2.0-2.5 m | 1.0-1.5, 1.5-2.0, 2.0-2.5 | Stem visibility and base count | Search |
| Stem finding | Minimum radius | 0.025 m | 0.015, 0.025, 0.05 | Small-stem sensitivity | Search |
| Stem finding | Minimum points | 200 | 50, 100, 200 | Stem sensitivity and false bases | Search |
| Wood graph | Edge length | 2 m | 1, 2, 3 | Connectivity and graph cost | Search |
| Wood graph | Cumulative gap | 3 m | 2, 3, 5 | Crown retention and merging | Search |
| Filtering | Minimum tree points | 200 | 50, 100, 200 | Small-tree retention | Search |
| Leaf graph | Voxel length | 0.5 m | 0.25, 0.5, 1.0 | Crown detail and graph size | Leaf-on search |
| Leaf graph | Edge length | 1 m | 0.5, 1, 2 | Crown connectivity | Leaf-on search |

Hidden DBSCAN, RANSAC, cylinder and graph-neighbour constants remain fixed.
They are algorithmic constants but are not exposed by the published interface;
changing them would define a more extensively modified method.

## Staged Development Search

The parameter groups interact strongly, so a full grid and a long one-factor
sequence would both waste compute. The executed design used a bounded
compatibility probe, a five-site screen and one all-development refinement
round. Ordered candidates and configuration hashes were recorded before each
stage; a seed alone was not treated as a frozen search.

1. **Stage 0 - compatibility and sensitivity gate.** Select one development
   plot per site by a predeclared median-point-count rule, before examining
   accuracy. Validate semantic output, local-shift round trip, coordinate
   multiplicity, unique assignment, both target patterns and resource capture.
2. **Stage 1 - broad screen.** Run at most 12 configurations including the
   published baseline across that five-site subset. Save valid, invalid and
   failed configurations. Conditional parameters enter only when their Stage 0
   gate justifies the additional semantic or context cost.
3. **Stage 2 - narrow refinement.** Evaluate at most three candidates per
   target across all 21 development plots, with one refinement round only.
4. **Stage 3 - freeze.** Select one leaf-off and one leaf-on configuration,
   write immutable configuration and selection manifests, record hashes and
   the Git commit, and obtain review before any tuned held-out run.

Selection considers mean plot F1, micro F1, every site, precision-recall
balance, cross-site variation, over- and under-segmentation, invalid plots,
runtime and distance from the published/default configuration. Ties prefer:

1. stronger cross-site stability;
2. fewer failed or invalid plots;
3. fewer changes from published/default;
4. lower runtime; and
5. only then a small metric advantage.

The machine-readable search definition is
[`for_instance_search_space.yml`](../configs/for_instance_search_space.yml).

## Evaluation Integrity

The shared target contract is IoU `>= 0.5` with maximum-cardinality one-to-one
matching. TP, FP and FN are summed before micro precision, recall and F1 are
calculated. Per-plot macro, site and split summaries remain distinct.

TLS2trees outputs separate tree files and upstream discards arbitrary source
fields. The legacy coordinate fallback is not sufficient because it
deduplicates quantised XYZ, does not prevent the same source point appearing in
multiple predicted trees, cannot represent an empty prediction set, and
silently prefers leaf-off files. The implemented adapter therefore:

- selects exactly one target pattern;
- records the raw local frame and projects labels to exact source rows;
- records LAS scale and the 0.001 m raw-to-representative smoke tolerance;
- counts duplicate coordinates within and across predictions;
- rejects conflicting many-to-one source assignments;
- quantifies unmatched predicted and reference coordinates;
- quantifies reference coordinate retention after downsampling;
- defines empty prediction and empty reference behaviour;
- retains matched, unmatched, over-segmentation and under-segmentation evidence;
- records runtime and peak memory; and
- has synthetic equivalence coverage against the shared one-to-one matcher.

The converter retains the source row nearest each 0.02 m voxel centre and a
map from every source row to that representative. Raw TLS2trees coordinates
are matched uniquely to representatives, then instance labels are projected
to all source rows in each voxel. The primary scored artefact is consequently
row-aligned `source_row_predictions.npz`; coordinate recovery is confined to
the raw-output adapter and is never the primary metric route.

Successful process exit is never sufficient evidence of a valid row.

## Barkla Compute Pattern

The executed dependency pattern was:

1. dataset inventory and split manifest on CPU;
2. tiling, coordinate shift and input conversion on CPU;
3. bundled semantic-model inference on one GPU per plot, justified by upstream
   CUDA support and subject to an environment smoke test;
4. instance and leaf attachment as CPU/high-memory arrays, reusing cached
   semantic outputs across parameter configurations where valid;
5. target-specific evaluation on CPU;
6. summary, runtime calibration and freeze gates on CPU; and
7. failed-task resubmission only into new run-specific roots.

The completed FRDR workflow provides only low-confidence scaling evidence: it
processed 205.6 million points in 19,099 cumulative process seconds, while one
plot exceeded 32 GiB and peaked near 49.6 GiB. FOR-instance uses a different
sensor modality, point distribution and the full semantic stage. Resource and
wall-time requests were calibrated from bounded development work before the
larger arrays. Queue time was recorded separately from execution time.

## Implementation Status And Results

The method-neutral manifest, deterministic five-site Stage 0 selection,
published/default conversion, semantic and instance runners, alignment,
evaluation, automated smoke gate, resource summary and modular Slurm chain are
implemented. Existing pilot files remain available under their legacy
diagnostic names.

The development smoke, compatibility probe, cross-site screen, all-development
refinement, configuration freeze and held-out execution completed. The
retained source-row predictions comprise 22 NPZ files covering both target
routes. The class-3-ignore evaluation reports leaf-on mean plot F1 `0.015023`
and micro F1 `0.016620` from 38 predictions, 323 references and 3 matches.
Leaf-off has 22 predictions and no matches and is retained as a diagnostic.

The separate development-only leaf-attachment screen completed all 45 planned
metrics: nine frozen leaf settings across one plot per site. All settings had
the same aggregate accuracy on this subset, so leaf graph voxel and edge
lengths did not explain the dominant transfer failure. Because the screen used
development data only and followed the held-out execution, it does not change
the frozen test result or authorize another tuned test run.

The published-default configuration and 11-plot test chain are independent of
the development-tuned selection. Run
`tls2trees_for-instance_published_default_held_out_test_20260721_122448`
completed all 22 metrics and retained both leaf targets. Its class-3-ignore
leaf-on headline produced 6 predictions for 323 references, no matches, mean
plot F1 `0.000000` and micro F1 `0.000000`; leaf-off produced 3 predictions
and no matches. The public evidence is
[`tls2trees_published_default_test_results.csv`](../examples/tls2trees_published_default_test_results.csv),
[`tls2trees_published_default_test_provenance.json`](../examples/tls2trees_published_default_test_provenance.json)
and
[`tls2trees_published_default_prediction_retention_manifest.json`](../examples/tls2trees_published_default_prediction_retention_manifest.json).
The negative result indicates weak transfer from terrestrial to UAV laser
scanning for this frozen pipeline, not a failure of the evaluation route.

### Final publication order

The three public finalisers must run sequentially because they write tracked
evidence and, for the two held-out routes, the same three result registries.
Do not submit them concurrently, pull or edit the checkout while a finalisation
job is pending. Complete publication in this order:

1. finalise the neutral retained-prediction evaluation, wait for the Slurm job,
   verify its off-Git receipt and commit any receipt-listed tracked changes;
2. finalise the independent `published_default` held-out result, wait for its
   Slurm job, verify its off-Git receipt, rebuild the public workbook and commit
   the receipt-listed tracked bundle together with the synchronized workbook;
   and
3. export the development-only leaf-screen evidence, verify the three-file
   bundle and commit it.

Each next finaliser starts only from a clean worktree. If a publication job is
interrupted during its multi-file commit, rerun that same finaliser before
starting another one. Tuned-result recovery additionally requires
`TLS2TREES_FINALIZE_RESULTS_RECOVERY_CONFIRMED=1`; published-default recovery
requires `TLS2TREES_PUBLISHED_DEFAULT_RESULTS_RECOVERY_CONFIRMED=1`. Recovery
for the leaf screen requires
`TLS2TREES_LEAF_SCREEN_PUBLICATION_RECOVERY_CONFIRMED=1`. Recovery accepts only
the exact finaliser-owned output, registry and staging paths and must never be
used to bypass unrelated worktree changes. All three Python finalisers also
hold one repository-wide publication lock and repeat the clean/recovery
worktree check while that lock is held.

The neutral retained-prediction finaliser also requires the exact evaluation
run ID, plan path and SHA-256, and historical evaluator path and SHA-256 from
the immutable evaluation state; internally consistent replacements are not a
valid recovery route.

Receipts live below ignored run-scoped `results/metadata/` paths. They verify a
publication but are not themselves committed. Commit only the tracked paths in
the receipt's `written_files` list, plus the synchronized workbook and the
public documentation updates required after a new headline result.

#### 1. Neutral retained-prediction finalisation

Start from the completed final-evaluation state and a clean checkout:

```bash
cd "$HOME/scratch/tree-seg-benchmark"

FINAL_EVALUATION_STATE="$(
  tr -d '\r\n' \
    < logs/tls2trees_for_instance/latest_final_evaluation_state_file.txt
)"
test -s "$FINAL_EVALUATION_STATE"

TLS2TREES_FINALIZE_RESULTS_CONFIRMED=1 \
  bash methods/tls2trees/slurm/for_instance/\
submit_held_out_results_finalisation.sh \
    "$FINAL_EVALUATION_STATE"
```

Monitor the exact finalisation state written by submission:

```bash
FINALISATION_STATE="$(
  tr -d '\r\n' \
    < logs/tls2trees_for_instance/latest_finalisation_state_file.txt
)"
test -s "$FINALISATION_STATE"

watch -n 30 \
  "bash methods/tls2trees/slurm/for_instance/monitor_held_out_results_finalisation.sh '$FINALISATION_STATE'"
```

Stop `watch` after it reports `accounting_state=COMPLETED` and
`TLS2TREES_RESULTS_READY_TO_COMMIT`, then verify the ignored receipt and inspect
the tracked bundle:

```bash
# shellcheck disable=SC1090
source "$FINALISATION_STATE"
test -s "$TLS2TREES_FINALIZE_RECEIPT_JSON"
git status --short
```

If the receipt is valid but no tracked file changed, no empty result commit is
required.

#### 2. Published-default finalisation

The independent test submission, workflow monitor and result-finalisation
command are specified in
[`for_instance_published_default_smoke.md`](for_instance_published_default_smoke.md).
After submitting that result finaliser, monitor its exact Slurm job:

```bash
PUBLISHED_FINALISATION_JOB="$(
  tr -d '\r\n' \
    < logs/tls2trees_for_instance/latest_published_default_finalisation_job_id.txt
)"
test -n "$PUBLISHED_FINALISATION_JOB"

watch -n 30 \
  "squeue -j '$PUBLISHED_FINALISATION_JOB' -o '%.18i %.30j %.10T %.10M %.10L %.30R'; sacct -X -j '$PUBLISHED_FINALISATION_JOB' --format=JobID,JobName%30,State,Elapsed,ExitCode"
```

After accounting reports `COMPLETED`, inspect only that job's logs and verify
the ignored receipt derived from the immutable published-default state:

```bash
tail -n 80 \
  "logs/tls2trees_for_instance/tls2trees_pd_finalise_${PUBLISHED_FINALISATION_JOB}.out"
tail -n 80 \
  "logs/tls2trees_for_instance/tls2trees_pd_finalise_${PUBLISHED_FINALISATION_JOB}.err"

PUBLISHED_STATE="$(
  tr -d '\r\n' \
    < logs/tls2trees_for_instance/latest_published_default_test_state_file.txt
)"
# shellcheck disable=SC1090
source "$PUBLISHED_STATE"
PUBLISHED_RECEIPT="$(
  dirname "$TLS2TREES_PD_TEST_SUMMARY_JSON"
)/published_default_publication_receipt.json"
test -s "$PUBLISHED_RECEIPT"
git status --short
```

#### 3. Development leaf-screen publication

This finaliser is synchronous and therefore has no separate publication job to
monitor. Confirm the source workflow first, then publish from its exact state:

```bash
LEAF_SCREEN_STATE="$(
  tr -d '\r\n' \
    < logs/tls2trees_for_instance/latest_leaf_screen_state_file.txt
)"
test -s "$LEAF_SCREEN_STATE"

bash methods/tls2trees/slurm/for_instance/\
monitor_development_leaf_screen.sh \
  "$LEAF_SCREEN_STATE"

TLS2TREES_LEAF_SCREEN_PUBLICATION_CONFIRMED=1 \
  bash methods/tls2trees/slurm/for_instance/\
finalise_development_leaf_screen_results.sh \
    "$LEAF_SCREEN_STATE"

git status --short -- \
  methods/tls2trees/examples/tls2trees_development_leaf_screen_\*
```

#### Workbook and documentation closure

Whenever a held-out finaliser changes the canonical headline, site or retention
registries, rebuild
`outputs/for_instance_benchmark_metrics/for_instance_method_benchmark_tracker.xlsx`
from those canonical public sources. Expand all four Excel tables, render and
visually inspect the `Comparable Results`, `Site Breakdown`,
`Protocol Alignment` and `Prediction Retention` sheets, and check that no
formula error, clipping, private path or external runtime link is present.

Before committing, update the public result indexes and narrative from the
observed final metrics; do not replace a pending claim before the corresponding
receipt exists. Then run:

```bash
python -m pytest -q \
  tests/test_for_instance_workbook_sync.py \
  tests/test_for_instance_public_result_gate.py \
  tests/test_tls2trees_publication_documentation.py \
  tests/test_benchmark_structure_contract.py
git diff --check
```

The full test suite must also pass before the result branch is merged.

## Risks And Open Ambiguities

- TLS2trees was designed for terrestrial laser-scanning inputs, while
  FOR-instance is UAV laser-scanning data. This is a domain-compatibility
  benchmark and results must be interpreted accordingly.
- The paper does not define a universal `n_tiles`; the value 5 is the closest
  official-example reconstruction, not a uniquely specified default.
- The bundled semantic stack and compiled PyTorch Geometric operations were
  validated on an NVIDIA L40S before benchmark execution.
- Terrain normalisation must reproduce the published DTM intent using available
  geometry without passing reference semantic annotations into the method.
- Upstream drops arbitrary fields, so the adapter must continue proving unique
  raw-coordinate-to-representative assignment before source-row projection.
- Leaf graph edge length is ignored by unmodified upstream code and requires a
  documented behavior-preserving fix at the published value.
- Runtime and memory do not scale reliably from the FRDR operational benchmark.

These ambiguities remain interpretation and reproducibility constraints for
the published-default run. They also explain why the completed
development-tuned score is evidence about this UAV-to-TLS transfer setting,
not a general estimate of TLS2trees accuracy on terrestrial laser scanning.
