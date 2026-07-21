# Release Readiness Checklist

Status date: 21 July 2026.

Complete this checklist against one clean commit. It is a release gate, not a
claim that the current working tree is already released. Record unknown or
unavailable evidence explicitly and stop when a required item cannot be
verified.

## Protocol And Result Freeze

- [ ] The release identifies `for_instance_pointwise_v1` as the frozen shared
  protocol: 11 supplied test plots, 323 references, source-row pointwise
  predictions, IoU `>= 0.5` and maximum-cardinality one-to-one matching.
- [ ] The two primary harmonised rows and three shared-protocol baselines are
  separated from the two class-3-ignore TLS2trees held-out rows.
- [ ] TLS2trees development-tuned is marked primary within its own protocol but
  `ranking_eligible=false` with
  `exclusion_reason=different_reference_scoring_mask` for the shared ranking.
- [ ] Diagnostic, historical, rejected, operational-only, candidate and failed
  evidence is retained outside the primary leaderboard.
- [ ] A reviewer confirms that no accepted numeric score, run identity or
  evidence path changed without an explicit audit finding.

## Canonical Data And Generated Artefacts

- [ ] The CSV files under `outputs/for_instance_benchmark_metrics/` are the
  canonical source, including the benchmark result registry, test-exposure
  ledger, method-development budget, environment provenance and diagnostic
  availability table.
- [ ] Derived site, plot-distribution and bootstrap-CI diagnostic summaries are
  regenerated deterministically from canonical tables:

  ```bash
  python scripts/reporting/build_for_instance_governance_outputs.py
  ```

- [ ] The review workbook is regenerated from canonical tables, not manually
  edited:

  ```bash
  python scripts/reporting/build_for_instance_workbook.py
  ```

- [ ] Workbook tables and aggregates reconcile with the CSV sources without
  Microsoft Excel or LibreOffice.
- [ ] The generated workbook has been visually checked for clipped columns,
  formula errors, broken links, private paths and ambiguous protocol labels.

## Repository And Environment Provenance

- [ ] The release records the exact repository commit and has a clean worktree.
- [ ] Every implemented method identifies its upstream repository, pinned
  commit and dirty-state evidence.
- [ ] Method-specific environment/container records are present; no method is
  described as using the shared utility environment when it requires another
  container or environment.
- [ ] Available Python, CUDA, PyTorch and MinkowskiEngine versions are recorded;
  unavailable values are `unknown`/`not_recorded`, and inapplicable values are
  `not_applicable`.
- [ ] Every checkpoint-backed row records checkpoint source, SHA-256 and stated
  training datasets, including known overlap limitations.
- [ ] The frozen TLS2trees config hashes still match their published provenance.

## Retention, Exposure And Limitations

- [ ] Every accepted accuracy row has a prediction-retention registry entry and
  its retained-file manifest/hash has been verified at the release commit.
- [ ] `test_exposure_ledger.csv` distinguishes test execution, metric viewing,
  prediction visualisation and later configuration change without invented
  dates or decisions.
- [ ] `method_development_budget.csv` contains only evidence-backed counts and
  does not convert resource requests into guessed compute hours.
- [ ] TreeLearn and TLS2trees validity audits are current and their unresolved
  questions are visible.
- [ ] Additional IoU thresholds, bootstrap intervals and semantic/instance
  decomposition are labelled diagnostic and do not alter primary estimates.
- [ ] Unsupported diagnostics and unavailable off-Git artefacts are explicit.
- [ ] Known dataset, modality, checkpoint-overlap, small-sample uncertainty and
  provenance limitations are included in the dissertation-facing narrative.

## Public Safety And Tests

- [ ] No raw point cloud, prediction array, checkpoint, container, full log,
  private absolute path, username, credential or secret is tracked.
- [ ] Generated LAS fixtures contain synthetic coordinates/labels only.
- [ ] Relative Markdown links resolve.
- [ ] The focused governance, diagnostic, integration, aggregate and workbook
  tests pass.
- [ ] The complete lightweight suite passes on macOS-compatible dependencies.
- [ ] `git diff --check` reports no whitespace error.
- [ ] Hosted CI passes at the exact release commit.

Suggested local verification:

```bash
git status --short --branch
python scripts/reporting/build_for_instance_governance_outputs.py --check
python scripts/reporting/build_for_instance_workbook.py --check
conda run -n tls2trees-local python -m pytest
git diff --check
```

The two check commands compare fresh deterministic payloads byte-for-byte with
the tracked generated artefacts. A standard environment created from the
pinned public requirements may use `python -m pytest` instead.

## Dissertation Release Planning

- [ ] The dissertation-facing release notes state the protocol and governance
  versions, exact commit, accepted result groups and unresolved limitations.
- [ ] The intended tag is reviewed against the evidence available at that
  commit.
- [ ] The branch is reviewed before merge; this governance task itself does not
  merge, tag or publish a GitHub release.

Possible tag sequence (suggestions only):

- `v0.1-protocol-frozen`
- `v0.2-for-instance-primary-results`
- `v1.0-dissertation-submission`

Create a tag only when the corresponding evidence boundary is true and the
full checklist has been signed off. A tag name must not imply a protocol or
result freeze that the tagged commit does not contain.
