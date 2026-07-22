# Published-checkpoint exposure audit

The public-safe 32-row table is
[`../examples/checkpoint_exposure_32_plots.csv`](../examples/checkpoint_exposure_32_plots.csv).
The retrieval date, persistent source URLs, immutable source commit/blob IDs
and checkpoint hash are recorded in
[`../examples/exposure_evidence_sources.json`](../examples/exposure_evidence_sources.json).

## Evidence

The original ForAINet paper reports 42 training, 14 validation and 11 test
plots. It states that model and post-processing choices were made on the
validation split. The official evaluation configuration lists 11 original
FOR-instance test files. After normalising the authors' PLY names to original
LAS relative paths, that list exactly equals the operational 11-plot held-out
set.

The official conversion code reads the original `data_split_metadata.csv`,
keeps all supplied test rows as test, and uses `random.seed(42)` to choose 25%
of the supplied training rows as validation. Thus every non-test plot belongs
to checkpoint training or validation, while the exact public 42/14 file-level
subdivision is not bundled.

## Result

- Held-out exact matches: 11 of 11.
- Held-out rows used for checkpoint fitting: 0 documented.
- Held-out rows used for model or threshold selection: 0 documented.
- Operational development rows mapped to the combined official fitting or
  validation pool: 21 of 21.
- Duplicate or unmatched operational rows: 0.
- Unresolved evidence: exact train-versus-validation role for each of the 21
  operational development plots and a cryptographic checkpoint-to-inventory
  manifest were not published.

The held-out exposure gate is a documentary pass: all 11 are explicitly and
exactly test-only in the official release. The unresolved 42/14 subdivision is
retained as a provenance limitation. This does not authorise held-out
inference; the development smoke, fine-tuning, freeze and one-time test gates
remain closed.
