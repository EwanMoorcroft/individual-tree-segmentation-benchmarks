# Published-pretrained development diagnostics

This workflow evaluates the unchanged official checkpoint on exactly the 21
frozen original FOR-instance development plots. It is diagnostic evidence for
later development-only fine-tuning design; it does not access or score the 11
held-out plots.

The prerequisite one-plot alignment review was accepted by the benchmark
operator on 2026-07-23. The public-safe decision record is
`examples/manual_alignment_confirmation_20260723.json`; the complete raw
evidence remains in the immutable Barkla smoke run.

## Preflight

`slurm/submit_development_preflight.sh` hashes and inventories only the
canonical development paths through `shared/for_instance_manifest.py`. It
writes an immutable manifest with all source identities, point/tree counts and
a resource estimate based on the successful official-runner smoke.

## Execution

`slurm/submit_published_pretrained_development.sh` requires both the accepted
preflight manifest and the exact smoke-confirmation artifact. It copies both
into a new immutable run root, submits tasks `0-20` with at most two A100 jobs
running at once, and then submits an `afterany` CPU summary. Each task uses one
A100, 12 CPUs, 128 GiB and a two-hour limit.

Each task retains:

- exact converted official inputs and the source-row sidecar;
- the raw official PLY and model/checkpoint audit evidence;
- the harmonised source-row prediction NPZ;
- shared-protocol metrics and resource use; and
- hashes plus a terminal task marker.

The summary fails closed unless all 21 task markers and required artifacts
exist and pass identity checks. Successful aggregation sums TP, FP and FN
before computing micro precision, recall and F1, and weights mean matched IoU
by the number of matches. It writes a full SHA-256 retention manifest.

The workflow does not submit training, tuning, or held-out inference.

## Verified result

Run
`forestformer3d__for-instance__published-pretrained__not-applicable__development__20260723T221726`
completed all 21 tasks with no failed marker. The CPU summary job completed in
44 seconds after the GPU array. Individual A100 task runtimes ranged from
3 minutes 12 seconds to 40 minutes 7 seconds; batch-step peak RSS ranged from
4.21 to 8.03 GiB. Queue waiting is not included in those runtimes.

Across 101,769,037 source rows, the shared evaluator found 740 true positives,
895 false positives and 67 false negatives. The resulting micro precision,
recall and F1 are `0.452599`, `0.916976` and `0.606061`; mean matched IoU
weighted by true positives is `0.911754`. Mean plot F1 is `0.594511`.

The unchanged checkpoint therefore recovers most reference trees but
substantially over-segments them: recall is high while precision is less than
half. The effect is most pronounced on NIBIO (site-micro precision `0.365693`,
recall `0.937198`, F1 `0.526102`). RMIT and SCION are materially stronger.
These are development diagnostics only and are not a held-out leaderboard row.

The summary retained 294 required artefacts totalling 16,565,931,840 bytes.
Independent recovery verification job `9912219` re-hashed all 294 artefacts,
re-read all 21 harmonised NPZ files, proved exact zero-based source-row
identity, reconciled every per-plot count with the summary and confirmed
`held_out_access=false`. It completed in 40 seconds with 0.09 GiB peak batch
RSS. The earlier verification job `9912217` failed before reading artefacts
because the directly executed verifier lacked the repository root on
`sys.path`; its root and logs are retained.

The verified next gate is the frozen development-only fine-tuning protocol in
[`fine_tuning_protocol.md`](fine_tuning_protocol.md).
