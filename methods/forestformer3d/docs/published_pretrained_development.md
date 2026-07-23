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
