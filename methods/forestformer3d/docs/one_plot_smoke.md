# ForestFormer3D development one-plot smoke

The only permitted input is development plot `CULS/plot_1_annotated.las`.
The submitter verifies the frozen split catalogue SHA-256 before submission and
contains no held-out path or held-out opt-in.

The workflow stages exact source-order XYZ and two loader-label cases. Both
cases use the same point binary, official source commit, official config,
unchanged checkpoint and identical seed. The reference case supplies
mapped classes 4/5/6 and positive tree IDs; the dummy case supplies zero labels.
The adapter rejects any difference in official `semantic_pred`,
`instance_pred` or `score` fields.

Strict PyTorch deterministic-algorithm mode is disabled because upstream uses
CUDA `index_reduce_(reduce='amax')`, for which PyTorch 1.13.1 exposes no
deterministic implementation. Both fresh processes still use seed 3407, and
the exact counterfactual comparison fails closed if realized nondeterminism or
loader labels alter any prediction.

The effective whole-plot method writes
`forestformer3d_smoke_test.ply`. Upstream's `UnifiedSegMetric` is incompatible
with this whole-plot path because the returned in-memory prediction represents
only the last region. The smoke registers a no-op MMEngine metric and validates
the complete official PLY independently; model prediction logic is unchanged.

The raw official PLY rows must exactly equal the normalized staged float32 XYZ.
The accepted NPZ restores the zero-based identity `source_row_index`, maps
upstream non-negative instance IDs to positive benchmark IDs and maps predicted
tree rows to benchmark class 4. Both official raw PLY files, the evaluation
sidecar, hashes and validation JSON are retained.

The pinned official `tools/test.py` omits `import torch`. The adapter verifies
the source commit and file SHA-256, then supplies `torch` as an initial global
when executing the unchanged file. This is an interface compatibility shim,
not a model-source or checkpoint modification.

The published checkpoint is already in the fixed RSKC sparse-weight layout,
but pinned `tools/test.py` applies the fix unconditionally. The adapter accepts
only the exact published checkpoint SHA-256 and preconditions its 49 sparse
tensors with the inverse permutation. It proves that the unchanged upstream
permutation reconstructs every original tensor exactly and records this in
`checkpoint_entrypoint_adapter.json`.

Submission from the clean Barkla `method/forestformer3d` checkout:

```bash
FF3D_SMOKE_CONFIRMED=1 \
FF3D_BENCHMARK_ROOT="$PWD" \
FF3D_MONITOR_SECONDS=30 \
bash methods/forestformer3d/slurm/submit_one_plot_smoke.sh
```

Passing automated gates does not authorize a full development run. First
inspect the raw and harmonised artifacts and record the required manual
alignment review.
