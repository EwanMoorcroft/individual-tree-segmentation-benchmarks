# FOR-instance full training run started 4 July 2026

This note records the initial SegmentAnyTree full-training submission. It is a
provenance record, not a result.

The job states below are the states recorded on 4 July. The training job was
later cancelled after producing an epoch-30 checkpoint, and its dependent
arrays were cancelled. Resume and validation outcomes are recorded in
[`training_progress_20260706.md`](training_progress_20260706.md).

## Scope

- Dataset roles: 16 training plots, five development validation plots and 11
  held-out test plots.
- Training profile: `full`.
- Training source: FOR-instance development data only.
- Test data converted for training: no.
- Training mode: from scratch.
- Requested upstream epochs: 150; the Hydra stop value is 151 because the
  upstream loop uses an exclusive stop.

## Barkla jobs

| Stage | Job ID | Recorded state on 4 July 2026 |
| --- | ---: | --- |
| Full split preparation | `9628892` | Completed |
| Full training | `9628896` | Running |
| Five-plot trained validation inference | `9628971` | Queued after training |
| Five-plot aligned validation evaluation | `9628972` | Queued after inference |

The training job used the Barkla working tree based on Git commit `3bd73b2`
with local workflow changes. That historical working tree was preserved while
the chain was active.

## Recorded Barkla source hashes

These hashes distinguish the running implementation from later repository
cleanup:

| File in the pre-layout working tree | SHA-256 |
| --- | --- |
| `scripts/data/prepare_segmentanytree_for_instance_training.py` | `81ed9e04a481a752ea0136e26588cad8355bf22215875d8de8d27f39ad87775d` |
| `scripts/methods/segmentanytree_runtime_patches/run_inference_for_pointwise_evaluation.sh` | `687d09b30b99914d10fb2bfa9e879f65b890eed1f37375c5359f47179590b953` |
| `scripts/slurm/train_segmentanytree_for_instance_full.sbatch` | `f8f9d2fa38487128fe4137cc37e500461c0ae9b128e41ccf14a90a55198d7c14` |
| `scripts/slurm/train_segmentanytree_for_instance_task.sh` | `78dabb6962fb9ac67800cc06168fa89850f06e7e93bc6dc7ed7e42b094a46328` |
| `tests/test_segmentanytree_for_instance_workflow.py` | `c72b1a071e4f69445df1cf39f942f54caf1cd44da488b633bb851999db1b9279` |

The current public repository uses the method-first paths under
`methods/segmentanytree/`. The hashes above remain the authoritative record
for the already-running Barkla job.

## Decision gate

Do not submit the held-out test set merely because training exits successfully.
First verify:

1. the checkpoint exists and its SHA-256 is recorded;
2. all five validation plots complete with equal-length aligned prediction and
   reference arrays;
3. checkpoint selection uses development validation only;
4. the selected checkpoint and evaluation settings are frozen; and
5. the test submission guard is explicitly reviewed.

Those gates were later completed; the selected checkpoint and held-out result
are recorded in
[`training_progress_20260706.md`](training_progress_20260706.md).
