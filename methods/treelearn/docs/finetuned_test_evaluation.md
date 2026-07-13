# TreeLearn selected-checkpoint held-out test

This route evaluates the frozen seed-42 epoch-35 checkpoint from the completed
long fine-tuning workflow exactly once. It performs no training, weight update,
post-processing selection or threshold selection.

## Frozen contract

- training uses the official local development subset only;
- evaluation uses the same 11 locally available official test plots used by
  the completed SegmentAnyTree and TreeX routes;
- the evaluation protocol is `for_instance_pointwise_v1` with source-row
  correspondence, the union evaluation mask, IoU `>= 0.5` and
  maximum-cardinality one-to-one matching;
- concurrency is two one-GPU inference tasks;
- five prediction artefacts are retained and hash-verified for every plot;
- results include per-plot, CULS, NIBIO, RMIT, SCION, TUWIEN and overall test
  summaries;
- existing state, freeze, prediction, metadata or table roots cause immediate
  refusal, preventing a repeated test run.

## Submission

Set the exact completed long-run ID, then submit only after manual test
authorization:

```bash
export TREELEARN_LONG_RUN_ID="treelearn_for-instance_fine_tuned_on_dev_long_YYYYMMDD_HHMMSS"

TREELEARN_FINETUNED_TEST_CONFIRMED=1 \
  bash methods/treelearn/slurm/submit_for_instance_finetuned_test.sh
```

The command prints a state-file path. Reconstruct it and monitor without logs:

```bash
STATE_FILE="$HOME/fastscratch/treelearn_finetuned_test_${TREELEARN_LONG_RUN_ID}.env"

watch -n 15 bash \
  methods/treelearn/slurm/monitor_for_instance_finetuned_test.sh \
  "$STATE_FILE"
```

Completion requires all 11 test tasks, the summary job and the final gate to
complete. The gate re-hashes 55 retained prediction files and writes a
completion record. A failed task is documented, but the result is not accepted
as complete and the test outputs must not be used to select another model.
