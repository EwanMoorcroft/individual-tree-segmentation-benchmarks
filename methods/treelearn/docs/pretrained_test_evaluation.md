# TreeLearn Pretrained Held-out Evaluation

This route performs the single authorised evaluation of the clean
authors-released `model_weights_finetuned.pth` checkpoint on the frozen local
11-plot FOR-instance test subset. In benchmark terminology this is the
`published_pretrained` TreeLearn variant: no FOR-instance weight update is
performed.

The route requires checkpoint MD5
`106a80de2991c5f23484a3f9d03e3b16`, upstream commit
`fd240ce7caa4c444fe3418aca454dc578bc557d4`, a clean benchmark checkout and the
same point-aligned evaluator used by the other headline methods. It freezes
all 11 input hashes and inventory counts before inference, allows two
concurrent GPU tasks, retains five raw or aligned prediction artefacts per
plot, and builds overall and site summaries. A stable submission guard refuses
a second test chain even when a new timestamp would otherwise create a new run
directory.

After explicit manual authorization:

```bash
cd "$HOME/scratch/tree-seg-benchmark"
git pull --ff-only

TREELEARN_PRETRAINED_TEST_CONFIRMED=1 \
  bash methods/treelearn/slurm/submit_for_instance_pretrained_test.sh
```

The submitter prints the state-file path. Recover it later and monitor without
reading logs:

```bash
STATE_FILE=$(ls -t \
  "$HOME"/fastscratch/treelearn_pretrained_test_treelearn_for-instance_published_pretrained_*.env \
  | head -1)

watch -n 15 bash \
  methods/treelearn/slurm/monitor_for_instance_pretrained_test.sh \
  "$STATE_FILE"
```

When the monitor reports `completion_status=verified`, display the result and
site breakdown:

```bash
source "$STATE_FILE"
column -s, -t < "$TREELEARN_TEST_FINAL_SUMMARY"
echo
column -s, -t < "$TREELEARN_TEST_TABLE_ROOT/site_summary.csv"
echo
python -m json.tool "$TREELEARN_TEST_COMPLETION_GATE"
```

Do not update the public tracker from partial array output. The fifth headline
row is added only after the completion gate verifies all 11 plots and all 55
retained prediction artefacts.

## Execution-only recovery

Run `treelearn_for-instance_published_pretrained_20260714_134109` completed ten
plots, but the pinned upstream pipeline crashed on test task 8
(`SCION/plot_31_annotated.las`) after producing no valid initial clusters. Its
nearest-neighbour fallback cannot fit without a reference cluster. This is an
execution fault, not permission to change the checkpoint, grouping parameters
or evaluator.

The guarded recovery route is restricted to that run and task. When initial
grouping contains only TreeLearn's unassigned label, it maps those labels to
TreeLearn's background label `0`. This records zero predicted instances for
the plot instead of inventing a cluster. The route archives the original
failure and partial aggregate evidence, removes only failed non-prediction
intermediates, reruns task 8, then rebuilds the unchanged 11-plot summary and
retention gate. The other ten predictions are not rerun.

Attempt 1 exposed another upstream bug: the optional cluster-visualization
writer cannot serialize a zero-row array. Attempt 2 skips only the empty
`cluster_coords_initial` and `cluster_coords` diagnostic LAS files. It
archives the failed attempt's pointwise prediction evidence and still writes
the full all-background prediction used by the evaluator.

```bash
cd "$HOME/scratch/tree-seg-benchmark"
git pull --ff-only

STATE_FILE="$HOME/fastscratch/treelearn_pretrained_test_treelearn_for-instance_published_pretrained_20260714_134109.env"

TREELEARN_PRETRAINED_TEST_RECOVERY_CONFIRMED=1 \
TREELEARN_PRETRAINED_TEST_RECOVERY_ATTEMPT=2 \
TREELEARN_PRETRAINED_TEST_STATE_FILE="$STATE_FILE" \
  bash methods/treelearn/slurm/submit_for_instance_pretrained_test_recovery.sh
```

Monitor the recovery state file printed by the submitter:

```bash
RECOVERY_STATE_FILE=$(ls -t \
  "$HOME"/fastscratch/treelearn_pretrained_test_recovery_treelearn_for-instance_published_pretrained_*.env \
  | head -1)

watch -n 15 bash \
  methods/treelearn/slurm/monitor_for_instance_pretrained_test_recovery.sh \
  "$RECOVERY_STATE_FILE"
```

## Completed result

Recovery attempt 2 completed. Jobs `9775443` through `9775446` passed, the
completion gate verified all 11 plots and all 55 retained prediction files,
and no model or post-processing selection was performed. The frozen result is
mean plot F1 `0.078944` and micro F1 `0.098694`.

The public aggregate tables, provenance record and interpretation are stored
in the [`completed result note`](pretrained_test_results_20260714.md). The
stable submission guard remains active; this held-out route must not be run
again.
