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
