# TreeLearn full FOR-instance development evaluation

## Scope

This guarded route evaluates the frozen published TreeLearn checkpoint on the
exact 21 locally available FOR-instance development plots. It performs no
training, fine-tuning, checkpoint selection, threshold selection or
post-processing selection. It cannot read or submit the held-out test split.

The route is ready but has not yet been run. Until its final gate completes,
the accepted one-plot smoke remains diagnostic adapter evidence rather than a
full development result.

## Frozen acceptance evidence

Submission revalidates the manually accepted smoke run
`treelearn_for-instance_published_pretrained_dev_smoke_20260712_135205`. The
accepted smoke used `CULS/plot_1_annotated.las`, preserved row count and order
across 1,816,672 points, and retained all five raw and aligned prediction
artefacts. It obtained precision `0.545455`, recall `1.000000` and F1
`0.705882` with TP 6, FP 5 and FN 0.

The full route freezes:

- TreeLearn commit `fd240ce7caa4c444fe3418aca454dc578bc557d4`;
- checkpoint MD5 `56a3d78f689ae7f1190906b975700311`;
- checkpoint SHA-256
  `5df2f92828f92755bc12e114eaebe83f7ecea94a74c25a6170b68844cc5e19bb`;
- fixed IoU threshold `>= 0.5` and maximum-cardinality one-to-one matching;
  and
- the clean benchmark commit at submission.

## Exact development inventory

The manifest builder selects only metadata rows whose split is exactly `dev`
and whose exact relative path exists under the FOR-instance dataset root. It
does not use basename fallback, path remapping or synthetic expansion, and it
does not inspect test LAS files.

The frozen inventory must contain exactly:

| Site | Plots | Points | Reference trees |
| --- | ---: | ---: | ---: |
| CULS | 2 | 4,901,588 | 27 |
| NIBIO | 14 | 79,435,164 | 414 |
| RMIT | 1 | 1,483,208 | 159 |
| SCION | 3 | 8,380,233 | 92 |
| TUWIEN | 1 | 7,568,844 | 115 |
| Total | 21 | 101,769,037 | 807 |

Each manifest row records its task index, exact relative path, collection,
split, point count, reference-tree count, input SHA-256 and split-metadata
SHA-256. Any inventory difference stops submission.

## Submit on Barkla

The TreeLearn environment, upstream checkout, checkpoint and `treebench`
environment must already pass the one-plot smoke setup. From a clean checkout
of the published branch, confirm both the full development submission and the
completed manual alignment review:

```bash
cd "$HOME/scratch/tree-seg-benchmark"
git pull --ff-only

TREELEARN_FULL_DEV_CONFIRMED=1 \
TREELEARN_DEV_ALIGNMENT_REVIEW_CONFIRMED=1 \
TREELEARN_ACCEPTED_SMOKE_RUN_ID=treelearn_for-instance_published_pretrained_dev_smoke_20260712_135205 \
  bash methods/treelearn/slurm/submit_for_instance_development.sh
```

The submitter rehashes the checkpoint, validates the clean repositories,
checks available scratch space, then prints the run ID, Slurm job IDs and
state-file path. Its preparation job rehashes the accepted smoke artefacts and
freezes the exact development inventory. The dependent combined
inference/evaluation GPU array runs tasks `0-20` with no more than two
concurrent tasks. An `afterany` summary accounts for every array task before
the final gate. Each GPU task uses one L40S GPU, 12 CPUs, 128 GB RAM and a
two-hour wall-time limit. The expected end-to-end runtime is approximately
1-4 hours after jobs start, but plot size and queueing can extend it.

Keep that benchmark checkout on the submitted commit while the dependency
chain is queued or running; do not pull or switch branches until the final gate
has finished.

No training or held-out test job is submitted.

## Monitor without logs

Use the state file printed by the submitter:

```bash
watch -n 15 bash methods/treelearn/slurm/monitor_for_instance_development.sh \
  <state_file>
```

The monitor reports queue state, elapsed time, time remaining, estimated end
time, accounted plots and final summary locations. It does not stream logs.

## Retained outputs

For `<run_id>`, the run-scoped roots are:

```text
data/interim/treelearn/for_instance/development_runs/<run_id>/
data/predictions/treelearn/for_instance_development/<run_id>/
results/metadata/treelearn_for_instance/development_runs/<run_id>/
results/tables/treelearn_for_instance/development_runs/<run_id>/
```

Each successful plot retains the original-resolution upstream full-forest LAZ
and NPZ, upstream pointwise NPZ, aligned prediction NPZ and aligned prediction
LAS. Metadata records their sizes and SHA-256 hashes. The aligned NPZ is the
primary evaluation input; the upstream voxel-level pointwise NPZ remains a
diagnostic artefact.

Evaluation records per-plot metrics, matches, unmatched predictions and
unmatched references. The summary stage accounts for all 21 expected plots,
retains documented failures, and produces these run-level files under the
table root:

- `plot_summary.csv`;
- `site_summary.csv` for CULS, NIBIO, RMIT, SCION and TUWIEN;
- `development_summary.csv` with count-aggregated micro metrics;
- `matches.csv`, `unmatched_predictions.csv` and
  `unmatched_references.csv`;
- `failures.csv`, retaining its schema even when no failures occur;
- `retention_manifest.json`; and
- `run_summary.json`.

## Completion gate

The summary stage accounts for every expected manifest row as either a fixed
completed evaluation or a documented failure. Every successful plot must pass
its retention checks, all result rows must remain `dev`, and the aggregate
tables must agree with the plot-level counts. TP, FP and FN are summed before
micro precision, recall and F1 are computed.

The final completion gate is stricter: all 21 plots must be complete, there
must be zero documented failures, and all 105 retained prediction artefacts
must pass their size and SHA-256 checks. Any missing, changed,
non-development or unretained result blocks that gate. A successful
development gate documents the frozen development result; it does not
authorise held-out evaluation. No TreeLearn held-out test route exists.
