# TreeLearn one-plot FOR-instance development smoke

## Scope

This route installs or verifies a pinned TreeLearn environment, runs one
FOR-instance development plot with the upstream default released checkpoint,
adapts the original-resolution output to source-row labels and evaluates it
with `for_instance_pointwise_v1`.

It does not train, fine-tune, run a full development array or access the
held-out test split. Its accepted evidence is now the prerequisite for the
separate guarded full development-only route.

## Frozen inputs

- Plot: `CULS/plot_1_annotated.las`
- Split: `dev`, verified from `data_split_metadata.csv`
- Expected points: `1816672`
- Expected reference trees: `6`
- TreeLearn commit: `fd240ce7caa4c444fe3418aca454dc578bc557d4`
- Checkpoint: `model_weights_20241213.pth`
- Official checkpoint persistent ID: `IMHF3G`
- Official checkpoint MD5: `56a3d78f689ae7f1190906b975700311`
- Checkpoint SHA-256:
  `5df2f92828f92755bc12e114eaebe83f7ecea94a74c25a6170b68844cc5e19bb`
- IoU threshold: `>= 0.5`
- Matching: maximum-cardinality one-to-one
- Tuned prediction filtering: disabled

The default checkpoint follows the upstream segmentation guide. The alternate
small-tree checkpoint is not part of this run.

## Accepted result

The accepted run is
`treelearn_for-instance_published_pretrained_dev_smoke_20260712_135205`.
Inference and evaluation completed with:

- 1,816,672 source points and 6 reference trees;
- 11 predicted trees, TP 6, FP 5 and FN 0;
- precision `0.545455`, recall `1.000000` and F1 `0.705882`;
- matching source and prediction row counts and preserved row order; and
- maximum absolute coordinate delta `0.00027683842927217484` m, below the
  frozen 0.005 m tolerance.

The upstream full-forest LAZ and NPZ, upstream pointwise NPZ, aligned
prediction NPZ and aligned prediction LAS were all retained with file sizes
and SHA-256 hashes. The result was manually reviewed and accepted only as the
adapter/alignment gate for full development evaluation. It is a one-plot
diagnostic result, not a cross-site or held-out benchmark result. A public-safe
record is stored in
[`../examples/accepted_development_smoke_20260712.json`](../examples/accepted_development_smoke_20260712.json).

## Barkla setup

The setup job is non-destructive: it creates missing paths, but refuses an
existing checkout at another commit, any tracked or untracked checkout change,
or an invalid checkpoint. Existing environments are only reused if the
completion marker, required imports and CUDA check pass. The dependent CPU
evaluation requires the existing `~/fastscratch/venvs/treebench` environment.

Fetch the published smoke branch before setup:

```bash
cd "$HOME/scratch/tree-seg-benchmark"
git fetch origin
git switch --track origin/benchmark/treelearn-dev-smoke
```

```bash
cd "$HOME/scratch/tree-seg-benchmark"
mkdir -p logs/treelearn_for_instance

TREELEARN_SETUP_CONFIRMED=1 \
  sbatch methods/treelearn/slurm/setup_treelearn_environment.sbatch
```

The official environment uses Python 3.10, PyTorch 2.0.0, CUDA 11.8 and
`spconv-cu118`. It also pins `setuptools==80.9.0` because TreeLearn's `munch`
dependency still imports `pkg_resources`. The setup output ends with
`status=treelearn-setup-verified` and prints the downloaded checkpoint SHA-256.
It retains the explicit Conda package list, pip freeze, upstream commit and
checkpoint hash under
`results/metadata/treelearn_for_instance/environment_setups/<setup_job_id>/`.

If an interrupted setup leaves a Conda prefix without the completion marker,
resume only that same guarded installation with:

```bash
TREELEARN_SETUP_CONFIRMED=1 \
TREELEARN_SETUP_RESUME_PARTIAL=1 \
  sbatch methods/treelearn/slurm/setup_treelearn_environment.sbatch
```

The resume first validates the existing prefix. If installation had already
completed, it skips package installation and only finishes checkpoint,
provenance and marker validation.

Setup is expected to take approximately 30-90 minutes. Check it without
streaming logs:

```bash
squeue -j <setup_job_id> -o "%.18i %.24j %.10T %.10M %.9L %.19e %R"
sacct -X -j <setup_job_id> \
  --format=JobID,JobName%24,State,Elapsed,Start,End,ExitCode
```

## Submit the development smoke

After setup succeeds:

```bash
cd "$HOME/scratch/tree-seg-benchmark"

TREELEARN_SMOKE_CONFIRMED=1 \
  bash methods/treelearn/slurm/submit_for_instance_one_plot_smoke.sh
```

The submitter prints a run ID, inference and evaluation job IDs, the frozen
checkpoint SHA-256 and a state-file path. It freezes the clean benchmark
checkout commit and refuses any pre-existing run root. No training or held-out
test job is submitted.

If inference completed but evaluation failed and the final table root does not
exist, retry only the CPU evaluation with the same run ID:

```bash
STATE_FILE="$HOME/fastscratch/treelearn_dev_smoke_<run_id>.env"
source "$STATE_FILE"
test "$(git rev-parse HEAD)" = "$TREELEARN_BENCHMARK_COMMIT"
RETRY_EVALUATION_JOB=$(sbatch --parsable \
  --export="ALL,TREELEARN_RUN_ID=$TREELEARN_RUN_ID,TREELEARN_EXPECTED_BENCHMARK_COMMIT=$TREELEARN_BENCHMARK_COMMIT" \
  methods/treelearn/slurm/evaluate_for_instance_one_plot_smoke.sbatch)
echo "retry_evaluation_job=$RETRY_EVALUATION_JOB"
sacct -X -j "$RETRY_EVALUATION_JOB" \
  --format=JobID,JobName%26,State,Elapsed,Start,End,ExitCode
```

The original state file continues to identify the first evaluation job, so
check the replacement job ID directly with `squeue` or `sacct`.

If inference itself created a run-scoped runtime, prediction or metadata root,
do not overwrite it. Submit again without forcing the old run ID so the
submitter creates a new timestamped run.

## Live monitor

Use the state file printed by the submitter:

```bash
watch -n 15 bash methods/treelearn/slurm/monitor_for_instance_one_plot_smoke.sh \
  "$HOME/fastscratch/treelearn_dev_smoke_<run_id>.env"
```

The monitor shows Slurm state, elapsed time, time remaining, estimated end time
and final metrics. It does not print logs.

## Run-scoped outputs

For `<run_id>` and `CULS_plot_1_annotated`, the retained roots are:

```text
data/interim/treelearn/for_instance/one_plot_smokes/<run_id>/
data/predictions/treelearn/for_instance_smokes/<run_id>/
results/metadata/treelearn_for_instance/one_plot_smokes/<run_id>/
results/tables/treelearn_for_instance/one_plot_smokes/<run_id>/
```

The raw runtime root retains the upstream original-resolution full-forest LAZ
and NPZ plus the voxel-level `pointwise_results.npz`. The prediction root
retains aligned NPZ and LAS outputs. The metadata records input,
split-manifest and checkpoint hashes, both benchmark and upstream commits,
the command, return code, resource evidence, file sizes and hashes for all
retained predictions.

Evaluation writes:

```text
metrics.json
matches.csv
unmatched_predictions.csv
unmatched_references.csv
```

## Semantic and point correspondence contract

- Reference tree points use FOR-instance classes `4`, `5` and `6`.
- Positive TreeLearn instance IDs map to predicted semantic class `4`.
- Prediction labels `0` and `-1` map to background class `0`.
- The aligned NPZ includes `source_row_index == np.arange(point_count)`.
- The adapted LAS includes `pred_treeID` and `pred_classification`.
- The upstream diagnostic `pointwise_results.npz` is not assumed to be aligned
  to source rows and is not used for primary evaluation.

## Success gate

The route passes only when:

- the dataset manifest identifies the plot exactly once as `dev`;
- the source point and reference-tree counts match the frozen inventory;
- the clean upstream checkout matches the pinned commit;
- CUDA, PyTorch, spconv and TreeLearn load;
- inference writes both required upstream outputs;
- original-resolution predictions preserve row count and coordinates;
- at least one positive predicted instance exists;
- adapted fields and row identifiers validate;
- fixed shared evaluation completes; and
- matched and unmatched tables are present.

These gates passed for the accepted run above, including the manual
development-plot alignment review and five-file retention check. The separately
guarded full development route described in
[`development_evaluation.md`](development_evaluation.md) subsequently
completed 21/21 plots. Held-out evaluation must not be submitted from this
smoke route.
