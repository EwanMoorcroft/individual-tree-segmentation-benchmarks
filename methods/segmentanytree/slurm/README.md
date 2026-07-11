# SegmentAnyTree Slurm workflows

The canonical workflow is grouped by stage:

- `environment/`: one-off container and Python-stack checks;
- `training/`: FOR-instance split preparation and model training;
- `inference/`: released-checkpoint, trained-validation and guarded test
  inference;
- `evaluation/`: export audits, aligned point-wise metrics and summaries.

The historical from-scratch training sequence was:

1. prepare the `full` split;
2. verify the manifest contains 16 training, five validation and zero selected
   test plots;
3. submit the guarded full-training chain;
4. inspect all five validation evaluations;
5. freeze one checkpoint and its settings; and
6. submit held-out test inference separately after explicit review.

Create log directories before calling `sbatch`; Slurm opens output files before
the job body runs.

## Preparation

```bash
cd ~/scratch/tree-seg-benchmark
mkdir -p logs/segmentanytree_for_instance

sbatch \
  --export=ALL,SEGMENTANYTREE_TRAIN_PROFILE=full,SEGMENTANYTREE_OVERWRITE=1 \
  methods/segmentanytree/slurm/training/prepare_for_instance_segmentanytree_splits.sbatch
```

Review
`results/metadata/segmentanytree_for_instance/training_splits/full_split_manifest.json`
after the job completes.

## Training and development validation

The wrapper derives the validation array range from the reviewed manifest,
creates the required log directory and records all submitted job IDs:

```bash
SEGMENTANYTREE_SUBMIT_CONFIRMED=1 \
  bash methods/segmentanytree/slurm/submit_full_training_chain.sh
```

It does not submit held-out test inference. The test jobs remain guarded in
`inference/run_segmentanytree_for_instance_test_from_checkpoint.sbatch` and
`evaluation/evaluate_segmentanytree_for_instance_test_from_checkpoint.sbatch`.

## Short checkpoint continuation

For a reviewed checkpoint, use the same guarded chain with an explicit target
epoch and hash. The following resource profile is the validated L40S
continuation configuration; it does not enable multi-GPU training:

```bash
CHECKPOINT=/absolute/path/to/run/PointGroup-PAPER.pt
CHECKPOINT_SHA256=$(sha256sum "$CHECKPOINT" | awk '{print $1}')

SEGMENTANYTREE_SUBMIT_CONFIRMED=1 \
SEGMENTANYTREE_TRAINING_RUN_ID="sat_for_quicktune_to_TARGET_$(date +%Y%m%d_%H%M%S)" \
SEGMENTANYTREE_RESUME_CHECKPOINT="$CHECKPOINT" \
SEGMENTANYTREE_RESUME_CHECKPOINT_SHA256="$CHECKPOINT_SHA256" \
SEGMENTANYTREE_TRAIN_EPOCHS=TARGET \
SEGMENTANYTREE_TRAIN_BATCH_SIZE=8 \
SEGMENTANYTREE_TRAIN_PARTITION=gpu-l40s \
SEGMENTANYTREE_TRAIN_TIME=04:00:00 \
SEGMENTANYTREE_TRAIN_CPUS=16 \
SEGMENTANYTREE_TRAIN_MEMORY=64G \
SEGMENTANYTREE_MEANSHIFT_JOBS=1 \
SEGMENTANYTREE_OMP_NUM_THREADS=1 \
bash methods/segmentanytree/slurm/submit_full_training_chain.sh
```

Replace `TARGET` with the absolute final epoch number. The wrapper verifies the
checkpoint hash and prior epoch history, then chains the five-plot development
validation inference and evaluation arrays. Checkpoint selection remains
development-only.

Historical released-checkpoint and export-audit scripts remain available for
traceability, but the trained validation route is the current canonical
workflow.

## Released-pretrained development smoke

The first current target is an isolated `published_pretrained` smoke on
`CULS/plot_1_annotated.las`, which is in the supplied development split. It
extracts the complete released model bundle, including Hydra overrides,
verifies the checkpoint SHA-256, runs aligned inference, evaluates the aligned
internal semantic and instance predictions, and rejects zero-instance output.

The route submits no held-out test or fine-tuning job:

```bash
cd ~/scratch/tree-seg-benchmark

SEGMENTANYTREE_PRETRAINED_DEV_SMOKE_CONFIRMED=1 \
  bash methods/segmentanytree/slurm/submit_published_pretrained_dev_smoke.sh
```

Expected resources and runtime:

- checkpoint extraction: `nodes`, one CPU, 2 GB, under 15 minutes;
- development inference: `gpu-l40s-low`, one GPU, eight CPUs, 48 GB, normally
  under one hour;
- aligned evaluation: `nodes`, two CPUs, 32 GB, normally under 30 minutes; and
- final gate: `nodes`, one CPU, 4 GB, normally under ten minutes.

The state file is written under
`~/fastscratch/segmentanytree_published_pretrained_dev_smoke_<timestamp>.env`.
The evidence JSON records the development split, checkpoint hash, upstream
commit, aligned point count, reference and prediction counts, and the manual
review gate. Existing outputs for a repeated timestamp are moved under
`~/fastscratch/segmentanytree_recovery_archive/`; they are not deleted.

Success requires a complete released model bundle, aligned semantic and
instance output, at least one reference instance and at least one predicted
instance. Missing Hydra metadata, a checkpoint mismatch, a non-development
plot, row-length disagreement or zero predicted instances stops the chain.

Review the evidence JSON before adding any held-out test submission route.

## Frozen released-pretrained held-out evaluation

After manual review of a successful development smoke, the released MLS
checkpoint may be evaluated once on the 11 held-out plots. The exact training
plot manifest is not bundled with the released checkpoint, so the submission
requires explicit acceptance of that provenance limitation and the result must
not be described as confirmed leakage-free.

```bash
SMOKE_EVIDENCE="$HOME/scratch/tree-seg-benchmark/results/metadata/segmentanytree_for_instance/stage1_smokes/<run_id>.json"

SEGMENTANYTREE_PUBLISHED_PRETRAINED_TEST_CONFIRMED=1 \
SEGMENTANYTREE_ACCEPT_UNRESOLVED_TRAINING_MANIFEST=1 \
SEGMENTANYTREE_DEV_SMOKE_EVIDENCE="$SMOKE_EVIDENCE" \
  bash methods/segmentanytree/slurm/submit_published_pretrained_test.sh
```

The wrapper freezes the checkpoint hash, Hydra overrides, upstream commit,
matching policy and IoU threshold before submission. It refuses an existing
test output root or freeze manifest rather than silently repeating evaluation.
It submits an 11-task GPU inference array, an 11-task CPU evaluation array and
one final completeness gate. No training or fine-tuning job is submitted.

Use the state file printed by the submitter for a quiet monitor with scheduler
state, elapsed time, time remaining, estimated end time and aligned metric
count, but no log output:

```bash
bash methods/segmentanytree/slurm/monitor_published_pretrained_test.sh \
  <state_file>
```

## Pretrained and fine-tuned comparison

The current fine-tuning route is isolated to the development split. It freezes
the reviewed released checkpoint, the 16/5/0 development split manifest and
the completed Stage 1 evidence before submitting any work:

```bash
RUN_ID=segmentanytree_for-instance_published_pretrained_20260710_231601
STAGE1_ROOT="$HOME/scratch/tree-seg-benchmark/results"

SEGMENTANYTREE_FINETUNE_DEV_CONFIRMED=1 \
SEGMENTANYTREE_STAGE1_TEST_FREEZE="$STAGE1_ROOT/metadata/segmentanytree_for_instance/test_freezes/$RUN_ID.json" \
SEGMENTANYTREE_STAGE1_FINAL_SUMMARY="$STAGE1_ROOT/tables/segmentanytree_for_instance/variants/$RUN_ID/held_out_test/final_summary.csv" \
  bash methods/segmentanytree/slurm/submit_finetuned_dev_validation.sh
```

The chain contains a one-epoch training smoke, a separate 35-epoch training
run initialised from the same released weights, five development inference
tasks, five aligned evaluations and a non-zero-instance gate. Training uses a
fresh optimiser and epoch history, batch size 8 and base learning rate
0.0001. It does not submit held-out inference or evaluation.

Use the printed state file to monitor scheduler state, elapsed time, remaining
time, estimated end time, checkpoint presence and the development metric count
without displaying logs:

```bash
bash methods/segmentanytree/slurm/monitor_finetuned_dev_validation.sh \
  <state_file>
```

Only a successful five-plot development gate permits a later manual decision
about one held-out evaluation. That evaluation is deliberately absent from
this submission route.

After that manual decision, freeze and submit exactly one held-out evaluation
of the accepted development run:

```bash
export SEGMENTANYTREE_FINETUNED_TEST_CONFIRMED=1
export SEGMENTANYTREE_TRAINING_RUN_ID=<accepted_fine_tuned_run_id>
bash methods/segmentanytree/slurm/submit_finetuned_test.sh
```

The wrapper verifies the development freeze, five-plot summary, training
metadata, released-weight compatibility and fine-tuned checkpoint hash. It
refuses any existing prediction, metric, table or freeze target. It submits no
training job and records that the test must not be repeated for setting
selection. Monitor the printed state file without logs using:

```bash
bash methods/segmentanytree/slurm/monitor_finetuned_test.sh <state_file>
```

The later target comparison has two variants:

1. `published_pretrained`: extract the released checkpoint from the pinned
   container, verify its hash and evaluate it without weight updates; and
2. `fine_tuned_on_dev`: initialise from those released weights with a fresh
   optimiser and epoch history, then train for 35 epochs on the 16 development
   training plots.

The earlier `retrained_from_dev` result is retained as historical evidence. It
is not required by this submission chain and is not included in the target
comparison CSV.

The fine-tune path is weight-only initialisation, not checkpoint resume. It
requires at least 95 percent of the current model state by tensor size to be
shape-compatible. A one-epoch full-data smoke job must complete before the
35-epoch job starts. The held-out test array remains locked until five aligned
development-validation metrics exist and contain predicted instances.

The combined wrapper remains historical implementation evidence. Do not run
it until the isolated released-pretrained development smoke has passed and a
separate held-out-test freeze boundary has been reviewed:

```bash
cd ~/scratch/tree-seg-benchmark

SEGMENTANYTREE_PRETRAINED_FINETUNE_CONFIRMED=1 \
  bash methods/segmentanytree/slurm/submit_pretrained_finetune_comparison.sh
```

This command is intentionally not part of the development-smoke route.

The final CSV is written under
`results/tables/segmentanytree_for_instance/pretrained_finetune_comparison/`
and contains one row for each target variant. It reports mean plot and micro
metrics under the aligned point-wise protocol. A failed smoke, hash,
compatibility, split, alignment or zero-prediction gate prevents its dependent
jobs from starting.

All run IDs and output roots are timestamped. Recovery moves partial pretrained
predictions, metadata, metrics and tables to
`~/fastscratch/segmentanytree_recovery_archive/` before resubmitting; it never
deletes them. Use `recover_pretrained_finetune_pretrained.sh` only after
reviewing the state file and confirming the fine-tune branch is healthy.

The `three_variation` script names remain as compatibility implementation
details for state files created before this plan changed. New work should use
the canonical commands above.
