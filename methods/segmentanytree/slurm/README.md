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

## Pretrained and fine-tuned comparison

The current target comparison has two variants:

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

From a clean Barkla checkout at the current `origin/main`:

```bash
cd ~/scratch/tree-seg-benchmark

SEGMENTANYTREE_PRETRAINED_FINETUNE_CONFIRMED=1 \
  bash methods/segmentanytree/slurm/submit_pretrained_finetune_comparison.sh

bash methods/segmentanytree/slurm/monitor_pretrained_finetune_comparison.sh --follow
```

The monitor clears and redraws five compact status lines once per minute. It
does not print scheduler logs. Historical timings imply roughly six to eight
hours after GPU allocation. The monitor recalculates the ETA from observed
epoch progress once full training is running.

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
