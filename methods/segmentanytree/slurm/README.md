# SegmentAnyTree Slurm workflows

The canonical workflow is grouped by stage:

- `environment/`: one-off container and Python-stack checks;
- `training/`: FOR-instance split preparation and model training;
- `inference/`: released-checkpoint, trained-validation and guarded test
  inference;
- `evaluation/`: export audits, aligned point-wise metrics and summaries.

The accepted training sequence is:

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
