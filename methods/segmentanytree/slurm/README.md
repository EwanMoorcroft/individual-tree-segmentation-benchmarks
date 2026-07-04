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

Historical released-checkpoint and export-audit scripts remain available for
traceability, but the trained validation route is the current canonical
workflow.
