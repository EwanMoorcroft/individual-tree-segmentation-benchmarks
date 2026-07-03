# SegmentAnyTree Barkla Deployment Notes

## Working Route

SegmentAnyTree was deployed on Barkla2 using Apptainer 1.3.6. The final
32-task prediction array ran on `gpu-l40s-low`; earlier validation used
`gpu-l40s`. The SIF was built from
`docker://maciekwielgosz/segment-any-tree:latest`; GPU visibility was confirmed
on `gpu42` and `gpu43`, both with NVIDIA L40S GPUs and approximately 46 GB
VRAM. The image and external repository remain outside Git.

Podman was also tested, but the Slurm job could not access its rootless runtime
directory:

```text
Failed to obtain podman configuration: lstat /run/user/<uid>: no such file or directory
```

Apptainer was used for the successful pilot and full inference array.

## Issues Resolved

1. The container initially failed to import pandas:
   `ImportError: cannot import name 'DtypeArg' from 'pandas._typing'`.
2. After that correction, scikit-learn failed with
   `ImportError: cannot import name 'METRIC_MAPPING64' from
   'sklearn.metrics._dist_metrics'`, indicating another mixed package
   installation.
3. A controlled userbase at
   `~/fastscratch/segmentanytree_pyuser_v1` was populated with NumPy 1.24.4,
   pandas 1.5.3, SciPy 1.10.1 and scikit-learn 1.3.2. Core imports, KDTree,
   PyTorch, torch-geometric, MinkowskiEngine, torch-points3d and `eval` then
   passed.
4. The SIF could not write its processed-data cache. A per-task fast-scratch
   directory is now bound over the expected container cache path.
5. Inference later blocked in a one-worker clustering
   `multiprocessing.Pool` after CUDA initialisation. Process inspection showed
   the parent and child waiting on futexes while GPU use stayed at zero. The
   opt-in serial pool compatibility layer preserves the one-worker map order
   without forking.
6. Instance prediction completed, but the final LAS writer attempted to cast
   floating `scan_angle` values to unsigned 16-bit integers. LAS point format 6
   uses signed scan angles. The runtime preparation script changes this field
   to signed 16-bit and rounds before conversion.

The first export repair exposed an indentation error in the prepared Python
source. The patch generator was corrected and compile-checked before
submission. The canonical pilot then completed inference and export without a
separate postprocessing job. The same route produced predictions for all 32
plots.

## Recorded Execution Outcome

- canonical pilot jobs 9548698, 9548699 and 9548700 completed inference,
  coordinate normalisation and the original evaluation;
- full prediction array 9548701 completed 32 of 32 tasks;
- full normalisation array 9548702 completed 32 of 32 tasks;
- full coordinate-rematched evaluation array 9548703 completed 32 of 32 tasks;
- summary job 9548704 completed;
- maximum recorded prediction memory was 9.608 GiB;
- cumulative per-plot prediction runtime was 13,430 seconds.

The process completion records establish operational success. They do not
validate the provisional accuracy values.

## Evaluation Failure Investigation

The first evaluator split each final LAZ by `PredInstance` and matched exported
XYZ coordinates back to the source LAS using a 0.02 m tolerance. That route
does not reproduce the SegmentAnyTree paper, which evaluates point-aligned
semantic and instance arrays directly.

The upstream final merge indexes intermediate tables by stringified XYZ
coordinates. Duplicate coordinates can therefore create ambiguous joins or
additional rows. This matters particularly for dense point clouds. The initial
CULS pilot export already contained 56 more rows than its source.

The strongest warning was NIBIO: the provisional result contained 575
reference trees, 1,591 predicted instances and only 12 accepted matches, with
16 of 20 plots producing no accepted match. This is inconsistent with the
published NIBIO result and is too large a discrepancy to treat as ordinary
forest-domain variation without first excluding export and evaluation errors.

The first test-split audit snapshot contained nine completed audits; all nine
failed, while two NIBIO tasks were still running. Row inflation among the
completed audits ranged from two rows for RMIT to 2,056,634 rows for TUWIEN.
Three inspected NIBIO examples gained 104,470, 164,290 and 164,168 rows. These
failures confirm that the final-LAZ route is not a valid benchmark-wide
point-wise evaluation input.

The pinned upstream tracker already computes a full-resolution instance array
aligned with `test_area_i.instance_labels`, but its instance `to_eval_ply`
block is commented out. The corrected rerun enables one narrow output call at
runtime, retains the existing semantic evaluation PLY and stops before the
coordinate-based final merge. No model weights or inference parameters are
changed.

## Remaining Risks And Required Checks

- The repaired Python stack must remain ahead of the container site-packages
  on `PYTHONPATH`.
- `torch-points-kernels` declares an older NumPy constraint even though the
  tested imports and pilot completed with NumPy 1.24.4.
- The checkpoint SHA-256 is
  `0b4d74b4644e37a16f59008ad0f5c62894fc4d2d906f3abd803bbfc5b5dd803a`;
  its precise upstream training scenario still requires confirmation.
- Final exports failed point-count and coordinate-multiplicity audits and must
  not be used for accuracy.
- The new aligned instance output requires a two-plot validation before the
  remaining test plots are submitted.
- Only the supplied FOR-instance test split may be compared with the paper.
- The paper-compatible matching policy and strict one-to-one policy must remain
  separate in summaries.

## Methodological Correction

The completed 32-plot array demonstrated that the container and inference path
worked, but it used the released checkpoint and did not update model weights
with FOR-instance development data. It is retained as a diagnostic
inference-only run.

The corrected workflow now mirrors the pinned upstream training preparation:
seed 42, a 25% internal validation sample from development plots, binary
tree/non-tree labels and no test files in the training data root. A small
training preflight must succeed before the 16-plot training and 5-plot
validation profile is submitted.

The released checkpoint is not used to initialise the primary corrected run.
Its resume path restores the saved training configuration and optimizer state,
which would define a separate fine-tuning experiment and requires its own
compatibility validation. Final test inference remains guarded until the full
development-trained checkpoint is selected and frozen.
