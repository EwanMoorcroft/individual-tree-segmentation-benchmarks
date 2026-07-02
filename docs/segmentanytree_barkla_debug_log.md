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

Apptainer was used for the successful pilot and full benchmark.

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
submission. The canonical pilot then completed prediction, normalisation and
evaluation without a separate postprocessing job. The same route completed all
32 full-array tasks and the final summary.

## Full-Run Outcome

- canonical pilot jobs 9548698, 9548699 and 9548700 completed;
- full prediction array 9548701 completed 32 of 32 tasks;
- full normalisation array 9548702 completed 32 of 32 tasks;
- full evaluation array 9548703 completed 32 of 32 tasks;
- summary job 9548704 completed;
- maximum recorded prediction memory was 9.608 GiB;
- cumulative per-plot prediction runtime was 13,430 seconds.

## Remaining Risks

- The repaired Python stack must remain ahead of the container site-packages
  on `PYTHONPATH`.
- `torch-points-kernels` declares an older NumPy constraint even though the
  tested imports and pilot completed with NumPy 1.24.4.
- The upstream merge added coordinate rows on the initial pilot. Output point
  count and coordinate coverage still require collection-level review.
- NIBIO produced only 12 accepted matches from 575 reference trees. Prediction
  overlays and reference compatibility require targeted validation.
- A rerun should be accepted only when the final LAZ exists and subsequent
  normalisation and evaluation both complete.
