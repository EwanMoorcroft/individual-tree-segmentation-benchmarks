# SegmentAnyTree Barkla Deployment Notes

## Working Route

SegmentAnyTree was deployed on Barkla2 using Apptainer 1.3.6 on the
`gpu-l40s` partition. The SIF was built from
`docker://maciekwielgosz/segment-any-tree:latest`; GPU visibility was confirmed
on `gpu42` and `gpu43`, both with NVIDIA L40S GPUs and approximately 46 GB
VRAM. The image and external repository remain outside Git.

Podman was also tested, but the Slurm job could not access its rootless runtime
directory:

```text
Failed to obtain podman configuration: lstat /run/user/<uid>: no such file or directory
```

Apptainer was used for all subsequent tests and the successful pilot.

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

The final export repair produced a valid LAZ, after which normalisation and
labelled evaluation completed.

## Scaling Risks

- The repaired Python stack must remain ahead of the container site-packages
  on `PYTHONPATH`.
- `torch-points-kernels` declares an older NumPy constraint even though the
  tested imports and pilot completed with NumPy 1.24.4.
- The upstream merge added 56 coordinate rows on the pilot. Output point count
  and coordinate matches must be checked for every plot.
- Plot size, clustering load, runtime and memory may vary substantially across
  the five FOR-instance collections.
- A full task should be accepted only when the final LAZ exists and subsequent
  normalisation and evaluation both complete.
