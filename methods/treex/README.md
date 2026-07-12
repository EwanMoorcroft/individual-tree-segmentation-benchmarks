# TreeX

## Method Summary

TreeX is the completed unsupervised FOR-instance benchmark implemented through
the `pointtree` package API. It is a deterministic baseline rather than a
trained or fine-tuned neural model.

## Upstream Repository And Citation

TreeX is provided by the external
[`ai4trees/pointtree`](https://github.com/ai4trees/pointtree) package and is not
vendored here. Cite the
[`treeX` paper](https://doi.org/10.48550/arXiv.2509.03633) when using the
algorithm. The completed Barkla run did not retain the exact installed package
version; that provenance gap is recorded in
[`configs/for_instance_benchmark.yml`](configs/for_instance_benchmark.yml).

## Training Mode Support

The registry records this row as `external_training_only` to show that no
FOR-instance development or test data were used for weight fitting. The method
configuration records the more specific method mode
`unsupervised_parameterised`.

## Input Requirements

The run uses the FOR-instance exact-path subset available locally on Barkla,
with annotated LAS inputs containing `treeID` and `classification` fields.
Tree material uses classes `4`, `5` and `6`.

## Output Contract

TreeX predictions are converted to public-safe per-plot and aggregate summary
tables. Full prediction outputs are local artifacts and must remain outside
Git.

## FOR-instance Compatibility

The run follows the `for_instance_pointwise_v1` protocol on the exact-path
local subset:

The committed public-safe results describe the exact-path-only FOR-instance
subset that exists locally on Barkla:

- 21 development plots;
- 11 held-out test plots; and
- 32 exact-path local plots retained from the Barkla mirror;
- harmonised union-mask metrics and a reference-labelled-mask diagnostic.

## Barkla Environment

The run depends on the Barkla `pointtree` environment documented in the method
config and runbook. External packages, full predictions and local backups are
not part of the public repository.

## Slurm Workflow

Start with:

- [`docs/for_instance_benchmark.md`](docs/for_instance_benchmark.md) for the
  Barkla run contract and interpretation;
- [`configs/for_instance_benchmark.yml`](configs/for_instance_benchmark.yml) for
  the fixed dataset, method and evaluation settings; and
- [`examples/treex_combined_dev_test_summary.csv`](examples/treex_combined_dev_test_summary.csv)
  for the authoritative 32-plot result table.

Current canonical equivalents are:

- preparation: [`scripts/make_treex_for_instance_exact_split_lists.py`](scripts/make_treex_for_instance_exact_split_lists.py);
- inference: [`scripts/run_treex_for_instance_plot.py`](scripts/run_treex_for_instance_plot.py);
- prediction adaptation and evaluation: [`scripts/evaluate_treex_for_instance_plot.py`](scripts/evaluate_treex_for_instance_plot.py);
- local public-result rebuild: [`scripts/rebuild_treex_public_results.py`](scripts/rebuild_treex_public_results.py);
- Barkla summarisation: [`scripts/create_treex_split_summary.py`](scripts/create_treex_split_summary.py) and [`scripts/create_treex_final_summaries.py`](scripts/create_treex_final_summaries.py);
- inference Slurm entrypoints: [`slurm/run_treex_for_instance_dev_array.sbatch`](slurm/run_treex_for_instance_dev_array.sbatch) and [`slurm/run_treex_for_instance_test_array.sbatch`](slurm/run_treex_for_instance_test_array.sbatch); and
- evaluation Slurm entrypoints: [`slurm/evaluate_treex_for_instance_array.sbatch`](slurm/evaluate_treex_for_instance_array.sbatch) and [`slurm/evaluate_treex_for_instance_test_array.sbatch`](slurm/evaluate_treex_for_instance_test_array.sbatch).

## Evaluation Route

The primary harmonised result uses the union of reference-tree and
predicted-tree points with maximum-cardinality one-to-one matching at IoU
`>= 0.5`. Held-out test mean plot F1 is `0.3831`; count-aggregated micro F1 is
`0.3627` from TP=177, FP=476 and FN=146. Reference-labelled-mask mean plot F1
`0.5222` is a secondary diagnostic and is not cross-method comparable.

## Known Limitations

The benchmark is limited to the exact-path local subset available on Barkla.
The full upstream split metadata lists more development and test paths than
were locally available for this run. The exact `pointtree` package version was
not captured in retained Barkla metadata; the config records this provenance
gap directly.

## Current Benchmark Status

- TreeX was run and evaluated, not trained or fine-tuned.
- Its development settings and held-out test result are frozen; the test split
  must not be rerun to select or refine settings.
- Public-safe summary CSVs, JSON metadata and small plots can be committed.
- Full prediction outputs should be backed up locally under `local_outputs/`
  and must stay ignored by Git.

The local backup was audited on 6 July 2026 and contains one `.npz` and one
`.las` final prediction for each of the 32 evaluated plots. These files are not
part of the public repository. Future metrics should be derived from these
retained aligned predictions without rerunning held-out inference.

Run `scripts/verify_treex_prediction_retention.py` after transfers or storage
maintenance. It validates and hashes the frozen 64 final prediction files;
pilot files are ignored and never substituted for a required plot pair.

The current status is recorded in [`../../BENCHMARKS.md`](../../BENCHMARKS.md).
