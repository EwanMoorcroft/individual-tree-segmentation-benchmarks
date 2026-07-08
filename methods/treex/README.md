# TreeX

TreeX is the completed unsupervised FOR-instance benchmark implemented through
the `pointtree` package API.

## Completed work

The committed public-safe results describe the exact-path-only FOR-instance
subset that exists locally on Barkla:

- 21 development plots;
- 11 held-out test plots; and
- 32 exact-path local plots retained from the Barkla mirror;
- strict and labelled-mask instance metrics.

The headline held-out test result is strict F1 `0.402`; labelled-mask test F1
is `0.522`. TreeX is a completed deterministic baseline here, not a trained or
fine-tuned model.

Start with:

- [`docs/for_instance_benchmark.md`](docs/for_instance_benchmark.md) for the
  Barkla run contract and interpretation;
- [`configs/for_instance_benchmark.yml`](configs/for_instance_benchmark.yml) for
  the fixed dataset, method and evaluation settings; and
- [`examples/treex_combined_dev_test_summary.csv`](examples/treex_combined_dev_test_summary.csv)
  for the authoritative 32-plot result table.

## Tracking boundary

- TreeX was run and evaluated, not trained or fine-tuned.
- Public-safe summary CSVs, JSON metadata and small plots can be committed.
- Full prediction outputs should be backed up locally under `local_outputs/`
  and must stay ignored by Git.

The local backup was audited on 6 July 2026 and contains one `.npz` and one
`.las` final prediction for each of the 32 evaluated plots. These files are not
part of the public repository.

TreeX / `pointtree` is an external dependency and is not vendored here.
