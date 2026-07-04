# Datasets

Dataset folders hold public configuration and dataset-specific inspection
workflows. They do not contain point-cloud data.

- [`for-instance/`](for-instance/) defines the labelled benchmark dataset.
  Its `treeID` field supports point-wise instance accuracy. The supplied split
  contains 21 development plots and 11 held-out test plots.
- [`wytham-woods/`](wytham-woods/) is a future candidate that requires
  reconstruction of per-tree files into a fair plot-level scene.

FRDR treeiso is currently represented through the completed TLS2trees method
workflow because its public work is an operational prediction benchmark rather
than a labelled accuracy benchmark. Dataset suitability is compared in
[`../docs/dataset_feasibility.md`](../docs/dataset_feasibility.md).
