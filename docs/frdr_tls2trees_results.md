# FRDR TLS2trees Prediction Results

## Scope

This benchmark ran the TLS2trees instance segmentation stage across all 16
plots in the FRDR treeiso terrestrial laser scanning dataset. Existing FRDR
`woods` values were mapped to the semantic labels required by the instance
stage. The semantic prediction stage was not run.

This is a prediction and operational benchmark. It is not an accuracy benchmark
because the FRDR files provide semantic wood/non-wood labels but do not provide
individual-tree reference instance labels.

## Run Environment

- System: University of Liverpool Barkla2
- Operating system: Rocky Linux 9
- Python: 3.12.10
- Environment: `~/fastscratch/venvs/treebench` Python venv
- TLS2trees repository: <https://github.com/tls-tools-ucl/TLS2trees>
- TLS2trees commit: `ca12cb73b2c736d80b020e8025f8d975d42e6f01`
- Instance entry point: `scripts/methods/tls2trees_patched/instance_patched.py`

The wrapper applies the documented compatibility correction for newer pandas
`groupby.apply` behaviour while leaving the external TLS2trees checkout
unchanged.

## Completion

All configured plots completed:

`LPine_plot1`, `LPine_plot2`, `Mixed_plot1`, `NPoplar_plot1`,
`NPoplar_plot2`, `NSpruce_plot1`, `NSpruce_plot2`, `NSpruce_plot3`,
`RPine_plot1`, `SBirch_plot1`, `SMaple_plot1`, `SPine_plot1`, `SPine_plot2`,
`SPine_plot3`, `TAspen_plot1`, and `TAspen_plot2`.

| Measure | Completed value |
| --- | ---: |
| Plots | 16 |
| Input points | 205,602,855 |
| Retained points | 205,602,854 |
| Dropped unknown points | 1 |
| Predicted trees | 2,036 |
| Points assigned to predicted trees | 27,131,496 |
| Cumulative per-plot process runtime | 19,099.301875 seconds |

`NSpruce_plot2` contained one point with `woods = 0.0`. The configured
`unknown_policy: drop` removed that point during conversion and recorded it in
the conversion and combined summary metadata.

The completed per-plot values are available in
[`examples/tls2trees_frdr_prediction_summary.csv`](../examples/tls2trees_frdr_prediction_summary.csv).
Prediction PLY files, converted PLY inputs and full logs are not distributed in
this repository.

## Memory Note

`Mixed_plot1` was initially killed for exceeding a 32 GiB Slurm allocation. It
completed successfully when rerun with a 96 GiB allocation. The successful run
recorded 49.602968 GiB peak memory usage.

The standard array allocation was 32 GiB. Future runs should treat 96 GiB as
the known allocation for `Mixed_plot1` while retaining the default for other
plots unless scheduler evidence supports a change.

## Interpretation And Limitations

Predicted tree count, predicted-tree point count, runtime, peak memory, return
code and completion status are available for each plot. These values describe
execution and output structure; they do not measure segmentation accuracy.

FRDR `woods` values identify wood and non-wood points. They do not identify
which individual tree owns each point. Consequently, F1, precision, recall and
IoU cannot be calculated from `woods` and are not reported here.

The supported evaluation pathway is documented in
[`evaluation_metrics.md`](evaluation_metrics.md). It requires external
individual-tree reference labels and intentionally refuses to run when those
labels are not supplied.
