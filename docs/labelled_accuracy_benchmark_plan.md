# Labelled Accuracy Benchmark Plan

## Immediate Pilot

The recommended next pilot is
`FORinstance_dataset/CULS/plot_1_annotated.las`. The inspected file contains
1,816,672 points and six positive reference trees, making it suitable for
testing the preparation, prediction and evaluation interfaces before larger
plots.

Expected fields:

- `treeID`: plot-wise individual-tree reference identifier;
- `classification`: semantic point class;
- first leaf-off reference classes: `4` (stem) and `6` (woody branches);
- ignored classes for the first leaf-off evaluation: `0`, `1`, `2`, `3`, `5`.

Class `5` contains live branches. It is excluded from the first pilot because
the configured TLS2trees instance stage produces leaf-off predictions. Future
leaf-on experiments may include class `5`, but they require a separate
configuration and result label.

## Split Control

Read `data_split_metadata.csv` before selecting any plot. Record each plot's
development or evaluation role in run metadata.

- Use development plots for adapter development, parameter selection and error
  analysis.
- Keep evaluation plots isolated from training and parameter selection.
- Do not select a pilot solely because its evaluation result is favourable.
- Record model checkpoints, training data, parameter provenance and split
  membership for every method.

These controls are required to avoid test-set leakage.

## Preparation Sequence

1. Inspect the selected LAS header, dimensions, class counts and positive
   `treeID` counts without modifying the source file.
2. Confirm the plot's split assignment and intended use.
3. Create a derived method input under `data/interim/`; preserve coordinates
   and keep `treeID` unavailable to the segmentation method.
4. Retain a separate evaluation reference containing coordinates and `treeID`.
5. Apply any semantic filtering consistently and record included and ignored
   classes.
6. Run one method through a wrapper that captures its exact version, command,
   runtime, peak memory, return code and output paths.
7. Convert predictions to one point-cloud file per predicted tree or another
   evaluator-supported instance representation.
8. Validate coordinate alignment, then perform one-to-one instance matching at
   documented coordinate and IoU tolerances.

Source files must remain unchanged.

## Evaluator Inputs And Outputs

The existing evaluator accepts:

- a prediction directory with one LAS, LAZ or PLY file per predicted tree; and
- either a reference directory with one file per tree or one labelled
  point cloud with a reference instance field.

Every accuracy result must include reference and prediction counts, TP, FP, FN,
precision, recall, F1, mean matched IoU, IoU threshold, coordinate tolerance,
ignored labels/classes, runtime and peak memory.

## Method Wrapper Requirements

Each method wrapper should:

- call a pinned external method rather than reimplementing it;
- construct commands as argument lists without shell interpolation;
- keep upstream repositories outside version control;
- separate method inputs, predictions, logs and metadata;
- refuse ambiguous inputs and non-empty output directories by default;
- support a dry-run where the upstream interface permits it;
- emit predictions in a documented instance representation;
- record method version, parameters, environment and resource use.

TLS2trees on FOR-instance is a compatibility test because FOR-instance is UAV
laser scanning data. SegmentAnyTree and TreeLearn or another deep learning
method are candidate accuracy methods, subject to their input and checkpoint
requirements.

## Wytham Follow-On

Wytham Woods provides one PLY per reference tree rather than a ready plot-level
scene. Before evaluation, define a reproducible scene reconstruction that
preserves filename-derived tree IDs and produces the same method input for all
methods. Reference IDs must not leak into method features.

No accuracy claim is valid until predictions have been matched against the
held-out reference instances using the documented evaluator settings.

The implemented first workflow is documented in
[`for_instance_tls2trees_pilot.md`](for_instance_tls2trees_pilot.md).
