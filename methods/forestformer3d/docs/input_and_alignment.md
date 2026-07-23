# ForestFormer3D input and alignment contract

The original LAS is immutable. Before transformation, the converter assigns
`source_row_index` in exact source order and records source hash, point
count, semantic inventory and positive reference-tree count.

ForestFormer3D receives XYZ model input. Reference `classification` and
`treeID` remain in an evaluation sidecar. The official test loader requires
semantic and instance masks, so the one-plot smoke creates two indexes over
the exact same point binary:

- reference: FOR-instance classes 4, 5 and 6 map to internal wood `1`, with
  positive `treeID`; all other rows map to internal ground/instance `0`;
- dummy: every semantic and instance loader label is `0`.

The two cases run in fresh work directories with the same checkpoint and seed.
`semantic_pred`, `instance_pred` and `score` must be exactly equal on every
row. This establishes that required loader labels do not influence predictions.

Primary alignment must use original row order or an official deterministic
inverse map. Coordinate rounding is prohibited as the primary route. The
adapter will fail on dropped, duplicated or reordered rows unless the official
inverse mapping accounts for every source row.

The official converter's coordinate operation is reproduced exactly: subtract
mean X, mean Y and minimum Z in float64, then cast XYZ to float32. Raw PLY XYZ
must equal those staged float32 rows exactly. Upstream instance IDs are
zero-based and use `-1` for background; harmonisation adds one to non-negative
IDs so benchmark-positive IDs never collide with ignored label `0`.

The development smoke remains subject to manual review after its automated
gates pass. Held-out data is not accepted by this workflow.
