# ForestFormer3D input and alignment contract

The original LAS is immutable. Before transformation, the converter will
assign `source_row_index` in exact source order and record source hash, point
count, semantic inventory and positive reference-tree count.

ForestFormer3D receives XYZ model input. Reference `classification` and
`treeID` remain evaluation-side information. If the official loader requires
placeholder labels, inference must pass a counterfactual test in which changing
those placeholders cannot change predictions.

Primary alignment must use original row order or an official deterministic
inverse map. Coordinate rounding is prohibited as the primary route. The
adapter will fail on dropped, duplicated or reordered rows unless the official
inverse mapping accounts for every source row.

Implementation of the converter and normaliser follows successful full-image
qualification; no inference job is available yet.
