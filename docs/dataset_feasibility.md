# Dataset Feasibility

This assessment separates completed prediction work from candidate datasets
that can support individual-tree accuracy evaluation. Dataset files remain
outside this repository.

## Summary

| Dataset | Inspection status | Instance labels | Accuracy status | Recommended role |
| --- | --- | --- | --- | --- |
| FRDR treeiso TLS | Downloaded; 16-plot TLS2trees run completed | Not present in the benchmark LAZ inputs | F1/IoU unavailable from `woods` | Completed prediction and operational benchmark |
| FOR-instance | Downloaded and inventoried; 32-plot SegmentAnyTree benchmark completed | `treeID` in annotated LAS files | Full prediction and F1/IoU evaluation completed | Completed primary accuracy benchmark |
| Wytham Woods | Downloaded, unpacked and inventoried | One segmented tree per file | F1/IoU feasible after scene reconstruction | TLS accuracy benchmark candidate |

## FRDR Treeiso TLS

| Field | Assessment |
| --- | --- |
| Source | [FRDR dataset record](https://www.frdr-dfdr.ca/repo/dataset/ccf5e057-11c6-4149-8387-d52b519f9b2c); [dataset publication](https://doi.org/10.3390/rs14236116) |
| Barkla path | `~/data/datasets/frdr_treeiso` |
| Sensor/data type | Terrestrial laser scanning plot point clouds |
| Annotation available locally | Binary `woods` semantic field: wood and non-wood |
| Individual-tree labels | No reference tree IDs in the benchmark LAZ inputs |
| Accuracy feasibility | F1, precision, recall and IoU cannot be derived from `woods` |
| Compatible methods | TLS2trees completed; other TLS and traditional methods remain possible prediction tests |
| Preprocessing | Method-specific conversion; TLS2trees mapping from `woods`; documented local-minimum `n_z` approximation |
| Risks and limitations | No instance accuracy reference; output counts depend on parameters and point density |
| Recommended role | Completed prediction, runtime, memory and output-validity benchmark |

## FOR-instance

| Field | Assessment |
| --- | --- |
| Source | [Zenodo record 8287792](https://doi.org/10.5281/zenodo.8287792) |
| Barkla path | `~/data/datasets/for_instance/FORinstance_dataset` |
| Download status | `FORinstance_dataset.zip` (approximately 1.6 GB) downloaded and unpacked; five collections, `data_split_metadata.csv` and collection-level `tree_data_*.csv` files present |
| Sensor/data type | Dense UAV laser scanning point clouds |
| Annotation | Plot-wise `treeID` plus semantic `classification` values |
| Individual-tree labels | Available for positive `treeID` values |
| Accuracy feasibility | One-to-one matched precision, recall, F1 and IoU are feasible |
| Compatible methods | SegmentAnyTree; TreeLearn and other deep learning methods; traditional baselines; TLS2trees compatibility test |
| Preprocessing | Respect `data_split_metadata.csv`; retain reference IDs separately; build method-specific inputs and prediction adapters |
| Risks and limitations | Collection and sensor heterogeneity; class imbalance; potential test-set leakage; confirm positive IDs before use |
| Recommended role | Completed primary SegmentAnyTree accuracy benchmark; collection-specific validation remains |

The inspected inventory contains 32 LAS files, 151,478,959 points and 1,130
positive reference tree IDs. Semantic classes are:

| Code | Meaning |
| ---: | --- |
| 0 | Unclassified or scattered unannotated points |
| 1 | Low vegetation |
| 2 | Terrain |
| 3 | Out-points |
| 4 | Stem |
| 5 | Live branches |
| 6 | Woody branches |

The SegmentAnyTree benchmark uses tree-material classes `4`, `5` and `6`, with
classes `0`, `1`, `2` and `3` ignored. Prediction, normalisation and F1/IoU
evaluation completed for all 32 plots. Aggregate and collection-level results
are in
[`segmentanytree_for_instance_results.md`](segmentanytree_for_instance_results.md).
The separate TLS2trees compatibility pilot retains its leaf-off `4` and `6`
filter.

## Wytham Woods

| Field | Assessment |
| --- | --- |
| Source | [Zenodo record 7307956](https://doi.org/10.5281/zenodo.7307956) |
| Barkla path | `~/data/datasets/wytham_woods/DATA_clouds_ply/DATA_clouds_ply` |
| Download status | `DATA_clouds_ply.zip` (approximately 1.4 GB) downloaded and unpacked; 876 PLY files present |
| Sensor/data type | Leaf-off terrestrial laser scanning with a RIEGL VZ-400 |
| Annotation | Each PLY represents one segmented reference tree; filenames provide tree IDs |
| Individual-tree labels | Available through per-tree files |
| Accuracy feasibility | Feasible after reconstructing aligned plot-level inputs and references |
| Compatible methods | TLS2trees; SegmentAnyTree; TreeLearn; traditional TLS methods |
| Preprocessing | Select a scene, merge trees while preserving IDs, construct method inputs, and validate coordinate alignment |
| Risks and limitations | Per-tree files contain only `x`, `y`, `z`; no ready plot scene or non-tree context; reconstruction choices can bias comparisons |
| Recommended role | Strong TLS accuracy benchmark candidate after a documented reconstruction protocol |

Wytham should not be treated as immediately method-ready. The same reconstructed
scene and reference definition must be used across methods for a fair
comparison.
