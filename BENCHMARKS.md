# Benchmark Registry

This repository is structured to hold multiple individual tree segmentation
benchmarks. Each benchmark should provide a reproducible configuration, input
adapter, method runner, scheduler workflow, metadata outputs and focused tests.

## Benchmark Registry

Registry rows describe dataset-method runs or explicit candidates. A completed
or provisional row must include the dataset slug, method slug, run label,
training mode declaration, evaluation mode, status and evidence file. Candidate
rows may use `pending` for run-specific fields until a run is scheduled.

| Dataset slug | Method slug | Run label | Training mode | Evaluation mode | Status | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| frdr-treeiso | tls2trees | tls2trees_frdr_prediction_benchmark | external_training_only | operational_prediction | Prediction benchmark completed | [`frdr_benchmark.yml`](methods/tls2trees/configs/frdr_benchmark.yml); [`tls2trees_frdr_prediction_summary.csv`](methods/tls2trees/examples/tls2trees_frdr_prediction_summary.csv) |
| for-instance | segmentanytree | released_checkpoint_coordinate_rematch | published_pretrained | provisional_coordinate_rematched | Provisional inference-only run completed; export audit failed | [`for_instance_benchmark.yml`](methods/segmentanytree/configs/for_instance_benchmark.yml); [`provisional_released_checkpoint_results.md`](methods/segmentanytree/docs/provisional_released_checkpoint_results.md) |
| for-instance | segmentanytree | segmentanytree_for-instance_published_pretrained_20260710_231601 | published_pretrained | harmonised_pointwise_test | Completed target baseline; test mean plot F1 0.4534, micro F1 0.4442 | [`per-plot results`](methods/segmentanytree/examples/sat_completed_target_plot_results_20260711.csv); [`overall results`](methods/segmentanytree/examples/sat_completed_target_results_20260711.csv); [`site results`](methods/segmentanytree/examples/sat_completed_target_site_results_20260711.csv); [`final result note`](methods/segmentanytree/docs/final_results_20260711.md) |
| for-instance | segmentanytree | segmentanytree_for-instance_fine_tuned_on_dev_20260711_002931 | fine_tuned_on_dev | harmonised_pointwise_test | Completed primary result; test mean plot F1 0.5447, micro F1 0.5320 | [`per-plot results`](methods/segmentanytree/examples/sat_completed_target_plot_results_20260711.csv); [`overall results`](methods/segmentanytree/examples/sat_completed_target_results_20260711.csv); [`site results`](methods/segmentanytree/examples/sat_completed_target_site_results_20260711.csv); [`final result note`](methods/segmentanytree/docs/final_results_20260711.md) |
| for-instance | segmentanytree | sat_for_quicktune_to49_20260706_140730 | retrained_from_dev | harmonised_pointwise_test | Completed historical result retained; test mean plot F1 0.4825, micro F1 0.4692 | [`sat_final_test_aligned_summary_sat_for_quicktune_to49_20260706_140730.csv`](methods/segmentanytree/examples/sat_final_test_aligned_summary_sat_for_quicktune_to49_20260706_140730.csv); [`provenance manifest`](methods/segmentanytree/examples/sat_final_test_aligned_provenance_sat_for_quicktune_to49_20260706_140730.json) |
| for-instance | segmentanytree | sat_for_quicktune_to55_20260707_214305 | retrained_from_dev | development_validation | Rejected validation regression | [`training_progress_20260706.md`](methods/segmentanytree/docs/training_progress_20260706.md) |
| for-instance | segmentanytree | segmentanytree_for-instance_fine_tuned_on_dev_20260708_215054_full | fine_tuned_on_dev | diagnostic_held_out_subset | Rejected; produced semantic tree predictions but zero instance predictions | [`training_progress_20260706.md`](methods/segmentanytree/docs/training_progress_20260706.md) |
| for-instance | treex | treex_for_instance_exact_path_subset | external_training_only | harmonised_pointwise_test | Completed and frozen unsupervised parameterised baseline; test mean plot F1 0.3831, micro F1 0.3627 | [`per-plot results`](methods/treex/examples/treex_combined_dev_test_summary.csv); [`split summary`](methods/treex/examples/treex_split_summary.csv); [`retention manifest`](methods/treex/examples/treex_prediction_retention_manifest.json) |
| for-instance | tls2trees | tls2trees_for_instance_leaf_off_pilot | external_training_only | legacy_oracle_semantic_coordinate_fallback | Legacy one-plot instance-stage diagnostic; not a target benchmark row | [`for_instance_accuracy.yml`](methods/tls2trees/configs/for_instance_accuracy.yml); [`for_instance_pilot.md`](methods/tls2trees/docs/for_instance_pilot.md) |
| for-instance | tls2trees | tls2trees_for-instance_published_default_held_out_test_20260721_122448 | external_training_only | class3_ignore_leaf_off_diagnostic | Completed target-specific diagnostic; 3 predictions, 323 references, zero matches | [`per-plot diagnostic`](methods/tls2trees/examples/tls2trees_published_default_leaf_off_test_plot_diagnostic.csv); [`site diagnostic`](methods/tls2trees/examples/tls2trees_published_default_leaf_off_test_site_diagnostic.csv); [`overall diagnostic`](methods/tls2trees/examples/tls2trees_published_default_leaf_off_test_diagnostic.csv); [`provenance`](methods/tls2trees/examples/tls2trees_published_default_test_provenance.json); [`retention manifest`](methods/tls2trees/examples/tls2trees_published_default_prediction_retention_manifest.json) |
| for-instance | tls2trees | tls2trees_for-instance_development_tuned_held_out_test_20260719_110219 | external_training_only | class3_ignore_leaf_off_diagnostic | Completed target-specific diagnostic; 22 predictions, 323 references, zero matches | [`per-plot diagnostic`](methods/tls2trees/examples/tls2trees_development_tuned_leaf_off_test_plot_diagnostic.csv); [`overall diagnostic`](methods/tls2trees/examples/tls2trees_development_tuned_leaf_off_test_diagnostic.csv); [`retention manifest`](methods/tls2trees/examples/tls2trees_development_tuned_prediction_retention_manifest.json) |
| for-instance | tls2trees | tls2trees_for-instance_published_default_held_out_test_20260721_122448 | external_training_only | class3_ignore_leaf_on_test | Completed without FOR-instance metric selection; mean plot F1 0.0000, micro F1 0.0000 | [`per-plot results`](methods/tls2trees/examples/tls2trees_published_default_test_plot_results.csv); [`site results`](methods/tls2trees/examples/tls2trees_published_default_test_site_results.csv); [`overall results`](methods/tls2trees/examples/tls2trees_published_default_test_results.csv); [`provenance`](methods/tls2trees/examples/tls2trees_published_default_test_provenance.json); [`retention manifest`](methods/tls2trees/examples/tls2trees_published_default_prediction_retention_manifest.json) |
| for-instance | tls2trees | tls2trees_for-instance_development_tuned_held_out_test_20260719_110219 | external_training_only | class3_ignore_leaf_on_test | Completed; mean plot F1 0.0150, micro F1 0.0166 | [`per-plot results`](methods/tls2trees/examples/tls2trees_development_tuned_test_plot_results.csv); [`overall results`](methods/tls2trees/examples/tls2trees_development_tuned_test_results.csv); [`site results`](methods/tls2trees/examples/tls2trees_development_tuned_test_site_results.csv); [`provenance`](methods/tls2trees/examples/tls2trees_development_tuned_test_provenance.json); [`retention manifest`](methods/tls2trees/examples/tls2trees_development_tuned_prediction_retention_manifest.json) |
| for-instance | tls2trees | tls2trees_for-instance_development_tuned_leaf_screen_20260720_193825 | external_training_only | development_leaf_attachment_screen | Completed development-only diagnostic; 45/45 valid metrics and identical aggregate accuracy across nine settings | [`plot results`](methods/tls2trees/examples/tls2trees_development_leaf_screen_plot_results.csv); [`candidate results`](methods/tls2trees/examples/tls2trees_development_leaf_screen_candidate_results.csv); [`provenance`](methods/tls2trees/examples/tls2trees_development_leaf_screen_provenance.json) |
| for-instance | treelearn | treelearn_for-instance_published_pretrained_dev_smoke_20260712_135205 | published_pretrained | development_smoke_harmonised_pointwise | Accepted adapter diagnostic; one CULS development plot F1 0.7059 | [`accepted smoke`](methods/treelearn/examples/accepted_development_smoke_20260712.json); [`runbook`](methods/treelearn/docs/one_plot_smoke.md) |
| for-instance | treelearn | treelearn_for-instance_published_pretrained_development_20260712_150030 | published_pretrained | full_development_harmonised_pointwise | Completed published-checkpoint development diagnostic with documented FOR-instance training overlap; mean plot F1 0.5156, micro F1 0.5108; excluded from leakage-free ranking | [`overall results`](methods/treelearn/examples/treelearn_completed_development_results_20260712.csv); [`site results`](methods/treelearn/examples/treelearn_completed_development_site_results_20260712.csv); [`result note`](methods/treelearn/docs/development_results_20260712.md) |
| for-instance | treelearn | treelearn_for-instance_fine_tuned_on_dev_20260712_164057 | fine_tuned_on_dev | internal_development_validation | Completed negative result; best checkpoint mean plot F1 0.4905 versus matched published baseline 0.5588; rejected before test | [`validation results`](methods/treelearn/examples/treelearn_finetune_validation_results_20260712.csv); [`result note`](methods/treelearn/docs/finetune_validation_results_20260712.md) |
| for-instance | treelearn | treelearn_for-instance_published_pretrained_20260714_134109 | published_pretrained | harmonised_pointwise_test | Completed clean authors-released baseline; test mean plot F1 0.0789, micro F1 0.0987 | [`per-plot results`](methods/treelearn/examples/treelearn_pretrained_test_plot_results_20260714.csv); [`overall results`](methods/treelearn/examples/treelearn_pretrained_test_results_20260714.csv); [`site results`](methods/treelearn/examples/treelearn_pretrained_test_site_results_20260714.csv); [`provenance`](methods/treelearn/examples/treelearn_pretrained_test_provenance_20260714.json); [`result note`](methods/treelearn/docs/pretrained_test_results_20260714.md) |
| for-instance | treelearn | treelearn_for-instance_fine_tuned_on_dev_long_20260712_233227 | fine_tuned_on_dev | harmonised_pointwise_test | Completed leakage-controlled primary result; test mean plot F1 0.3647, micro F1 0.3319 | [`per-plot results`](methods/treelearn/examples/treelearn_finetuned_test_plot_results_20260713.csv); [`overall results`](methods/treelearn/examples/treelearn_finetuned_test_results_20260713.csv); [`site results`](methods/treelearn/examples/treelearn_finetuned_test_site_results_20260713.csv); [`result note`](methods/treelearn/docs/finetuned_test_results_20260713.md) |
| for-instance | randlanet | pending | pending | harmonised_pointwise_test | Candidate accuracy benchmark | Add only with a method folder, adapter, runbook and synthetic tests. |
| for-instance | pointnetpp | pending | pending | harmonised_pointwise_test | Candidate accuracy benchmark | Add only with a method folder, adapter, runbook and synthetic tests. |
| for-instance | pointgroup | pending | pending | harmonised_pointwise_test | Candidate accuracy benchmark | Add only with a method folder, adapter, runbook and synthetic tests. |
| for-instance | softgroup | pending | pending | harmonised_pointwise_test | Candidate accuracy benchmark | Add only with a method folder, adapter, runbook and synthetic tests. |
| for-instance | hais | pending | pending | harmonised_pointwise_test | Candidate accuracy benchmark | Add only with a method folder, adapter, runbook and synthetic tests. |
| for-instance | mask3d | pending | pending | harmonised_pointwise_test | Candidate accuracy benchmark | Add only with a method folder, adapter, runbook and synthetic tests. |
| wytham-woods | tls2trees | pending | pending | coordinate_fallback_after_scene_reconstruction | Candidate TLS accuracy benchmark | [`benchmark.yml`](datasets/wytham-woods/benchmark.yml) |
| wytham-woods | segmentanytree | pending | pending | harmonised_pointwise_after_scene_reconstruction | Candidate accuracy benchmark | Plot-level input reconstruction required. |
| wytham-woods | treelearn | pending | pending | harmonised_pointwise_after_scene_reconstruction | Candidate accuracy benchmark | Plot-level input reconstruction required. |
| newfor | segmentanytree | pending | pending | pending | External comparison dataset; not implemented here | Add only through a separate documented dataset config. |

Dataset display names used in method documentation are FRDR treeiso TLS,
FOR-instance and Wytham Woods. The registry uses stable lower-case slugs so new
method rows can be sorted and compared mechanically.

The FRDR LAZ files do not contain individual-tree reference instance labels.
The TLS2trees workflow therefore preserves predictions and operational metadata
but does not report IoU/F1 without an external instance reference.

No candidate accuracy row indicates a completed method run or a reported
accuracy result. Dataset readiness and remaining preprocessing are documented
in [`docs/dataset_feasibility.md`](docs/dataset_feasibility.md).

The FOR-instance headline tracker contains five rows using the established
shared pointwise protocol and two completed TLS2trees rows using the explicit
class-3-ignore scoring domain. All use the supplied 11-plot
test split, 323 reference instances, IoU `>= 0.5` and maximum-cardinality
one-to-one matching. Protocol and mask fields remain visible, so results with
different scoring domains are not silently ranked together. TreeX has no
fine-tuning stage. TreeLearn development and checkpoint-sweep results and the
TLS2trees leaf-off targets remain in the diagnostics table. Each TLS2trees
variant retains both target prediction sets as 22 hash-verified source-row
files.

The accepted TreeLearn smoke is adapter evidence from one CULS development
plot, not a headline benchmark result. It preserved the source row count and
order across 1,816,672 points, retained all five raw and aligned prediction
artefacts and obtained F1 `0.705882` under the shared point-wise protocol. It
authorised only the frozen 21-plot development route. That route subsequently
completed with mean plot F1 `0.515571` and count-aggregated micro F1 `0.510760`.
CULS has the highest site mean F1 (`0.715010`) and NIBIO the lowest
(`0.446965`). All 105 prediction artefacts remain retained for future metrics.
The December 2024 checkpoint has documented FOR-instance validation/test
training overlap, so these values are a published-method reproduction and are
excluded from leakage-free ranking.
The separate inherited-overlap development fine-tune evaluated ten retained checkpoints on
the frozen five-plot validation subset. None exceeded the matched published
baseline, so the fine-tuned route is rejected and was not submitted to the
held-out test. Its 250 raw and aligned validation prediction artefacts remain
hash-verified on Barkla; it inherits the same checkpoint overlap. A guarded
replacement starts from the authors-released L1W-fine-tuned checkpoint whose
stated training data excludes FOR-instance, uses the fixed 16/5 development
split and freezes an epoch-35 checkpoint trained on the 16 training plots. Its
one-time 11-plot test completed with mean plot F1 `0.364685` and micro F1
`0.331924`. All 55 raw and aligned prediction artefacts remain hash-verified;
the result is directly comparable with the completed SegmentAnyTree and TreeX
test rows. The published December 2024 TreeLearn checkpoint remains a separate
overlap-affected development reproduction and is not a leakage-free test
baseline.

The unchanged clean authors-released TreeLearn checkpoint was also evaluated
once on the same frozen 11-plot test subset. It obtained mean plot F1
`0.078944` and micro F1 `0.098694` (micro precision `0.092896`, micro recall
`0.105263`, TP `34`, FP `332`, FN `289`). All 55 raw and aligned prediction
artefacts passed retention verification. An independent recomputation audit
re-hashed the files, reproduced the aggregate result and found recall `0/74`
for reference trees below 10 m and `34/249` for trees at least 10 m. This is
the genuine frozen result for that clean checkpoint and pinned pipeline; the
authors document the checkpoint as targeting trees above 10 m, so it must not
be interpreted as TreeLearn's best possible performance on small-tree data.

The permitted training mode values for completed or provisional runs are
`published_pretrained`, `fine_tuned_on_dev`, `retrained_from_dev` and
`external_training_only`. For deterministic or rule-based methods that do not
fit weights, `external_training_only` records that no FOR-instance development
or test data were used for training; the method-specific configuration should
also record the non-learning method mode.

The first SegmentAnyTree workflow completed inference for all 32 FOR-instance
LAS files with the released checkpoint. Its coordinate-rematched metrics are
provisional because they do not preserve point alignment. The completed target
comparison evaluated the released checkpoint with aligned point-wise outputs,
then fine-tuned those weights on FOR-instance development data and evaluated
the frozen selected checkpoint once. The primary fine-tuned result has mean
plot F1 `0.5447` and micro F1 `0.5320`; the released baseline has mean plot F1
`0.4534` and micro F1 `0.4442`. Their predictions and aligned evaluation files
remain in separate retained Barkla output roots for future metrics. The
completed site breakdown shows the strongest fine-tuned mean F1 on SCION
(`0.7206`) and the weakest on TUWIEN (`0.3662`); RMIT is the only site whose
mean F1 decreases relative to the released baseline.

The earlier from-scratch checkpoint is retained as historical evidence. It is
`sat_for_quicktune_to49_20260706_140730`, with mean aligned F1 `0.537` on the
five development validation plots. On the 11 held-out test plots, mean plot F1
is `0.4825` and count-aggregated micro F1 is `0.4692`; these values must not be
attributed to either completed target variant.
The later `to55` continuation is rejected because validation fell to `0.451`.
No completed target test result may be rerun for setting selection.
See the
[`shared protocol`](docs/protocols/for-instance.md),
[`runbook`](methods/segmentanytree/docs/for_instance_benchmark.md) and
[`final result`](methods/segmentanytree/docs/final_results_20260711.md).

## Adding A Benchmark

Additions should include:

1. A config named for the dataset and method.
2. A dataset inspection or conversion adapter where required.
3. A wrapper around the upstream method rather than a reimplementation.
4. Slurm jobs for inspection, preparation, prediction and summarisation.
5. Metadata recording for inputs, versions, commands, runtime and outputs.
6. Evaluation only when suitable reference labels are available.
7. Synthetic tests that do not require private or large datasets.
8. A concise runbook documenting environment, assumptions and limitations.

Raw data, external repositories, predictions, logs and large derived files must
remain outside Git.
