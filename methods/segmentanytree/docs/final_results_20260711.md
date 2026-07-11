# SegmentAnyTree FOR-instance final results

## Completion status

The SegmentAnyTree FOR-instance benchmark is complete under
`for_instance_pointwise_v1`. Both target variants were frozen before their
one-time 11-plot held-out evaluations. No test plot was used for training,
checkpoint selection or post-processing selection.

The primary result is the released-weight fine-tuned run
`segmentanytree_for-instance_fine_tuned_on_dev_20260711_002931`. It used 16
development training plots, five fixed development validation plots and 35
epochs with base learning rate `0.0001`, batch size `8`, a fresh optimiser and
fresh epoch history. Its five-plot development mean F1 was `0.5667`.

## Held-out test results

| Training mode | Run ID | Mean plot F1 | Micro F1 | Mean precision | Mean recall | TP | FP | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fine_tuned_on_dev` | `segmentanytree_for-instance_fine_tuned_on_dev_20260711_002931` | **0.5447** | **0.5320** | **0.4297** | **0.7806** | 237 | 331 | 86 |
| `retrained_from_dev` historical | `sat_for_quicktune_to49_20260706_140730` | 0.4825 | 0.4692 | 0.3807 | 0.6954 | 202 | 336 | 121 |
| `published_pretrained` | `segmentanytree_for-instance_published_pretrained_20260710_231601` | 0.4534 | 0.4442 | 0.3230 | 0.8086 | 247 | 542 | 76 |

All rows use the 11 held-out plots, IoU threshold `0.5`, the union of reference
and predicted tree points, and maximum-cardinality one-to-one matching. The
fine-tuned run is the completed primary SegmentAnyTree result. Relative to the
published checkpoint it improves mean plot F1 by `0.0913` and reduces false
positives by 211, while mean matched IoU remains high at `0.8585`.

The released checkpoint is retained as a comparison baseline. Its exact
training plot manifest was not bundled, so it must not be described as
confirmed leakage-free. The historical retrained run remains valid historical
evidence but is not the primary result. The rejected 8 July fine-tune remains a
diagnostic only because it produced zero accepted instances.

The machine-readable aggregate values are in
[`../examples/sat_completed_target_results_20260711.csv`](../examples/sat_completed_target_results_20260711.csv),
the site-level values are in
[`../examples/sat_completed_target_site_results_20260711.csv`](../examples/sat_completed_target_site_results_20260711.csv),
with run and freeze locations in
[`../examples/sat_completed_target_provenance_20260711.json`](../examples/sat_completed_target_provenance_20260711.json).

## Held-out site results

The 11 held-out plots cover five sites. Values below are site means across
plots, so NIBIO contributes six plots, SCION contributes two, and the other
sites contribute one each.

| Variant | Site | Plots | Mean F1 | Mean precision | Mean recall | Mean matched IoU | TP | FP | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `published_pretrained` | CULS | 1 | 0.4545 | 0.2941 | 1.0000 | 0.9766 | 20 | 48 | 0 |
| `published_pretrained` | NIBIO | 6 | 0.4263 | 0.2884 | 0.8381 | 0.8588 | 133 | 326 | 28 |
| `published_pretrained` | RMIT | 1 | 0.5000 | 0.4318 | 0.5938 | 0.8165 | 38 | 50 | 26 |
| `published_pretrained` | SCION | 2 | 0.5857 | 0.4429 | 0.8644 | 0.9009 | 37 | 47 | 6 |
| `published_pretrained` | TUWIEN | 1 | 0.3040 | 0.2111 | 0.5429 | 0.7157 | 19 | 71 | 16 |
| `fine_tuned_on_dev` | CULS | 1 | 0.4938 | 0.3279 | 1.0000 | 0.9875 | 20 | 41 | 0 |
| `fine_tuned_on_dev` | NIBIO | 6 | 0.5356 | 0.4043 | 0.8073 | 0.8548 | 129 | 190 | 32 |
| `fine_tuned_on_dev` | RMIT | 1 | 0.4768 | 0.4138 | 0.5625 | 0.7957 | 36 | 51 | 28 |
| `fine_tuned_on_dev` | SCION | 2 | **0.7206** | **0.5989** | 0.9044 | 0.9140 | 39 | 26 | 4 |
| `fine_tuned_on_dev` | TUWIEN | 1 | 0.3662 | 0.3611 | 0.3714 | 0.7032 | 13 | 23 | 22 |

Fine-tuning improves mean F1 on CULS (`+0.0393`), NIBIO (`+0.1093`), SCION
(`+0.1350`) and TUWIEN (`+0.0622`). RMIT decreases by `0.0232`. SCION is the
strongest fine-tuned site, while TUWIEN remains the weakest. The main change is
lower over-segmentation: false positives fall by 7 on CULS, 136 on NIBIO, 21
on SCION and 48 on TUWIEN. This comes with small recall losses on NIBIO and a
larger recall loss on TUWIEN; CULS retains full recall.

## Retained artefacts

Large artefacts remain on Barkla outside Git. Future point-wise metrics require
both aligned files for every plot:

- `Instance_results_forEval_0.ply`, containing predicted and reference instance
  labels in source-point order; and
- `semantic_segmentation_*.ply`, containing predicted and reference semantic
  labels in the same point order.

The completed target roots are:

- released predictions:
  `data/predictions/segmentanytree/for_instance_variants/segmentanytree_for-instance_published_pretrained_20260710_231601/held_out_test/`;
- fine-tuned validation predictions:
  `data/predictions/segmentanytree/for_instance_trained_validation/segmentanytree_for-instance_fine_tuned_on_dev_20260711_002931/`;
- fine-tuned test predictions:
  `data/predictions/segmentanytree/for_instance_trained_test/segmentanytree_for-instance_fine_tuned_on_dev_20260711_002931/`;
- corresponding metadata under
  `results/metadata/segmentanytree_for_instance/`;
- corresponding aggregate tables under
  `results/tables/segmentanytree_for_instance/`; and
- fine-tuned and historical checkpoints under
  `~/fastscratch/segmentanytree_for_instance_checkpoints/`.

Run the retention verifier after checkout updates or storage maintenance:

```bash
RUN_ID=segmentanytree_for-instance_fine_tuned_on_dev_20260711_002931
python methods/segmentanytree/scripts/evaluation/verify_completed_sat_retention.py \
  --output "results/metadata/segmentanytree_for_instance/retention_manifests/${RUN_ID}.json"
```

Success prints `status=retention-verified`. The generated manifest inventories
the aligned predictions, run metadata, per-plot metrics, summaries, freezes and
three accepted checkpoints. It also records the total retained SAT prediction
file count and size. Do not delete or move these roots without regenerating and
reviewing the manifest. The 11 July Barkla verification completed successfully;
its status and manifest location are recorded in the public provenance JSON.

## Closure

The target test evaluations are final and must not be repeated for setting
selection. Further metrics should be derived from the retained aligned PLY
files and existing predictions, without rerunning inference or changing the
frozen labels.

Site-level differences are generated from the retained per-plot metric JSONs,
without rerunning inference:

```bash
PRETRAINED=segmentanytree_for-instance_published_pretrained_20260710_231601
FINETUNED=segmentanytree_for-instance_fine_tuned_on_dev_20260711_002931
python methods/segmentanytree/scripts/evaluation/summarise_completed_sat_sites.py \
  --variant "published_pretrained=results/metadata/segmentanytree_for_instance/variants/${PRETRAINED}/held_out_test" \
  --variant "fine_tuned_on_dev=results/metadata/segmentanytree_for_instance/trained_test/${FINETUNED}" \
  --expected-plots 11 \
  --expected-split test \
  --output results/tables/segmentanytree_for_instance/completed_target_site_summary.csv
```

The output contains separate CULS, NIBIO, RMIT, SCION and TUWIEN rows for both
target variants. Its committed public-safe copy is
[`../examples/sat_completed_target_site_results_20260711.csv`](../examples/sat_completed_target_site_results_20260711.csv).
