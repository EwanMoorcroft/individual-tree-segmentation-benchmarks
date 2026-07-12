# TreeLearn development fine-tuning validation

Run `treelearn_for-instance_fine_tuned_on_dev_20260712_164057` started from
the published TreeLearn checkpoint and used only the frozen FOR-instance
development split. A seed-42 partition assigned 16 plots to training and five
to internal validation. The held-out test split was not accessed.

The fixed epoch-100 checkpoint obtained mean plot F1 `0.462298` and micro F1
`0.421384` on the five validation plots. A retained checkpoint sweep evaluated
epochs 10 through 100 on the same plots. Epoch 70 had the highest mean plot F1
(`0.490504`); epoch 10 had the highest fine-tuned micro F1 (`0.427035`). No
fine-tuned checkpoint exceeded the matched published-checkpoint baseline of
mean plot F1 `0.558769` and micro F1 `0.547529`.

The fine-tuning route is therefore a completed negative result and is rejected
for held-out testing. The published checkpoint remains the selected TreeLearn
variant. Every checkpoint evaluation retained and hash-verified the five raw
or aligned prediction artefacts for each validation plot: 25 artefacts per
checkpoint and at least 250 across the ten evaluated checkpoints. These files
remain outside Git in run-scoped Barkla paths.

Public-safe evidence is stored in
[`treelearn_finetune_validation_results_20260712.csv`](../examples/treelearn_finetune_validation_results_20260712.csv)
and
[`treelearn_finetune_validation_provenance_20260712.json`](../examples/treelearn_finetune_validation_provenance_20260712.json).
