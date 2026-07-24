# Development-only fine-tuning protocol

## Scope and prerequisites

This protocol starts from the byte-identical official
`epoch_3000_fix.pth` checkpoint and can read only the 21 original FOR-instance
development plots. It does not read, infer on, visualise or score a held-out
plot. Preparation requires the independent verification record for the
completed published-pretrained development run, including all retained hashes
and exact source-row alignment.

The repository's canonical split algorithm samples five indices from the
ordered 21-plot development manifest using
`random.Random(42).sample(range(21), 5)`. The frozen validation indices are
`0, 3, 7, 8, 20`, corresponding to:

- `CULS/plot_1_annotated.las`;
- `NIBIO/plot_11_annotated.las`;
- `NIBIO/plot_19_annotated.las`;
- `NIBIO/plot_21_annotated.las`; and
- `TUWIEN/train.las`.

The remaining 16 plots are the only weight-update inputs. Their 73,099,755
source rows are staged separately from the five validation plots' 28,669,282
rows. The training loader receives one manifest record per training plot;
each access applies the unchanged official random 16 m cylinder crop, 0.2 m
grid sample, maximum 640,000-point sample and official augmentations.

## Frozen model and optimisation design

The architecture, 16 m cylinder radius, 300 queries, 0.2 m voxel size, losses,
merge behaviour and official training pipelines are unchanged. Original
FOR-instance classes 4, 5 and 6 remain mapped to the official internal wood
class; other points are loader-required ground. No reference data is available
to the later inference process.

The one permitted configuration is:

- initialization: official checkpoint SHA-256
  `01037a648596832238ac72ea2f5eef87ceaf5aeb399e56ff4b760ba1ed1c777e`;
- load mode: `load_from` with `resume=false`, so weights are loaded but the
  released epoch and optimizer state are not resumed;
- model preparation stage: `prepare_epoch=-1`, using the upstream-exposed
  setting so the already-trained query-score path remains active from the
  first fine-tuning step;
- optimizer: AdamW, learning rate `1e-5`, weight decay `0.05`, gradient norm
  limit 10;
- schedule: PolyLR, power `0.9`, over exactly 560 data-loader iterations
  corresponding to 280 optimizer steps;
- precision: official float32;
- seed: 42;
- 35 epochs, 16 examples per epoch, micro-batch size 1, accumulation 2,
  effective batch size 2, 560 examples and 280 optimizer steps; and
- checkpoints after epochs 7, 14, 21, 28 and 35, retaining optimizer state.

The lower learning rate is the sole optimization change from the upstream
`1e-4` default. It is a conservative single-configuration adaptation from an
already converged checkpoint. Gradient accumulation preserves the frozen
effective batch size after a real batch of two exceeded the 80 GiB A100 on an
instance-dense crop; it is a resource implementation, not a tuned
configuration. Thirty-five epochs follow the repository's current
development-fine-tune precedent, but epochs are not treated as equal across
methods; the exact example and optimizer-step budget above is the auditable
exposure.

Before the 35-epoch job, a technical smoke must prove that the official
checkpoint loads tensor-for-tensor through the official training Runner with
epoch and iteration counters at zero. It then performs exactly one optimizer
step on the first frozen training plot. The smoke output cannot be selected
and cannot change the scientific schedule; it informs only the runtime and
resource estimate.

## Validation and selection

Each of the five frozen checkpoints is evaluated once on all five frozen
development-validation plots through the same full-plot adapter and
`for_instance_pointwise_v1` evaluator as the published checkpoint. This gives
25 validation evaluations. The matched published baseline is extracted from
the already verified 21-plot development run for the same five paths.

Selection maximises arithmetic mean plot F1. Ties are resolved first by higher
micro F1 from summed TP, FP and FN, then by the earlier checkpoint epoch. No
threshold, architecture, merge rule, learning rate, seed or epoch set may be
changed after validation begins. Every checkpoint remains reported even when
the selected fine-tune underperforms the matched published baseline.

The selected checkpoint, exact training and validation manifests, effective
configuration, checkpoint table, environment identities and retained
prediction hashes must be frozen before a held-out readiness record can be
prepared. Held-out execution still requires separate explicit authorisation
for both variants.

## Execution stages

1. `submit_finetune_preparation.sh` copies the independently verified
   development inputs, freezes the 16/5 manifest and builds effective configs.
2. `submit_finetune_initialization_smoke.sh` proves exact Runner loading and
   performs one non-selectable optimizer step.
3. `submit_finetune_training.sh` executes the frozen 35-epoch job and
   inventories epochs 7, 14, 21, 28 and 35.
4. The validation and selection workflow evaluates only the five frozen
   validation plots and produces the selected-checkpoint freeze.

Every stage has a unique non-overwriting root or marker, uses the method
monitor, reports queue waiting separately from runtime and leaves
`held_out_access=false`.
