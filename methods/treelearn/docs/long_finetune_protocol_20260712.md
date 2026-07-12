# TreeLearn clean-checkpoint long fine-tuning protocol

## Purpose

This development-only route prepares a leakage-controlled TreeLearn candidate
for the same FOR-instance pointwise benchmark used by SegmentAnyTree and
TreeX. It does not access or submit the held-out test split.

## Checkpoint provenance

The upstream default `model_weights_20241213.pth` is retained as a published
method reproduction with documented FOR-instance validation/test training
overlap. The long route instead uses the authors-released
`model_weights_finetuned.pth` checkpoint, persistent ID
`doi:10.25625/VPMPID/8CIIW0`, MD5
`106a80de2991c5f23484a3f9d03e3b16`. The authors describe it as noisy-label
pretraining followed by fine-tuning on their L1W benchmark.

## Frozen data and budget

- exact relative-path mapping from the supplied `data_split_metadata.csv`;
- full supplied catalogue locked to 56 development and 26 test entries;
- local 32-plot benchmark locked to the exact-path 21-development/11-test
  subset used by every completed method;
- split-metadata SHA-256 rechecked at preparation and every downstream stage;
- held-out test point-cloud files are not opened by this route;
- seed-42 split: 16 tuning-training and five validation plots;
- 32 deterministic, SHA-256-inventoried crops per training plot;
- tuning view: 512 crops from the 16 training plots, matching the completed
  short route's proven total crop-bank size while balancing plots exactly;
- 48 deterministic generation attempts per plot absorb upstream rejection of
  invalid chunks; the lexicographically first 32 valid NPZ files are retained
  and every discarded filename is recorded in the crop inventory;
- 35 epochs, 714 examples per epoch, batch size 2;
- 24,990 examples and 12,495 optimizer steps per trial;
- checkpoints at epochs 7, 14, 21, 28 and 35.

The supplied metadata defines the `dev`/`test` boundary but does not define
training and validation roles inside `dev`. Every one of the 21 locally
available development paths must be an exact member of the 56 metadata rows
marked `dev`. The seed-42 16/5 subdivision is made only within those 21 paths;
no metadata row marked `test` can enter tuning or validation.

The 512 stored crops are resampled across epochs with the frozen TreeLearn
augmentations; training exposure remains 24,990 examples and 12,495 optimizer
steps. The 35-epoch headline matches the completed SegmentAnyTree fine-tune. Epochs
are not treated as equal compute across architectures, so examples, batch size
and optimizer steps are also retained. TreeX has no optimizer or epochs.

Some FOR-instance LAS files store integer-valued `treeID` labels in a floating
extra dimension. Normalisation accepts numeric storage only after checking that
every value is finite and integral and that the mapped labels round-trip
losslessly through the original LAS dtype.

## Search and selection

One frozen full-model configuration uses learning rate `1e-5`. It runs with
eight seeds so all eight L40S GPUs can be used and seed variability can be
reported. The comparable candidate is preregistered as seed 42 at epoch 35;
the other seven seeds cannot select it. Each of five checkpoints is evaluated
on all five fixed validation plots for learning-curve diagnostics. Five
additional tasks evaluate the unchanged clean checkpoint on the same
validation plots. Earlier checkpoints cannot select the candidate.

Each trial records Python, NumPy, PyTorch, CUDA, cuDNN and spconv versions,
sorted input order, RNG seed and deterministic cuDNN settings. Bitwise identity
is not claimed because pinned sparse CUDA kernels may remain nondeterministic.

Crop generation is CPU-only and therefore uses the `nodes` partition rather
than reserving idle GPUs. Training and validation use `%8` arrays, allowing all
eight L40S GPUs to be occupied when scheduler capacity is available.
Validation is submitted as 41 checkpoint-level array tasks; each GPU task runs
the same five validation plots sequentially. This preserves the complete
eight-seed, five-epoch and clean-baseline matrix while staying below Barkla's
per-user submitted-job limit.

Every validation plot retains five raw/adapted prediction artefacts. Selection
requires exactly 1,025 unique files and verifies size and SHA-256. The final
candidate is the fixed configuration's seed-42 epoch-35 checkpoint, trained only
on the fixed 16 training plots. No all-development refit is performed.

The final gate copies the selected checkpoint, environment records, frozen
evaluation/training/crop configs, per-plot crop inventories and selection
tables from fastscratch into ignored run-scoped project scratch paths. Every
retained control file is size- and SHA-256-inventoried.

## Evaluation contract

Future primary comparison uses all 11 locally available held-out test plots under
`for_instance_pointwise_v1`: source-row-aligned labels, union evaluation mask,
IoU threshold 0.5 and maximum-cardinality one-to-one matching. Site summaries
remain separate for CULS, NIBIO, RMIT, SCION and TUWIEN. Development scores
cannot be pooled with test scores.

The long chain ends at a hashed selected checkpoint with
`held_out_test_accessed=false`. A separate manual authorization is required
before one frozen evaluation of the clean unchanged checkpoint and one frozen
evaluation of the selected fine-tuned checkpoint on all 11 test plots.
