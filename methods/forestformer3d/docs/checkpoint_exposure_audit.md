# ForestFormer3D checkpoint exposure audit

Protocol: `forestformer3d_checkpoint_exposure_v1`.

The released checkpoint was trained on FOR-instanceV2. Eligibility therefore
depends on exact plot identities, not on the later release's train/test labels
alone.

The audit reads the repository's frozen paths from
`shared/for_instance_manifest.py` and a filename-only snapshot of the official
Zenodo `train_val_data.zip` and `test_data.zip` previews. It does not download
either point-cloud archive.

The only accepted mapping rule is:

```text
<collection>/<stem>.las
  -> <collection>_<collection>_<stem>_<official-role>.ply
```

No fuzzy, case-insensitive or region-only matching is used. Each original path
must match exactly one `train`, `val` or `test` member.

The reviewed 32-row output records:

- 21 development paths, of which 13 map to official training and eight to
  official validation;
- 11 held-out paths, all mapping to official test;
- zero held-out paths mapping to training or validation;
- zero unmatched or multiply matched paths.

The inventory snapshot SHA-256 is
`348199b3818381b949b8a335ecd789fc41f9247b1086f2a8e82f1cfbb0357138`.
The table was manually reviewed on 2026-07-23.

Decision: `eligible_exposure_gate_passed`.

This decision applies to checkpoint exposure only. It does not authorize
held-out execution and does not establish source-row or inference validity.

To regenerate into new, non-existing output paths:

```bash
python methods/forestformer3d/scripts/data/audit_checkpoint_exposure.py \
  --inventory methods/forestformer3d/examples/zenodo_v2_archive_inventory_20260723.csv \
  --output-csv /new/path/checkpoint_exposure.csv \
  --output-summary /new/path/checkpoint_exposure.json
```
