# ForestFormer3D public examples

This directory contains public-safe metadata and schemas only. It must not
contain point coordinates, point clouds, checkpoints, containers, raw
predictions, full logs or private machine paths.

`zenodo_v2_archive_inventory_20260723.csv` is a filename-only snapshot of the
official Zenodo archive previews. It includes all PLY member names visible in
the two relevant archives; it does not contain archive payloads.

`checkpoint_exposure_audit_20260723.csv` is the exact 32-row mapping from the
repository's frozen original FOR-instance paths to that inventory.
`checkpoint_exposure_summary_20260723.json` records the inventory hash and the
eligibility decision.
