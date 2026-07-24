# ForestFormer3D results

## Published checkpoint on development

The unchanged official checkpoint completed all 21 original development plots
in run
`forestformer3d__for-instance__published-pretrained__not-applicable__development__20260723T221726`.
This is a diagnostic result, not a held-out ranking row.

| Plots | Points | TP | FP | FN | Precision | Recall | Micro F1 | Mean plot F1 | Matched IoU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 21 | 101,769,037 | 740 | 895 | 67 | 0.452599 | 0.916976 | 0.606061 | 0.594511 | 0.911754 |

The large precision-recall imbalance shows that the published checkpoint
generally detects reference trees but over-segments them on this development
subset. NIBIO is the main precision weakness. The result is suitable evidence
for the frozen development-only fine-tuning objective, but it says nothing
about held-out performance.

All 294 required retained artefacts (16,565,931,840 bytes) passed independent
SHA-256 verification. The verifier also loaded every harmonised prediction,
proved exact source-row identity, reconciled all per-plot counts and confirmed
`held_out_access=false`.

No ForestFormer3D held-out accuracy result exists.
