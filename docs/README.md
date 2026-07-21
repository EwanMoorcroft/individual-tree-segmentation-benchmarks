# Documentation Index

Repository-wide contracts and operational guidance live here. Method-specific
runbooks, configs and result notes remain with their method under `methods/`.

## Protocols

- [`protocols/for-instance.md`](protocols/for-instance.md): fixed FOR-instance
  split, alignment, mask, matching and reporting contract.
- [`protocols/result-governance.md`](protocols/result-governance.md): controlled
  result roles, ranking eligibility, exposure, budgets, future run IDs and
  generated-output rules.
- [`protocols/method-adapter-acceptance-checklist.md`](protocols/method-adapter-acceptance-checklist.md): required gates before a method is reported as comparable.
- [`evaluation_metrics.md`](evaluation_metrics.md): metric definitions and the
  distinction between operational, diagnostic, pending and completed results.

## Operations

- [`barkla_workflow.md`](barkla_workflow.md): repository transfer, job
  submission and bounded monitoring on Barkla.
- [`slurm_resource_guide.md`](slurm_resource_guide.md): resource profiles and
  success/failure indicators.
- [`dataset_feasibility.md`](dataset_feasibility.md): dataset readiness and
  benchmark suitability.

## Status Records

- [`repository_governance_audit.md`](repository_governance_audit.md):
  repository-wide findings, actions, unresolved risks and follow-up.
- [`release_readiness.md`](release_readiness.md): commit-specific public-release
  checklist and suggested dissertation tag sequence.
- [`plans/labelled-accuracy.md`](plans/labelled-accuracy.md): historical
  SegmentAnyTree planning/status snapshot; not the current result registry.
- [`../methods/treelearn/docs/for_instance_validity_audit.md`](../methods/treelearn/docs/for_instance_validity_audit.md):
  TreeLearn coordinate, tiling, filtering, feature, alignment and checkpoint
  validity checklist.
- [`../methods/tls2trees/docs/for_instance_validity_audit.md`](../methods/tls2trees/docs/for_instance_validity_audit.md):
  TLS2trees stage-count, material/mask, attachment and failure-attribution
  validity checklist.
- [`../BENCHMARKS.md`](../BENCHMARKS.md): concise dataset-method registry.

## Templates

- [`templates/method_run_provenance.yml`](templates/method_run_provenance.yml):
  public-safe environment, upstream, checkpoint, governance and retention
  provenance template.

Raw data, predictions, checkpoints, containers and logs are intentionally not
part of this documentation tree.
