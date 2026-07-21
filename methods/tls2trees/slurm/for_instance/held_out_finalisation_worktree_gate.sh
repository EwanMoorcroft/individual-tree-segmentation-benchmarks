#!/usr/bin/env bash

# Validate the repository state immediately before tuned held-out publication.
# Normal publication requires a completely clean worktree.  Recovery is an
# explicit, narrow exception for files that the finaliser itself can leave after
# an interrupted multi-file commit; the Python finaliser still validates and
# deterministically regenerates every permitted public output.
tls2trees_validate_held_out_finalisation_worktree() {
  local project_root="${1:?project root is required}"
  local recovery_confirmed="${2:-0}"
  local status_output

  status_output=$(
    git -C "$project_root" status --porcelain=v1 --untracked-files=all
  ) || {
    echo "Unable to inspect the finalisation worktree." >&2
    return 2
  }

  case "$recovery_confirmed" in
    0)
      test -z "$status_output" || {
        echo "Refusing result finalisation from a dirty worktree." >&2
        echo "Use recovery confirmation only for an interrupted tuned finalisation." >&2
        return 2
      }
      return 0
      ;;
    1)
      ;;
    *)
      echo "TLS2TREES_FINALIZE_RESULTS_RECOVERY_CONFIRMED must be 0 or 1." >&2
      return 2
      ;;
  esac

  local -a public_paths=(
    methods/tls2trees/examples/tls2trees_development_tuned_test_plot_results.csv
    methods/tls2trees/examples/tls2trees_development_tuned_test_site_results.csv
    methods/tls2trees/examples/tls2trees_development_tuned_test_results.csv
    methods/tls2trees/examples/tls2trees_development_tuned_leaf_off_test_plot_diagnostic.csv
    methods/tls2trees/examples/tls2trees_development_tuned_leaf_off_test_site_diagnostic.csv
    methods/tls2trees/examples/tls2trees_development_tuned_leaf_off_test_diagnostic.csv
    methods/tls2trees/examples/tls2trees_development_tuned_test_provenance.json
  )
  local -a registry_paths=(
    outputs/for_instance_benchmark_metrics/for_instance_method_benchmark_results.csv
    outputs/for_instance_benchmark_metrics/for_instance_method_development_diagnostics.csv
    outputs/for_instance_benchmark_metrics/for_instance_prediction_retention_registry.csv
  )
  local -a finaliser_write_paths=(
    "${public_paths[@]}"
    "${registry_paths[0]}"
    "${registry_paths[1]}"
    "${registry_paths[2]}"
  )
  local -a allowed_paths=()
  local path directory basename
  for path in "${public_paths[@]}" "${registry_paths[@]}"; do
    allowed_paths+=("$path")
  done
  for path in "${finaliser_write_paths[@]}"; do
    directory=${path%/*}
    basename=${path##*/}
    allowed_paths+=(
      "$directory/.$basename.tls2trees-held-out-finalisation.tmp"
    )
  done

  local entry status changed_path candidate permitted
  while IFS= read -r entry; do
    [[ -n "$entry" ]] || continue
    status=${entry:0:2}
    changed_path=${entry:3}
    if [[ "$status" != " M" && "$status" != "??" ]]; then
      echo "Recovery rejects Git status '$status' for: $changed_path" >&2
      return 2
    fi
    permitted=0
    for candidate in "${allowed_paths[@]}"; do
      if [[ "$changed_path" == "$candidate" ]]; then
        permitted=1
        break
      fi
    done
    if [[ "$permitted" != "1" ]]; then
      echo "Recovery rejects unrelated worktree path: $changed_path" >&2
      return 2
    fi
    if [[ -L "$project_root/$changed_path" ]]; then
      echo "Recovery rejects symbolic link at publication path: $changed_path" >&2
      return 2
    fi
  done <<< "$status_output"
  return 0
}
