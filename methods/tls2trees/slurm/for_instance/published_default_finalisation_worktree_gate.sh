#!/usr/bin/env bash

# Validate repository state immediately before published-default publication.
# Recovery is intentionally limited to unstaged modifications and untracked
# files that an interrupted deterministic publication can leave behind.
tls2trees_validate_published_default_finalisation_worktree() {
  local project_root="${1:?project root is required}"
  local recovery_confirmed="${2:-0}"
  local status_output

  status_output=$(
    git -C "$project_root" status --porcelain=v1 --untracked-files=all
  ) || {
    echo "Unable to inspect the published-default finalisation worktree." >&2
    return 2
  }

  case "$recovery_confirmed" in
    0)
      test -z "$status_output" || {
        echo "Refusing published-default finalisation from a dirty worktree." >&2
        return 2
      }
      return 0
      ;;
    1)
      ;;
    *)
      echo "TLS2TREES_PUBLISHED_DEFAULT_RESULTS_RECOVERY_CONFIRMED must be 0 or 1." >&2
      return 2
      ;;
  esac

  local -a publication_targets=(
    methods/tls2trees/examples/tls2trees_published_default_test_plot_results.csv
    methods/tls2trees/examples/tls2trees_published_default_test_site_results.csv
    methods/tls2trees/examples/tls2trees_published_default_test_results.csv
    methods/tls2trees/examples/tls2trees_published_default_leaf_off_test_plot_diagnostic.csv
    methods/tls2trees/examples/tls2trees_published_default_leaf_off_test_site_diagnostic.csv
    methods/tls2trees/examples/tls2trees_published_default_leaf_off_test_diagnostic.csv
    methods/tls2trees/examples/tls2trees_published_default_prediction_retention_manifest.json
    methods/tls2trees/examples/tls2trees_published_default_test_provenance.json
    outputs/for_instance_benchmark_metrics/for_instance_method_benchmark_results.csv
    outputs/for_instance_benchmark_metrics/for_instance_method_development_diagnostics.csv
    outputs/for_instance_benchmark_metrics/for_instance_prediction_retention_registry.csv
  )
  local -a allowed_paths=("${publication_targets[@]}")
  local path directory basename
  for path in "${publication_targets[@]}"; do
    directory=${path%/*}
    basename=${path##*/}
    allowed_paths+=(
      "$directory/.$basename.tls2trees-published-default-finalisation.tmp"
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
