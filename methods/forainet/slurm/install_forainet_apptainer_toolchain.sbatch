#!/usr/bin/env bash
#SBATCH --job-name=forai_tool
#SBATCH --partition=nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/forai_tool_%j.out
#SBATCH --error=logs/forai_tool_%j.err

set -euo pipefail

: "${FORAINET_BENCHMARK_ROOT:?set FORAINET_BENCHMARK_ROOT}"
: "${FORAINET_TOOLCHAIN_ROOT:?set FORAINET_TOOLCHAIN_ROOT}"
: "${FORAINET_TOOLCHAIN_EVIDENCE_ROOT:?set FORAINET_TOOLCHAIN_EVIDENCE_ROOT}"
: "${FORAINET_EXPECTED_BENCHMARK_COMMIT:?set FORAINET_EXPECTED_BENCHMARK_COMMIT}"
: "${APPTAINER_CACHEDIR:?set APPTAINER_CACHEDIR}"

installer_url="https://raw.githubusercontent.com/apptainer/apptainer/v1.3.6/tools/install-unprivileged.sh"
installer_sha256="41574717e85e03cdf40597819c927250d0772186b943b8869c8ec8dfcb5b86d1"
release_rpm_url="https://github.com/apptainer/apptainer/releases/download/v1.3.6/apptainer-1.3.6-1.x86_64.rpm"
release_rpm_sha256="1890dd3df87b06b0a9b2845b81b5709c0033fcca5673b03cc69ce9cb755e9605"
probe_definition="$FORAINET_BENCHMARK_ROOT/methods/forainet/containers/fakeroot-apt-probe.def"

test ! -e "$FORAINET_TOOLCHAIN_ROOT"
test ! -e "$FORAINET_TOOLCHAIN_EVIDENCE_ROOT"
test -f "$probe_definition"
test "$(git -C "$FORAINET_BENCHMARK_ROOT" rev-parse HEAD)" = \
  "$FORAINET_EXPECTED_BENCHMARK_COMMIT"
test -z "$(git -C "$FORAINET_BENCHMARK_ROOT" status --porcelain)"
for required_command in curl rpm2cpio cpio
do
  command -v "$required_command" >/dev/null
done

mkdir -p "$FORAINET_TOOLCHAIN_EVIDENCE_ROOT" "$APPTAINER_CACHEDIR"
installer="$FORAINET_TOOLCHAIN_EVIDENCE_ROOT/install-unprivileged-v1.3.6.sh"
curl --fail --location --silent --show-error "$installer_url" --output "$installer"
printf '%s  %s\n' "$installer_sha256" "$installer" | sha256sum --check

release_rpm="$FORAINET_TOOLCHAIN_EVIDENCE_ROOT/apptainer-1.3.6-1.x86_64.rpm"
curl --fail --location --silent --show-error "$release_rpm_url" --output "$release_rpm"
printf '%s  %s\n' "$release_rpm_sha256" "$release_rpm" | sha256sum --check

bash "$installer" -e -d el8 -v "$release_rpm" "$FORAINET_TOOLCHAIN_ROOT" \
  > "$FORAINET_TOOLCHAIN_EVIDENCE_ROOT/install.stdout" \
  2> "$FORAINET_TOOLCHAIN_EVIDENCE_ROOT/install.stderr"

apptainer="$FORAINET_TOOLCHAIN_ROOT/bin/apptainer"
test -x "$apptainer"
test -x "$FORAINET_TOOLCHAIN_ROOT/x86_64/utils/bin/fakeroot-sysv"
apptainer_version="$("$apptainer" --version)"
test "$apptainer_version" = "apptainer version 1.3.6-1"
printf '%s\n' "$apptainer_version" \
  > "$FORAINET_TOOLCHAIN_EVIDENCE_ROOT/apptainer_version.txt"

node_tmp="$(mktemp -d "/tmp/forai-tool-${SLURM_JOB_ID}-XXXXXXXX")"
cleanup_node_tmp() {
  case "$node_tmp" in
    "/tmp/forai-tool-${SLURM_JOB_ID}-"*) rm -rf -- "$node_tmp" ;;
    *) return 1 ;;
  esac
}
trap cleanup_node_tmp EXIT
export APPTAINER_TMPDIR="$node_tmp"

probe_image="$FORAINET_TOOLCHAIN_EVIDENCE_ROOT/fakeroot-apt-probe.sif"
cd "$FORAINET_BENCHMARK_ROOT"
"$apptainer" build --fakeroot "$probe_image" "$probe_definition" \
  > "$FORAINET_TOOLCHAIN_EVIDENCE_ROOT/probe.stdout" \
  2> "$FORAINET_TOOLCHAIN_EVIDENCE_ROOT/probe.stderr"
"$apptainer" test "$probe_image"
sha256sum "$probe_image" > "$FORAINET_TOOLCHAIN_EVIDENCE_ROOT/probe.sha256"
sha256sum "$probe_definition" > "$FORAINET_TOOLCHAIN_EVIDENCE_ROOT/probe_definition.sha256"

find "$FORAINET_TOOLCHAIN_ROOT" -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$FORAINET_TOOLCHAIN_EVIDENCE_ROOT/toolchain_files.sha256"
printf '%s\n' complete > "$FORAINET_TOOLCHAIN_EVIDENCE_ROOT/status.txt"
