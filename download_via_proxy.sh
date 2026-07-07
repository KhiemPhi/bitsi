#!/usr/bin/env bash
# download_via_proxy.sh — fetch datasets on this egress-restricted server.
#
# Meta's fwdproxy enforces a destination ALLOWLIST. Probed 2026-07-07:
#   ALLOW : pypi.org, files.pythonhosted.org, huggingface.co (API + small inline files)
#   BLOCK : github.com, raw.githubusercontent.com, shapenet.cs.stanford.edu       (403)
#   BLOCK : cdn-lfs.huggingface.co, cas-bridge.xethub.hf.co  (HF large-file CDN)  (403/502)
#
# CONSEQUENCE: small HF files (JSON/metadata) download fine, but LARGE LFS/Xet
# blobs (e.g. the 83 MB ShapeNetPart .h5 files) STALL at 0 bytes because their
# CDN host is blocked. To actually pull big datasets on this server you must
# EITHER (a) get cdn-lfs.huggingface.co + cas-bridge.xethub.hf.co allowlisted via
# Meta's egress process, (b) use an internal mirror, or (c) download elsewhere
# and rsync into $BITSI_DATA_ROOT. This script is ready for path (a): once the
# CDN is allowlisted it works unchanged.
#
# Usage:
#   ./download_via_proxy.sh                     # download the core set (shapenetpart)
#   ./download_via_proxy.sh shapenetpart        # a specific dataset
#   ./download_via_proxy.sh shapenetpart modelnet40
#   ./download_via_proxy.sh --list              # list known datasets and exit
#
# Overrides:
#   PROXY=host:port  HF_BIN=/path/to/hf  BITSI_DATA_ROOT=/data  HF_TOKEN=hf_xxx
set -euo pipefail

PROXY="${PROXY:-fwdproxy:8080}"
ROOT="${BITSI_DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/datasets}"
HF_BIN="${HF_BIN:-$HOME/.venvs/bitsi_sim/bin/hf}"

# repo  repo_type  dest-subdir  include-glob("*"=all)   -- HF mirrors, allowlisted
declare -A REPO=(
  [shapenetpart]="larryshaw0079/ShapeNetPart"
  [modelnet40]="jxie/modelnet40"
)
declare -A KIND=( [shapenetpart]=dataset [modelnet40]=dataset )
declare -A DEST=( [shapenetpart]="P4/shapenetpart" [modelnet40]="P4/modelnet40" )
declare -A INCL=( [shapenetpart]="*"          [modelnet40]="*" )
CORE=(shapenetpart)   # default set when no args given

export http_proxy="http://${PROXY}"  https_proxy="http://${PROXY}"
export HTTP_PROXY="http://${PROXY}"  HTTPS_PROXY="http://${PROXY}"
export no_proxy="${no_proxy:-.intern.facebook.com,.fbinfra.net,127.0.0.1,localhost,::1}"
export HF_HUB_DISABLE_TELEMETRY=1

if [ "${1:-}" = "--list" ]; then
  printf "%-14s %-9s %-28s %s\n" KEY TYPE REPO DEST
  for k in "${!REPO[@]}"; do printf "%-14s %-9s %-28s %s\n" "$k" "${KIND[$k]}" "${REPO[$k]}" "$ROOT/${DEST[$k]}"; done
  exit 0
fi

[ -x "$HF_BIN" ] || { echo "ERROR: hf CLI not found at $HF_BIN" >&2
  echo "  install: https_proxy=http://${PROXY} $HOME/.venvs/bitsi_sim/bin/pip install huggingface_hub" >&2; exit 1; }

TARGETS=("$@"); [ ${#TARGETS[@]} -eq 0 ] && TARGETS=("${CORE[@]}")

echo "== dataset download via proxy ${PROXY} =="
echo "   hf   : $HF_BIN"
echo "   root : $ROOT"
[ -n "${HF_TOKEN:-}" ] && TOKEN_ARG=(--token "$HF_TOKEN") || TOKEN_ARG=()

for key in "${TARGETS[@]}"; do
  repo="${REPO[$key]:-}"
  [ -z "$repo" ] && { echo "  [skip] unknown dataset '$key' (see --list)"; continue; }
  dest="$ROOT/${DEST[$key]}"
  echo; echo ">> $key  <-  $repo  (${KIND[$key]})  ->  $dest"
  mkdir -p "$dest"
  "$HF_BIN" download "$repo" --repo-type "${KIND[$key]}" \
    --include "${INCL[$key]}" --local-dir "$dest" "${TOKEN_ARG[@]}"
done

echo; echo "Done. Downloaded: ${TARGETS[*]}"
