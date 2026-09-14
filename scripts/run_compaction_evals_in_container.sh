#!/usr/bin/env bash
# SPEC-0044 AC-9 — run the compaction soak and degradation drills inside a
# containerized Hermes, and stamp every receipt with the container's identity.
#
# Wrapper AGENTS.md rule 8: all testing of Hermes code changes happens in a
# container (containerized Hermes, temp HERMES_HOME, real imports) BEFORE any
# change reaches fork-dev/development or live. This is the documented command the
# AC-9 receipts name.
#
# Usage (from the repo root, or anywhere with an explicit --repo):
#   scripts/run_compaction_evals_in_container.sh [--repo DIR] [--turns N]
#                                                [--runtime docker|podman]
#                                                [--image IMG] [--results DIR]
#                                                [--allow-dirty]
#
# What it does:
#   1. resolves the runtime (docker, else podman) and the image;
#   2. copies the repo INTO a scratch dir (the source tree is never mounted
#      read-write, and the evals never read the live tree in place);
#   3. runs `soak.py` and the three drills inside the container with a TEMP
#      HERMES_HOME, exporting HERMES_EVAL_CONTAINER_* so each receipt identifies
#      the image id/digest and the source commit;
#   4. copies the receipts back into RESULTS, then asserts each one actually
#      carries container provenance — a bare host receipt fails this script.
#
# The receipts land in evals/compaction/results/ by default and are committed.

set -euo pipefail

RUNTIME="${CONTAINER_RUNTIME:-}"
IMAGE="${HERMES_COMPACTION_EVAL_IMAGE:-nousresearch/hermes-agent:latest}"
REPO=""
TURNS=120
RESULTS=""
ALLOW_DIRTY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --turns) TURNS="$2"; shift 2 ;;
    --runtime) RUNTIME="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --results) RESULTS="$2"; shift 2 ;;
    --allow-dirty) ALLOW_DIRTY=1; shift ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ -z "$RUNTIME" ]; then
  if command -v docker >/dev/null 2>&1; then RUNTIME=docker
  elif command -v podman >/dev/null 2>&1; then RUNTIME=podman
  else echo "no container runtime found (docker or podman)" >&2; exit 2
  fi
fi

if [ -z "$REPO" ]; then
  REPO="$(git rev-parse --show-toplevel)"
fi
REPO="$(cd "$REPO" && pwd)"
[ -d "$REPO/evals/compaction" ] || { echo "not a hermes repo: $REPO" >&2; exit 2; }
if [ -z "$RESULTS" ]; then RESULTS="$REPO/evals/compaction/results"; fi
mkdir -p "$RESULTS"

COMMIT="$(git -C "$REPO" rev-parse HEAD)"
BRANCH="$(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
DIRTY="$(git -C "$REPO" status --porcelain | head -c 200)"
if [ -n "$DIRTY" ] && [ "$ALLOW_DIRTY" -ne 1 ]; then
  echo "refusing to receipt a dirty tree (pass --allow-dirty for local iteration):" >&2
  echo "$DIRTY" >&2
  exit 2
fi

# Image identity: id + digest, resolved BEFORE the run so the receipt is pinned
# to what actually executed.
IMAGE_ID="$($RUNTIME image inspect --format '{{.Id}}' "$IMAGE")"
IMAGE_DIGEST="$($RUNTIME image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$IMAGE" \
  | head -n1 | tr -d '\n')"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/spec0044-eval-XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

echo "runtime=$RUNTIME image=$IMAGE commit=$COMMIT branch=$BRANCH"
echo "workspace=$WORK"

# Copy the tree into the scratch dir: the container reads a COPY, never the live
# tree in place.
mkdir -p "$WORK/src"
tar -C "$REPO" -cf - \
  --exclude='.git' --exclude='node_modules' --exclude='__pycache__' \
  --exclude='.worktrees' --exclude='*.egg-info' \
  . | tar -C "$WORK/src" -xf -

CONTAINER_ENV=(
  -e HERMES_HOME=/tmp/hermes-home
  # /src must precede the image's own /opt/hermes install (which sits on
  # sys.path via a .pth file), so the code UNDER TEST is the copied tree.
  -e PYTHONPATH=/src
  -e "HERMES_EVAL_CONTAINER_IMAGE=$IMAGE"
  -e "HERMES_EVAL_CONTAINER_IMAGE_ID=$IMAGE_ID"
  -e "HERMES_EVAL_CONTAINER_IMAGE_DIGEST=$IMAGE_DIGEST"
  -e "HERMES_EVAL_CONTAINER_COMMIT=$COMMIT"
  -e "HERMES_EVAL_CONTAINER_RUNTIME=$RUNTIME"
  -e "PYTHONDONTWRITEBYTECODE=1"
)

run_in_container() {
  # $1 = receipt path inside the container; remaining = the python command
  local receipt="$1"; shift
  "$RUNTIME" run --rm \
    -v "$WORK/src:/src:ro" \
    -v "$WORK/out:/out" \
    -w /src \
    --entrypoint sh \
    "${CONTAINER_ENV[@]}" \
    -e "HERMES_EVAL_CONTAINER_COMMAND=$*" \
    "$IMAGE" -c "mkdir -p /out \"\$(dirname '$receipt')\" && \"\$@\"" sh "$@"
}

mkdir -p "$WORK/out"

echo "=== soak (turns=$TURNS) ==="
run_in_container /out/soak_results.json python3 evals/compaction/soak.py \
  --turns "$TURNS" --json /out/soak_results.json

for drill in aux-down storage-ro kill-mid; do
  echo "=== drill $drill ==="
  run_in_container "/out/drill_${drill}.json" python3 evals/compaction/drills.py \
    "$drill" --json "/out/drill_${drill}.json"
done

# Verify container provenance BEFORE copying anything into the repo, so a bare
# host receipt can never land in results/.
python3 - "$WORK/out" <<'PY'
import json, sys, pathlib
out = pathlib.Path(sys.argv[1])
files = [out / "soak_results.json", out / "drill_aux-down.json",
         out / "drill_storage-ro.json", out / "drill_kill-mid.json"]
bad = []
for f in files:
    if not f.is_file():
        bad.append(f"{f.name}: missing")
        continue
    doc = json.loads(f.read_text())
    c = doc.get("container")
    if not isinstance(c, dict) or not c.get("container_image"):
        bad.append(f"{f.name}: no container provenance in receipt")
if bad:
    print("AC-9 FAIL: receipts lack container provenance:", *bad, sep="\n  ")
    sys.exit(1)
print("AC-9 provenance check: all four receipts identify the container image/commit")
PY

cp "$WORK/out/soak_results.json"          "$RESULTS/soak_results.json"
cp "$WORK/out/drill_aux-down.json"        "$RESULTS/drill_aux-down.json"
cp "$WORK/out/drill_storage-ro.json"      "$RESULTS/drill_storage-ro.json"
cp "$WORK/out/drill_kill-mid.json"        "$RESULTS/drill_kill-mid.json"

echo "receipts written to $RESULTS (commit $COMMIT)"
