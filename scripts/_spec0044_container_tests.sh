#!/usr/bin/env bash
# SPEC-0044 test-plan item 1: run the modified test files inside the
# containerized Hermes image (temp HERMES_HOME), per wrapper AGENTS.md rule 8.
#
# The image ships the app venv WITHOUT pytest, and its interpreter is 3.13 while
# the dev venv is 3.11 — so we stage the PURE-PYTHON test deps on the host with
# the dev interpreter and mount them in. The tree is mounted READ-ONLY at /src and
# PYTHONPATH puts it ahead of the image's own /opt/hermes install, so the code
# under test is the mounted copy, not the image's baked-in code.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${HERMES_EVAL_IMAGE:-nousresearch/hermes-agent:latest}"
# Interpreter used ONLY to stage pytest into $DEPS_HOST; the container runs its
# own python3. Default is python3 from PATH — point HERMES_EVAL_PYTHON at a
# virtualenv interpreter if the system python3 lacks pip or cannot reach PyPI.
PYTHON="${HERMES_EVAL_PYTHON:-python3}"
DEPS_HOST="${HERMES_EVAL_DEPS:-/tmp/spec0044-testdeps}"

FILES="$*"
if [ -z "$FILES" ]; then
  FILES="tests/agent/test_compaction_dump.py tests/agent/test_compaction_extract.py \
tests/agent/test_compaction_producer.py tests/agent/test_compaction_backstop_wiring.py \
tests/agent/test_compaction_map.py tests/agent/test_compaction_swap.py \
tests/agent/test_compaction_rehydrate.py tests/agent/test_compaction_rehydrate_binding.py \
tests/agent/test_compaction_pipeline_lock.py tests/agent/test_compaction_anti_thrash.py"
fi

# Stage pytest + its pure-python deps for the container interpreter.
if [ ! -d "$DEPS_HOST/pytest" ]; then
  echo "staging test deps into $DEPS_HOST"
  "$PYTHON" -m pip install --quiet --target "$DEPS_HOST" "pytest<9" >/dev/null
fi
[ -d "$DEPS_HOST/pytest" ] || { echo "failed to stage pytest" >&2; exit 3; }

echo "image=$IMAGE"
echo "files=$(echo $FILES | wc -w)"

docker run --rm \
  -v "$REPO":/src:ro \
  -v "$DEPS_HOST":/testdeps:ro \
  -w /src \
  --entrypoint sh \
  -e HERMES_HOME=/tmp/hermes-home \
  -e PYTHONPATH=/testdeps:/src \
  -e TZ=UTC \
  -e LANG=C.UTF-8 \
  "$IMAGE" -c '
    set -e
    echo "container python: $(python3 -V)"
    python3 -c "import pytest; print(\"pytest\", pytest.__version__)"
    python3 -m pytest -q -p no:cacheprovider "$@"
  ' sh $FILES
