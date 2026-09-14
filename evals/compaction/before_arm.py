"""SPEC-0043/0044 BEFORE-arm evaluation entry point.

D7 (SPEC-0044): there is exactly ONE before-arm implementation. It lives in
``online_eval._before_arm`` and drives the REAL ``ContextCompressor``
single-call summary path; this module is a thin CLI over it so the historical
``before_arm.py`` entry point cannot drift into a second, bespoke prompt that
only *looks* like the production path.

Usage (repo root, venv):
    python evals/compaction/before_arm.py --provider custom \
        --base-url http://127.0.0.1:8081/v1 --model Qwen3.8-v3:27b \
        [--json /tmp/before_arm.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).parent))

import fixtures as fx_mod  # noqa: E402
import online_eval as OE  # noqa: E402


def run(*, model: str, provider: str, base_url: str | None) -> dict:
    """Score the real ContextCompressor single-call summary on the labeled
    correction fixture, with the same per-class scorer the AFTER arm uses."""
    call_log: list = []
    fixture = fx_mod.build_correction_fixture()
    return OE._before_arm(fixture, model=model, provider=provider,
                          base_url=base_url, call_log=call_log)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="custom")
    parser.add_argument("--base-url", default="http://127.0.0.1:8081/v1")
    parser.add_argument("--model", default="Qwen3.8-v3:27b")
    parser.add_argument("--json", default="/tmp/before_arm_results.json")
    args = parser.parse_args()
    result = run(model=args.model, provider=args.provider, base_url=args.base_url)
    text = json.dumps(result, indent=2, default=str)
    print(text)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(text)
    return 0 if result.get("evidence", {}).get("method_invoked") else 1


if __name__ == "__main__":
    sys.exit(main())
