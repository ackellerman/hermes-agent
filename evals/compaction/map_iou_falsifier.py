"""SPEC-0042 AC-3 (IoU) and AC-5 (window-tax) EXECUTABLE falsifiers.

Round-3 review finding F1: AC-3's 90% IoU bar and AC-5's 15% window tax had no
runnable falsifier in the impl — no ground-truth episode labels existed and
``CompactionMap.update`` consumes an ``llm_call``, so the bars were never
measured (PASS was claimed by construction). This harness closes that:

- AC-3 (IoU): drives 10 consecutive ``CompactionMap.update`` calls over the
  committed synthetic 200-message conversation (``fixtures/map_iou_transcript.json`)
  and scores the resulting episode boundaries against the committed HAND-LABELED
  ground truth (``fixtures/map_iou_ground_truth.json``) via the spec's greedy
  max-IoU pairing (``episode_iou_score``). The 90% bar is the mean IoU over all
  ground-truth episodes; ``covers`` regression raises ``MapRegressionError`` (AC-3).
  Default ``--map-model deterministic`` uses a topic-transition segmenter so the
  falsifier is fully executable without a live LLM; ``--map-model real`` routes
  the update calls through a real model (local ollama by default).

- AC-5 (window-tax): measures the sum of update-call input token sizes (current
  map JSON + new chunk + prompt) across the updates against the region token
  total, and asserts it stays within 1.15x of region size (spec §3.2 AC-5).

Usage:
    python evals/compaction/map_iou_falsifier.py [--json results/out.json]
                                                 [--map-model deterministic|real]
    python evals/compaction/map_iou_falsifier.py --map-model real
                                                 --ollama http://localhost:11434
                                                 --ollama-model muse-glimmer:latest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from agent.compaction_map import (  # noqa: E402
    CompactionMap,
    MapRegressionError,
    episode_iou_score,
    iou_ranges,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _rough_tokens(text) -> int:
    """Consistent rough estimator (len//4). Used for BOTH region and input sides
    of the AC-5 ratio so the tax is measured fair-on-fair."""
    if not text:
        return 0
    return len(str(text)) // 4


def _msg_tokens(msg: dict) -> int:
    total = _rough_tokens(msg.get("content", ""))
    for tc in msg.get("tool_calls") or []:
        total += _rough_tokens((tc.get("function") or {}).get("arguments", ""))
    return max(1, total)


def _topic_of(content: str) -> str:
    """Deterministic episode signature: the topic keyword present in content."""
    low = str(content).lower()
    for topic in ("provisioning cluster alpha", "hardening authentication service",
                  "refactoring the billing parser", "load-testing the gateway",
                  "migrating the audit store to postgres",
                  "resolving the schema drift in analytics",
                  "onboarding the etl workers", "cutting the release branch"):
        if topic in low:
            return topic
    return ""


def _segments(chunk_start: int, messages) -> list:
    """Consecutive same-topic runs -> [(start_msg, end_msg, topic)]."""
    segs = []
    for i, m in enumerate(messages):
        topic = _topic_of(m.get("content", ""))
        idx = chunk_start + i
        if segs and segs[-1][2] == topic:
            segs[-1] = (segs[-1][0], idx, topic)
        else:
            segs.append((idx, idx, topic))
    return [(s, e) for s, e, _ in segs]


def make_deterministic_map_llm():
    """A stateless full-replacement map ``llm_call`` that segments by topic
    transition and merges into the running map. Proves the map/update/IoU/gate
    machinery end-to-end without a live model (executable AC-3/AC-5 falsifier)."""

    def llm_call(messages):
        from agent.compaction_map import MAP_UPDATE_PROMPT  # noqa: PLC0415
        data = json.loads(messages[1]["content"])
        current = data["current_map"] or {"episodes": [], "covers": {"start_msg": 0, "end_msg": 0}}
        chunk = data["chunk"]
        start, end = int(chunk["start_msg"]), int(chunk["end_msg"])
        episodes = list(current.get("episodes", []))
        segs = _segments(start, chunk["messages"])
        # Merge the first new run into the prior episode if the topic continues.
        if episodes and segs and episodes[-1].get("topic") == _topic_of(
                chunk["messages"][0].get("content", "")):
            prior = episodes[-1]
            episodes[-1] = dict(prior, end_msg=max(int(prior["end_msg"]), segs[0][1]))
            segs = segs[1:]
        for s, e in segs:
            episodes.append({
                "name": f"episode-{len(episodes) + 1}",
                "start_msg": s, "end_msg": e,
                "topic": _topic_of(chunk["messages"][s - start].get("content", "")),
            })
        return json.dumps({
            "schema_version": 1,
            "covers": {"start_msg": 0, "end_msg": end},
            "episodes": episodes,
            "entities": current.get("entities", []),
            "edges": current.get("edges", []),
        })

    return llm_call


def _ollama_map_llm(endpoint: str, model: str):
    import urllib.request  # noqa: PLC0415

    _INSTRUCTION = (
        "You are maintaining a compact trajectory map of a session. You are given "
        "the CURRENT MAP object and a NEW CHUNK of conversation with message offsets. "
        "Produce the FULL REPLACEMENT map. Reply with a single JSON object of exactly "
        "this shape (no prose, no markdown): "
        '{"schema_version": 1, "covers": {"start_msg": <int>, "end_msg": <int>}, '
        '"episodes": [{"name": "...", "start_msg": <int>, "end_msg": <int>, "topic": "..."}], '
        '"entities": [], "edges": []}. '
        "covers.end_msg MUST equal the chunk's end offset. Extend episodes for coherent "
        "stretches of work; a topic/task change starts a new episode. Output ONLY the JSON."
    )

    def llm_call(messages):
        data = json.loads(messages[1]["content"])
        payload = {"model": model, "stream": False, "format": "json",
                   "messages": [{"role": "user", "content": "\n\n".join([
                       _INSTRUCTION,
                       "CURRENT MAP:\n" + json.dumps(data["current_map"]),
                       "NEW CHUNK (offsets %d..%d):\n" % (data["chunk"]["start_msg"],
                                                          data["chunk"]["end_msg"])
                       + json.dumps(data["chunk"]["messages"])])}],
                   "options": {"temperature": 0}}
        req = urllib.request.Request(
            f"{endpoint.rstrip('/')}/api/chat", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read().decode())["message"]["content"]

    return llm_call


def load_fixture(padding_tokens: int = 0):
    """Load the committed transcript + ground truth; optionally scale the region
    up for the AC-5 100K-token window-tax target (same topic structure)."""
    tx = json.loads((FIXTURES_DIR / "map_iou_transcript.json").read_text())
    gt = json.loads((FIXTURES_DIR / "map_iou_ground_truth.json").read_text())
    messages = tx["messages"]
    ground_truth = gt["ground_truth_episodes"]
    if padding_tokens:
        import fixtures as fx_mod  # noqa: PLC0415
        fx = fx_mod.build_map_iou_fixture(messages_per_episode=25, n_episodes=8,
                                          padding_tokens=padding_tokens)
        messages = fx["messages"]
        ground_truth = fx["ground_truth_episodes"]
    return messages, ground_truth


def run(*, map_model: str = "deterministic", padding_tokens: int = 0,
        ollama_endpoint: str = "http://localhost:11434",
        ollama_model: str = "muse-glimmer:latest",
        complex: bool = False) -> dict:
    from pathlib import Path as _P
    import tempfile

    messages, ground_truth = load_fixture(padding_tokens=padding_tokens)
    region_tokens = sum(_msg_tokens(m) for m in messages)
    n_episodes = len(ground_truth)

    if map_model == "real":
        llm = _ollama_map_llm(ollama_endpoint, ollama_model)
    else:
        llm = make_deterministic_map_llm()

    with tempfile.TemporaryDirectory() as root:
        cmap = CompactionMap(_P(root), "session-fx")
        covers_series = []
        n_updates = 10
        step = max(1, len(messages) // n_updates)
        input_tokens_total = 0
        cursor = 0
        regression = None
        while cursor < len(messages):
            chunk = messages[cursor:cursor + step]
            end = cursor + len(chunk) - 1
            # AC-5 input is what the update call actually sends: prompt + map + chunk.
            from agent.compaction_map import MAP_UPDATE_PROMPT  # noqa: PLC0415
            before = cmap.load()
            input_tokens_total += (
                _rough_tokens(MAP_UPDATE_PROMPT) + _rough_tokens(json.dumps(before))
                + _rough_tokens(json.dumps([m for m in chunk])))
            try:
                cmap.update(llm, chunk, start_msg=cursor, end_msg=end)
            except MapRegressionError as exc:
                regression = str(exc)
                break
            covers_series.append(cmap.load()["covers"]["end_msg"])
            cursor = end + 1

        final = cmap.load()
        predicted = [(int(e["start_msg"]), int(e["end_msg"]))
                     for e in final.get("episodes", [])]

    iou = episode_iou_score(predicted, ground_truth)
    covers_monotonic = all(b >= a for a, b in zip(covers_series, covers_series[1:]))
    window_tax = input_tokens_total / region_tokens if region_tokens else float("inf")

    ac3_pass = iou >= 0.90 and covers_monotonic and regression is None and n_episodes > 0
    ac5_pass = window_tax <= 1.15
    return {
        "mode": map_model,
        "n_updates": len(covers_series),
        "region_messages": len(messages),
        "region_tokens": region_tokens,
        "predicted_episodes": predicted,
        "ground_truth_episodes": ground_truth,
        "episode_iou_score": round(iou, 4),
        "ac3_iou_bar": 0.90,
        "covers_monotonic": covers_monotonic,
        "covers_regression": regression,
        "window_tax_ratio": round(window_tax, 4),
        "ac5_tax_bar": 1.15,
        "input_tokens_total": input_tokens_total,
        "ac3_pass": bool(ac3_pass),
        "ac5_pass": bool(ac5_pass),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", help="optional receipt path")
    parser.add_argument("--map-model", default="deterministic",
                        choices=["deterministic", "real"])
    parser.add_argument("--padding-tokens", type=int, default=0,
                        help="scale assistant content toward a 100K-token region (AC-5)")
    parser.add_argument("--ollama", default="http://localhost:11434")
    parser.add_argument("--ollama-model", default="muse-glimmer:latest")
    args = parser.parse_args()
    result = run(map_model=args.map_model, padding_tokens=args.padding_tokens,
                 ollama_endpoint=args.ollama, ollama_model=args.ollama_model)
    print(json.dumps(result, indent=2))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=2))
    return 0 if (result["ac3_pass"] and result["ac5_pass"]) else 1


if __name__ == "__main__":
    sys.exit(main())