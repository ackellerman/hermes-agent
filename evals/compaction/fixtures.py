"""SPEC-0042 labeled fixtures: synthetic conversations with corrections,
decisions, commitments, abandoned trajectories, and world effects — the ground
truth that drives the AC-8 correction recall eval and the AC-11 gate stability
check.

Fixture format (JSON, evals/compaction/fixtures/*.json): a list of messages
plus per-class labels. Each correction label carries a ``counterfactual_anchor``:
the specific later message range that changed as a direct result of the
correction (a follow-up citation validates intent only if it matches this range,
± 1 message tolerance for boundary drift — spec AC-8).
"""

import json
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def build_correction_fixture() -> dict:
    """A conversation containing one labeled correction with its
    counterfactual anchor, one commitment, one decision, one abandoned
    trajectory, one world effect, and one unrecoverable artifact."""
    messages = [
        {"role": "user", "content": "Set up the export service. Use CSV format for the reports."},
        {"role": "assistant", "content": "Starting the export service with CSV format.",
         "tool_calls": [{"id": "c1", "function": {"name": "terminal",
                       "arguments": json.dumps({"command": "python export.py --format csv"})}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "CSV exporter started on port 8080."},
        {"role": "user", "content": "Wait — actually, use JSONL format, not CSV. "
         "The downstream pipeline can't parse CSV safely."},
        {"role": "assistant", "content": "Understood: switching to JSONL format.",
         "tool_calls": [{"id": "c2", "function": {"name": "terminal",
                       "arguments": json.dumps({"command": "python export.py --format jsonl"})}}]},
        {"role": "tool", "tool_call_id": "c2",
         "content": "JSONL exporter restarted on port 8080. Config written to export.yaml."},
        # counterfactual anchor: messages 5-6 — the exporter was RESTARTED with
        # jsonl, an action that would not have happened without the correction.
        {"role": "user", "content": "Also commit to shipping the export module by Friday."},
        {"role": "assistant", "content": "Commitment noted: export module ships Friday."},
        {"role": "user", "content": "Should we use Postgres or SQLite for the audit log?"},
        {"role": "assistant", "content": "Decision: Postgres — concurrent audit writers; "
         "rejected SQLite (single-writer bottleneck at our volume)."},
        {"role": "user", "content": "Try the experimental streaming transport too."},
        {"role": "assistant", "content": "Abandoned: streaming transport — kept dropping "
         "connections under load; not worth the complexity now."},
        {"role": "user", "content": "Generate a summary report artifact."},
        {"role": "assistant", "content": "Wrote audit-summary.txt to /tmp (ephemeral node, "
         "unrecoverable): totals per day, 3 anomalies flagged."},
        {"role": "user", "content": "Deploy the exporter to staging."},
        {"role": "assistant", "content": "Deployed to staging (world effect: staging now runs "
         "the JSONL exporter; staging-42 provisioned)."},
    ]
    labels = {
        "corrections": [
            {"what": "use JSONL, not CSV", "cites": [3, 3],
             "counterfactual_anchor": [5, 6]},
        ],
        "commitments": [{"what": "ship export module Friday", "cites": [7, 8]}],
        "decisions": [{"what": "Postgres for audit log", "cites": [9, 10],
                       "rejected_alternatives": ["SQLite"]}],
        "abandoned": [{"what": "streaming transport", "cites": [11, 12]}],
        "artifacts": [{"what": "audit-summary.txt", "cites": [13, 14],
                       "recoverable": False}],
        "world_effects": [{"what": "staging runs JSONL exporter", "cites": [15, 16]}],
    }
    return {"messages": messages, "labels": labels}


def build_gate_fixture() -> dict:
    """Dump content holding a correction, a commitment, and an unrecoverable
    artifact — the AC-10 gate input class."""
    fx = build_correction_fixture()
    return {"messages": fx["messages"], "labels": fx["labels"]}


_MAP_TOPICS = [
    "provisioning cluster alpha",
    "hardening authentication service",
    "refactoring the billing parser",
    "load-testing the gateway",
    "migrating the audit store to Postgres",
    "resolving the schema drift in analytics",
    "onboarding the ETL workers",
    "cutting the release branch",
]

_MAP_ROLE_TURNS = [
    ("user", "Let's work on: {topic}."),
    ("assistant", "Starting work on {topic}; outlining the plan.",
     ["terminal"], "plan for {topic} written"),
    ("tool", "{topic}: dry run complete"),
    ("user", "Please push the {topic} changes through staging."),
    ("assistant", "Staged the {topic} change set.",
     ["terminal"], "{topic} change set staged"),
    ("tool", "{topic}: staging deploy ok"),
    ("assistant", "Verified {topic} end to end; noting the result.",
     ["read_file"], "{topic} verification read"),
    ("tool", "{topic}: verification passed"),
]


def build_map_iou_fixture(*, messages_per_episode: int = 25,
                          n_episodes: int = 8,
                          padding_tokens: int = 0) -> dict:
    """Synthetic conversation with DISCRETE topic blocks — each episode is a
    solid run of one ``_MAP_TOPICS`` keyword (a task), so a deterministic
    topic-transition segmenter can recover the boundaries. Used by the AC-3 IoU
    falsifier (episode boundaries vs hand-labeled ground truth >= 90% IoU) and
    the AC-5 window-tax falsifier (sum of update inputs <= 1.15 x region).

    ``padding_tokens`` adds neutral filler to every assistant content so the
    region can be scaled toward the spec's 100K-token AC-5 target without
    changing the topic structure.
    """
    messages = []
    if padding_tokens:
        target_chars = padding_tokens * 4
        filler = ("Neutral progress note. " * ((target_chars // 23) + 1))[:target_chars]
    else:
        filler = ""
    ground_truth = []
    base = 0
    for ep_idx, topic in enumerate(_MAP_TOPICS[:n_episodes]):
        start = base
        for k in range(messages_per_episode):
            role, text, *rest = _MAP_ROLE_TURNS[k % len(_MAP_ROLE_TURNS)]
            content = text.format(topic=topic) + (filler if role == "assistant" else "")
            msg = {"role": role, "content": content}
            if len(rest) >= 1 and isinstance(rest[0], list) and rest[0]:
                msg["tool_calls"] = [{
                    "id": f"tc-{ep_idx}-{k}",
                    "function": {"name": rest[0][0], "arguments": '{"target":"%s"}' % topic}}]
            if role == "tool":
                msg["tool_call_id"] = f"tc-{ep_idx}-{k - 1}"
            messages.append(msg)
            base += 1
        ground_truth.append([start, base - 1])
    return {"messages": messages, "ground_truth_episodes": ground_truth}


def write_fixtures() -> list:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    fixtures = {
        "correction_commitment_decision.json": build_correction_fixture(),
        "gate_three_classes.json": build_gate_fixture(),
    }
    paths = []
    for name, fx in fixtures.items():
        p = FIXTURES_DIR / name
        p.write_text(json.dumps(fx, indent=2))
        paths.append(str(p))
    # AC-3/AC-5 fixture: synthetic topic-blocked conversation + hand-labeled
    # ground-truth episode boundaries (committed data, the IoU reference).
    map_fixture = build_map_iou_fixture()
    p = FIXTURES_DIR / "map_iou_transcript.json"
    p.write_text(json.dumps({"messages": map_fixture["messages"]}, indent=2))
    paths.append(str(p))
    p = FIXTURES_DIR / "map_iou_ground_truth.json"
    p.write_text(json.dumps({"ground_truth_episodes": map_fixture["ground_truth_episodes"],
                             "schema": "list of [start_msg, end_msg] inclusive episode ranges"},
                            indent=2))
    paths.append(str(p))
    return paths


if __name__ == "__main__":
    for path in write_fixtures():
        print(path)