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
    return paths


if __name__ == "__main__":
    for path in write_fixtures():
        print(path)