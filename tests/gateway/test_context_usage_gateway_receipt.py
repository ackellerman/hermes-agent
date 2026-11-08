"""The one-shot context reading (``context_usage``) survives the gateway-backed ``chat``/``-z`` path.

``run_conversation`` stamps ``context_usage`` on every run result (fork-carried, see UPSTREAM-PENDING).
Since upstream moved ``hermes chat --oneshot`` / ``-z`` onto a gateway client, the result the client
prints is the receipt the gateway owner committed, not the in-process dict: the in-process turn
projection (``gateway/run_turn_runner.py``) and the managed worker's result frame
(``agent/managed_worker.py``) both keep only listed keys. Without ``context_usage`` in those lists the
stream-json ``result.context`` and the usage file's ``context`` are always null.
"""
import io

from agent.managed_worker import accept_result, encode_frame, read_frame, result_frame
from hermes_cli.stream_json import result_context

READING = {"context_used": 12345, "context_max": 200000, "context_percent": 6,
           "context_source": "provider_usage", "context_estimated": False}
PUBLIC = {"used": 12345, "max": 200000, "percent": 6, "source": "provider_usage", "estimated": False}


def test_in_process_turn_projection_keeps_context_usage():
    from gateway import run_turn_runner
    assert "context_usage" in run_turn_runner._LEDGER_PASSTHROUGH_KEYS, \
        "the gateway turn projection drops the one-shot context reading"
    projected = {key: {"context_usage": READING}.get(key) for key in run_turn_runner._LEDGER_PASSTHROUGH_KEYS
                 if key in {"context_usage": READING}}
    assert result_context(projected) == PUBLIC


def _frame_round_trip(result):
    wire = read_frame(io.BytesIO(encode_frame({"type": "result", "result": result_frame(result)})))
    return accept_result(wire["result"])


def test_managed_worker_result_frame_keeps_context_usage():
    accepted, _usage = _frame_round_trip({"final_response": "ok", "completed": True, "context_usage": READING})
    assert accepted.get("context_usage") == READING
    assert result_context(accepted) == PUBLIC


def test_managed_worker_context_usage_none_and_shape_guard():
    accepted, _ = _frame_round_trip({"final_response": "ok", "context_usage": None})
    assert result_context(accepted) is None
    # Unknown or nested keys in the reading are refused at the owner boundary, like billing_block.
    for bad in ({"final_response": "x", "context_usage": {**READING, "extra": 1}},
                {"final_response": "x", "context_usage": {"context_used": {"nested": 1}}},
                {"final_response": "x", "context_usage": "not a dict"}):
        try:
            accept_result(bad)
        except ValueError:
            continue
        raise AssertionError(bad)
