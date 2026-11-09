"""``display.subagent_status: summary``: one status message per ``delegate_task`` call.

The gateway posts the message on the call's first ``subagent.start`` and edits it in place as
children finish. Grandchildren are indented under their parent's line in the same message. The
message has its own send path (tool progress off still shows it), stays out of
``cleanup_progress``, and is never sent to a bot peer or on a muted turn. Once posted, later edits
land even after the turn's generation moved on (background delegations).
"""

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

import gateway.run_turn_progress as progress_mod
import hermes_yaml
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource
from gateway.turn_context import TurnContext
from tests.waiting import wait_until

SUMMARY_CONFIG = {"display": {"platforms": {"discord": {"subagent_status": "summary"}}}}


class StatusCaptureAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.DISCORD)
        self.sent: list[dict] = []
        self.edits: list[dict] = []
        self._lock = threading.Lock()

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        with self._lock:
            message_id = f"msg-{len(self.sent) + 1}"
            self.sent.append({"chat_id": chat_id, "content": content, "reply_to": reply_to,
                              "metadata": metadata, "message_id": message_id})
        return SendResult(success=True, message_id=message_id)

    async def edit_message(self, chat_id, message_id, content, metadata=None) -> SendResult:
        with self._lock:
            self.edits.append({"chat_id": chat_id, "message_id": message_id, "content": content,
                               "metadata": metadata, "at": time.monotonic()})
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}

    def latest(self, message_id: str) -> str:
        """The message's current text: its last edit, else what was sent."""
        with self._lock:
            for edit in reversed(self.edits):
                if edit["message_id"] == message_id:
                    return edit["content"]
            for sent in self.sent:
                if sent["message_id"] == message_id:
                    return sent["content"]
        return ""


class _StubGatewayRunner:
    def __init__(self, adapter):
        self.adapter = adapter
        self.notices: list[str] = []

    def _delivery_adapter_for(self, source):
        return self.adapter

    async def _deliver_platform_notice(self, source, content):
        self.notices.append(content)


@pytest.fixture
def loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()


@pytest.fixture(autouse=True)
def _fast_edits(monkeypatch):
    """Edits coalesce on a 1.5 s window in production; most cases only need the ordering."""
    monkeypatch.setattr(progress_mod, "SUBAGENT_STATUS_EDIT_INTERVAL", 0.05, raising=False)


def _source(**kw):
    return SessionSource(platform=Platform.DISCORD, chat_id="chan-1", chat_type="thread", **kw)


def _turn(loop, *, user_config=SUMMARY_CONFIG, still_current=None, source=None, runner=None, **ctx_kw):
    from gateway.run_turn_runner import TurnRunner
    gateway_runner = runner or _StubGatewayRunner(StatusCaptureAdapter())
    current = still_current if still_current is not None else [True]
    ctx = TurnContext(
        source=source or _source(), user_config=user_config, _run_still_current=lambda: current[0],
        _loop_for_step=loop, _progress_reply_to="anchor-9", _progress_metadata={"thread_id": "t-1"},
        **ctx_kw,
    )
    return TurnRunner(gateway_runner, ctx), gateway_runner, current


def _start(turn, sid, *, call, goal, model="claude-opus-5-5", parent=None, depth=0, route=None, count=2, index=0):
    kw = {"subagent_id": sid, "depth": depth, "model": model, "goal": goal, "task_index": index,
          "task_count": count, "delegate_call_id": call}
    if parent is not None:
        kw["parent_id"] = parent
    if route is not None:
        kw["route_label"] = route
    turn.progress_callback("subagent.start", preview=goal, **kw)


def _complete(turn, sid, *, call, goal, status="completed", duration=192.0, parent=None, depth=0, model="claude-opus-5-5"):
    kw = {"subagent_id": sid, "depth": depth, "model": model, "goal": goal, "delegate_call_id": call,
          "status": status, "duration_seconds": duration, "summary": "boom" if status != "completed" else "done"}
    if parent is not None:
        kw["parent_id"] = parent
    turn.progress_callback("subagent.complete", preview=kw["summary"], **kw)


def _wait(predicate, timeout=5.0):
    return wait_until(predicate, timeout=timeout)


def _quiesce(loop, runner, timeout=5.0):
    """Every update scheduled so far has run, and no message still has an edit pending."""
    asyncio.run_coroutine_threadsafe(asyncio.sleep(0), loop).result(timeout)
    wait_until(lambda: not any(m.pumping for m in list(getattr(runner, "_subagent_status_calls", {}).values())),
               timeout=timeout)


def test_one_message_with_route_label_nesting_and_outcomes(loop):
    """(1) Two top-level children plus a grandchild: one send, then edits in place."""
    turn, runner, _ = _turn(loop)
    adapter = runner.adapter
    _start(turn, "sa-1", call="call-A", goal="audit the fixtures", model="deepseek-v4.1-flash:cloud", route="flash")
    _start(turn, "sa-2", call="call-A", goal="draft the test", index=1)
    # The grandchild belongs to sa-2's own delegate_task call.
    _start(turn, "sa-3", call="call-B", goal="check the fixture", parent="sa-2", depth=1, count=1)
    _complete(turn, "sa-3", call="call-B", goal="check the fixture", parent="sa-2", depth=1,
              status="failed", duration=41.0)
    _complete(turn, "sa-2", call="call-A", goal="draft the test", duration=192.0)

    assert _wait(lambda: len(adapter.sent) == 1 and "✓ 3m12s" in adapter.latest("msg-1"))
    _quiesce(loop, runner)
    assert len(adapter.sent) == 1, "grandchild's own call must not post separately"
    assert adapter.edits, "later events edit the first message"
    assert {e["message_id"] for e in adapter.edits} == {"msg-1"}
    sent = adapter.sent[0]
    assert sent["reply_to"] == "anchor-9" and sent["metadata"] == {"thread_id": "t-1"}

    lines = adapter.latest("msg-1").splitlines()
    assert lines[0] == "🧩 subagents (3)"
    assert lines[1] == '↳ flash · deepseek-v4.1-flash:cloud — "audit the fixtures"  … running'
    assert lines[2] == '↳ claude-opus-5-5 — "draft the test"  ✓ 3m12s'
    assert lines[3] == '    ↳ claude-opus-5-5 — "check the fixture" (nested)  ✗ failed 41s'


def test_concurrent_calls_get_separate_messages(loop):
    """(2) Two delegate_task calls in one turn: two messages, no cross-talk."""
    turn, runner, _ = _turn(loop)
    adapter = runner.adapter
    _start(turn, "a-1", call="call-A", goal="alpha one")
    _start(turn, "b-1", call="call-B", goal="beta one")
    _start(turn, "a-2", call="call-A", goal="alpha two", index=1)
    _start(turn, "b-2", call="call-B", goal="beta two", index=1)
    _complete(turn, "a-1", call="call-A", goal="alpha one", duration=5)
    _complete(turn, "b-2", call="call-B", goal="beta two", status="error", duration=7)

    def texts():
        return [adapter.latest(s["message_id"]) for s in list(adapter.sent)]

    assert _wait(lambda: len(adapter.sent) == 2
                 and any("alpha one" in text and "✓ 5s" in text for text in texts())
                 and any("beta two" in text and "✗ error 7s" in text for text in texts()))
    _quiesce(loop, runner)
    assert len(adapter.sent) == 2
    by_msg = {s["message_id"]: adapter.latest(s["message_id"]) for s in adapter.sent}
    alpha = next(text for text in by_msg.values() if "alpha one" in text)
    beta = next(text for text in by_msg.values() if "beta one" in text)
    assert "beta" not in alpha and "alpha" not in beta
    assert "alpha two" in alpha and "beta two" in beta


def test_posts_with_tool_progress_off(loop):
    """(3) tool_progress off (no progress queue) + summary: the message still posts."""
    config = {"display": {"tool_progress": "off", "platforms": {"discord": {"subagent_status": "summary"}}}}
    turn, runner, _ = _turn(loop, user_config=config, progress_queue=None, tool_progress_enabled=False)
    _start(turn, "sa-1", call="call-A", goal="audit")
    assert _wait(lambda: len(runner.adapter.sent) == 1)


def test_never_in_cleanup_ids(loop):
    turn, runner, _ = _turn(loop, _cleanup_progress=True)
    _start(turn, "sa-1", call="call-A", goal="audit", count=1)
    _complete(turn, "sa-1", call="call-A", goal="audit", duration=3)
    assert _wait(lambda: len(runner.adapter.sent) == 1 and "✓ 3s" in runner.adapter.latest("msg-1"))
    assert turn._ctx._cleanup_msg_ids == []


@pytest.mark.parametrize("gate", ["bot_peer", "muted"])
def test_no_message_for_bot_peer_or_muted_turn(loop, gate):
    """(4) A bot peer and a muted turn get no status message."""
    if gate == "bot_peer":
        turn, runner, _ = _turn(loop, source=_source(is_bot=True))
    else:
        turn, runner, _ = _turn(loop, mute_notification_reply=True)
    _start(turn, "sa-1", call="call-A", goal="audit", count=1)
    _complete(turn, "sa-1", call="call-A", goal="audit", duration=3)
    _quiesce(loop, runner)
    assert runner.adapter.sent == [] and runner.adapter.edits == []


def test_detached_completion_edits_existing_message(loop):
    """(5) The turn's generation moves on, then a child finishes: its line is edited, no new send."""
    turn, runner, current = _turn(loop)
    adapter = runner.adapter
    _start(turn, "sa-1", call="call-A", goal="background job", count=1)
    assert _wait(lambda: len(adapter.sent) == 1)
    current[0] = False
    _complete(turn, "sa-1", call="call-A", goal="background job", duration=65)
    assert _wait(lambda: "✓ 1m05s" in adapter.latest("msg-1"))
    _quiesce(loop, runner)
    assert len(adapter.sent) == 1


def test_stale_turn_never_opens_a_message(loop):
    turn, runner, _ = _turn(loop, still_current=[False])
    _start(turn, "sa-1", call="call-A", goal="audit", count=1)
    _quiesce(loop, runner)
    assert runner.adapter.sent == []


def test_thirty_children_stay_under_cap_with_collapse_line(loop):
    """(6) 30 children: content stays ≤1900 chars, the oldest finished lines collapse."""
    turn, runner, _ = _turn(loop)
    adapter = runner.adapter
    for i in range(30):
        _start(turn, f"sa-{i}", call="call-A", goal=f"task number {i} " + "x" * 80, count=30, index=i)
    for i in range(30):
        _complete(turn, f"sa-{i}", call="call-A", goal=f"task number {i}", duration=10 + i,
                  status="failed" if i % 3 == 0 else "completed")
    assert _wait(lambda: adapter.sent and "✓ 39s" in adapter.latest("msg-1"))
    _quiesce(loop, runner)
    for text in [s["content"] for s in adapter.sent] + [e["content"] for e in adapter.edits]:
        assert len(text) <= 1900
    final = adapter.latest("msg-1")
    assert final.splitlines()[0] == "🧩 subagents (30)"
    collapse = [line for line in final.splitlines() if line.startswith("… ") and " more (" in line]
    assert len(collapse) == 1
    hidden = int(collapse[0].split()[1])
    assert hidden > 0
    shown = [line for line in final.splitlines() if line.startswith("↳")]
    assert hidden + len(shown) == 30
    # The newest lines stay visible; the oldest are the ones collapsed.
    assert '"task number 29' in final and '"task number 0 ' not in final
    ok = sum(1 for i in range(hidden) if i % 3 != 0)
    assert collapse[0] == f"… {hidden} more ({ok} ok, {hidden - ok} failed)"


def test_rapid_completions_coalesce_into_at_most_two_edits(loop, monkeypatch):
    """(7) Five completions within 1 s under the real 1.5 s window: ≤2 edits, the last shows all five.
    A loop barrier after each completion lets the pump run between them, so without the window
    each completion would get its own edit."""
    monkeypatch.setattr(progress_mod, "SUBAGENT_STATUS_EDIT_INTERVAL", 1.5)
    turn, runner, _ = _turn(loop)
    adapter = runner.adapter
    for i in range(5):
        _start(turn, f"sa-{i}", call="call-A", goal=f"job {i}", count=5, index=i)
    assert _wait(lambda: len(adapter.sent) == 1)
    _quiesce(loop, runner)  # the starts' own edit (if any) lands
    before = len(adapter.edits)
    for i in range(5):  # back to back: well within 1 s, inside one 1.5 s edit window
        _complete(turn, f"sa-{i}", call="call-A", goal=f"job {i}", duration=2 + i)
        for _ in range(3):  # the kick's schedule, the pump's step, and its edit all get to run
            asyncio.run_coroutine_threadsafe(asyncio.sleep(0), loop).result(5)
    assert _wait(lambda: adapter.latest("msg-1").count("✓") == 5, timeout=6)
    _quiesce(loop, runner)
    new_edits = adapter.edits[before:]
    assert 1 <= len(new_edits) <= 2
    assert new_edits[-1]["content"].count("✓") == 5


@pytest.mark.parametrize("config", [
    {"display": {"subagent_status": "off"}},
    hermes_yaml.safe_load("display:\n  platforms:\n    discord:\n      subagent_status: off\n"),
    {"display": {}},
    {},
], ids=["string-off", "yaml-bare-off", "key-absent", "no-display"])
def test_off_bare_off_and_absent_post_nothing(loop, config):
    """(8) Config off, YAML bare off (loads as False) and an absent key: no message."""
    turn, runner, _ = _turn(loop, user_config=config)
    _start(turn, "sa-1", call="call-A", goal="audit", count=1)
    _complete(turn, "sa-1", call="call-A", goal="audit", status="failed", duration=3)
    _quiesce(loop, runner)
    assert runner.adapter.sent == [] and runner.adapter.edits == []
    assert len(runner.notices) == 1, "the failure notice is unchanged when the message is off"


def test_line_without_route_has_no_label(loop):
    """(9) A child with no route shows its model only."""
    turn, runner, _ = _turn(loop)
    _start(turn, "sa-1", call="call-A", goal="plain child", model="claude-opus-5-5", count=1)
    assert _wait(lambda: len(runner.adapter.sent) == 1)
    assert runner.adapter.latest("msg-1").splitlines()[1] == '↳ claude-opus-5-5 — "plain child"  … running'


def _route(label=None, backend="default", error=None):
    from tools.delegate_tool_routes import TaskRoute
    return TaskRoute(creds={"model": "claude-opus-5-5", "provider": "anthropic", "base_url": None, "api_key": None,
                            "api_mode": None}, routing_cfg={}, label=label, backend=backend, error=error)


def _build_call(monkeypatch, parent_agent, goals, routes, *, prefix, parent_id=None, depth=0, model):
    """One real ``_build_children`` call whose stub children carry real relays to ``parent_agent``."""
    import tools.delegate_tool as dt
    from tools.delegate_tool import _build_child_progress_callback

    def fake_build(*, task_index, goal, parent_agent, task_count, **_kw):
        ref: dict = {}
        relay = _build_child_progress_callback(task_index, goal, parent_agent, task_count,
                                               subagent_id=f"{prefix}{task_index}", parent_id=parent_id,
                                               depth=depth, model=model, session_ref=ref)
        return SimpleNamespace(_progress_identity_ref=ref, tool_progress_callback=relay)

    monkeypatch.setattr(dt, "_build_child_preserving_parent_tools", fake_build)
    children, err = dt._build_children(
        [{"goal": goal} for goal in goals], [None] * len(goals), routes, top_role="leaf", max_iterations=5,
        parent_agent=parent_agent, live_deleg_id=None, live_writers=[],
    )
    assert err is None
    return [child for _i, _t, child in children]


def test_unrouted_grandchild_of_a_routed_child_has_no_route_label(loop, monkeypatch):
    """A grandchild's events are re-relayed through its routed parent child: they must not pick up
    that child's route label (or its call id) on the way up."""
    turn, runner, _ = _turn(loop)
    top = SimpleNamespace(session_id="parent-sess", _delegate_spinner=None, _subagent_id=None,
                          tool_progress_callback=turn.progress_callback)
    (child,) = _build_call(monkeypatch, top, ["routed child"], [_route(label="flash", backend="ollama")],
                           prefix="sa-c", model="model-c")
    child_agent = SimpleNamespace(session_id="child-sess", _delegate_spinner=None, _subagent_id="sa-c0",
                                  tool_progress_callback=child.tool_progress_callback)
    (grand,) = _build_call(monkeypatch, child_agent, ["plain grandchild"], [_route()], prefix="sa-g",
                           parent_id="sa-c0", depth=1, model="model-g")
    child.tool_progress_callback("subagent.start", preview="routed child")
    grand.tool_progress_callback("subagent.start", preview="plain grandchild")

    adapter = runner.adapter
    assert _wait(lambda: "plain grandchild" in adapter.latest("msg-1"))
    _quiesce(loop, runner)
    assert len(adapter.sent) == 1
    lines = adapter.latest("msg-1").splitlines()
    assert lines[1] == '↳ flash · model-c — "routed child"  … running'
    assert lines[2] == '    ↳ model-g — "plain grandchild" (nested)  … running'
    call_ids = {child._progress_identity_ref["delegate_call_id"], grand._progress_identity_ref["delegate_call_id"]}
    assert len(call_ids) == 2, "the grandchild keeps its own call id through the relay"
    assert set(getattr(runner, "_subagent_status_calls", {})) == call_ids


def test_goal_truncated_to_sixty_chars(loop):
    turn, runner, _ = _turn(loop)
    goal = "g" * 200
    _start(turn, "sa-1", call="call-A", goal=goal, count=1)
    assert _wait(lambda: len(runner.adapter.sent) == 1)
    quoted = runner.adapter.latest("msg-1").splitlines()[1].split('"')[1]
    assert len(quoted) <= 60 and quoted.startswith("g" * 50)


def test_failure_notice_unchanged_in_summary_mode(loop):
    turn, runner, _ = _turn(loop)
    _start(turn, "sa-1", call="call-A", goal="scan the repo", count=1)
    _complete(turn, "sa-1", call="call-A", goal="scan the repo", status="failed", duration=8)
    assert _wait(lambda: len(runner.notices) == 1)
    assert '"scan the repo"' in runner.notices[0]


def test_state_dropped_when_every_line_is_final(loop):
    turn, runner, _ = _turn(loop)
    _start(turn, "sa-1", call="call-A", goal="one", count=1)
    _complete(turn, "sa-1", call="call-A", goal="one", duration=1)
    assert _wait(lambda: "✓ 1s" in runner.adapter.latest("msg-1"))
    assert _wait(lambda: not getattr(runner, "_subagent_status_calls", {}))


class FlakyEditAdapter(StatusCaptureAdapter):
    """Edits numbered in ``fail`` (1-based) fail the way Discord reports any edit exception:
    ``success=False`` with the error text and no ``retryable`` flag."""

    def __init__(self, fail):
        super().__init__()
        self.fail = set(fail)
        self.attempts = 0

    async def edit_message(self, chat_id, message_id, content, metadata=None) -> SendResult:
        with self._lock:
            self.attempts += 1
            failed = self.attempts in self.fail
        if failed:
            return SendResult(success=False, error="503 Service Unavailable")
        return await super().edit_message(chat_id, message_id, content, metadata=metadata)


def test_failed_edit_is_retried_and_the_final_state_lands(loop):
    """A non-retryable edit failure (Discord 5xx) does not freeze the message at '… running'."""
    adapter = FlakyEditAdapter(fail={1})
    turn, runner, _ = _turn(loop, runner=_StubGatewayRunner(adapter))
    _start(turn, "sa-1", call="call-A", goal="one")
    _start(turn, "sa-2", call="call-A", goal="two", index=1)
    assert _wait(lambda: len(adapter.sent) == 1)
    _quiesce(loop, runner)
    _complete(turn, "sa-1", call="call-A", goal="one", duration=4)
    _complete(turn, "sa-2", call="call-A", goal="two", duration=6)
    assert _wait(lambda: adapter.latest("msg-1").count("✓") == 2)
    _quiesce(loop, runner)
    assert adapter.attempts >= 2 and len(adapter.sent) == 1
    assert "running" not in adapter.latest("msg-1")


def test_edit_failures_count_consecutively_not_cumulatively(loop):
    """Every other edit fails over a long run (more failures in total than the give-up limit):
    each success resets the count, so every completion still lands."""
    adapter = FlakyEditAdapter(fail=range(1, 40, 2))
    turn, runner, _ = _turn(loop, runner=_StubGatewayRunner(adapter))
    n = progress_mod._SUBAGENT_STATUS_MAX_EDIT_FAILURES + 3
    for i in range(n):
        _start(turn, f"sa-{i}", call="call-A", goal=f"job {i}", count=n, index=i)
    assert _wait(lambda: len(adapter.sent) == 1)
    _quiesce(loop, runner)
    for i in range(n):
        _complete(turn, f"sa-{i}", call="call-A", goal=f"job {i}", duration=1 + i)
        assert _wait(lambda i=i: adapter.latest("msg-1").count("✓") == i + 1), f"completion {i} never landed"
    assert len(adapter.fail & set(range(1, adapter.attempts + 1))) >= progress_mod._SUBAGENT_STATUS_MAX_EDIT_FAILURES


@pytest.mark.parametrize(("value", "expected"), [
    (True, "summary"), ("on", "summary"), ("ON", "summary"), ("summary", "summary"), (" Summary ", "summary"),
    (False, "off"), ("off", "off"), ("OFF", "off"), ("", "off"), ("none", "off"), ("disabled", "off"),
    ("summery", "off"), ("verbose", "off"),
])
def test_subagent_status_normaliser_table(value, expected):
    """Opt-in: only an explicit yes enables it; empty, unknown strings and typos are off."""
    from gateway.display_config import resolve_display_setting
    config = {"display": {"platforms": {"discord": {"subagent_status": value}}}}
    assert resolve_display_setting(config, "discord", "subagent_status", "off") == expected
    assert resolve_display_setting({"display": {"subagent_status": value}}, "discord", "subagent_status") == expected


def test_subagent_status_absent_is_off():
    from gateway.display_config import resolve_display_setting
    assert resolve_display_setting({}, "discord", "subagent_status") == "off"
    assert resolve_display_setting({"display": {}}, "discord", "subagent_status", "summary") == "off"


def test_refused_task_does_not_keep_the_call_open(loop, monkeypatch):
    """A route-refused task builds no child and never starts: once the built children finish,
    the call is over and its state is dropped."""
    turn, runner, _ = _turn(loop)
    top = SimpleNamespace(session_id="parent-sess", _delegate_spinner=None, _subagent_id=None,
                          tool_progress_callback=turn.progress_callback)
    built = _build_call(monkeypatch, top, ["one", "refused", "two"],
                        [_route(), _route(backend="ollama", error="no route"), _route()], prefix="sa-", model="m")
    assert len(built) == 2
    for child, goal in zip(built, ("one", "two")):
        child.tool_progress_callback("subagent.start", preview=goal)
    for child in built:
        child.tool_progress_callback("subagent.complete", preview="done", status="completed", duration_seconds=2)
    adapter = runner.adapter
    assert _wait(lambda: adapter.latest("msg-1").count("✓") == 2)
    assert _wait(lambda: not getattr(runner, "_subagent_status_calls", {}))
    assert adapter.latest("msg-1").splitlines()[0] == "🧩 subagents (2)"


def test_call_stays_open_until_every_built_child_started(loop):
    """One child finished, its sibling has not started yet: the call is not over, so the sibling's
    start edits the same message instead of posting a second one."""
    turn, runner, _ = _turn(loop)
    adapter = runner.adapter
    _start(turn, "sa-1", call="call-A", goal="one", count=2)
    _complete(turn, "sa-1", call="call-A", goal="one", duration=1)
    assert _wait(lambda: "✓ 1s" in adapter.latest("msg-1"))
    _quiesce(loop, runner)
    assert "call-A" in getattr(runner, "_subagent_status_calls", {})
    _start(turn, "sa-2", call="call-A", goal="two", count=2, index=1)
    assert _wait(lambda: '"two"' in adapter.latest("msg-1"))
    _quiesce(loop, runner)
    assert len(adapter.sent) == 1


def test_header_shows_started_of_expected(loop):
    """The header counts started children against the call's built children while some are pending."""
    turn, runner, _ = _turn(loop)
    adapter = runner.adapter
    _start(turn, "sa-0", call="call-A", goal="job 0", count=3)
    assert _wait(lambda: len(adapter.sent) == 1)
    assert adapter.sent[0]["content"].splitlines()[0] == "🧩 subagents (1/3)"
    _start(turn, "sa-1", call="call-A", goal="job 1", count=3, index=1)
    _start(turn, "sa-2", call="call-A", goal="job 2", count=3, index=2)
    assert _wait(lambda: adapter.latest("msg-1").splitlines()[0] == "🧩 subagents (3)")


def test_send_and_edits_carry_reply_to_and_metadata(loop):
    turn, runner, _ = _turn(loop)
    adapter = runner.adapter
    _start(turn, "sa-1", call="call-A", goal="one", count=1)
    assert _wait(lambda: len(adapter.sent) == 1)
    _quiesce(loop, runner)
    _complete(turn, "sa-1", call="call-A", goal="one", duration=1)
    assert _wait(lambda: "✓ 1s" in adapter.latest("msg-1"))
    _quiesce(loop, runner)
    assert [(s["chat_id"], s["reply_to"], s["metadata"]) for s in adapter.sent] == [("chan-1", "anchor-9", {"thread_id": "t-1"})]
    assert adapter.edits and all(e["metadata"] == {"thread_id": "t-1"} for e in adapter.edits)


def test_local_viewer_turn_posts_nothing(loop):
    """A local route's viewers get subagent.* through the event stream; no adapter message."""
    config = {"display": {"subagent_status": "summary"}}
    source = SessionSource(platform=Platform.LOCAL, chat_id="local-1", chat_type="dm")
    turn, runner, _ = _turn(loop, user_config=config, source=source)
    turn._approval_owner = (SimpleNamespace(sessions={}), "s", 1)
    _start(turn, "sa-1", call="call-A", goal="audit", count=1)
    _complete(turn, "sa-1", call="call-A", goal="audit", duration=3)
    _quiesce(loop, runner)
    assert runner.adapter.sent == [] and runner.adapter.edits == []


class NoEditAdapter(StatusCaptureAdapter):
    edit_message = BasePlatformAdapter.edit_message  # an adapter that cannot edit in place


def test_adapter_without_edit_support_posts_nothing(loop):
    turn, runner, _ = _turn(loop, runner=_StubGatewayRunner(NoEditAdapter()))
    _start(turn, "sa-1", call="call-A", goal="audit", count=1)
    _complete(turn, "sa-1", call="call-A", goal="audit", duration=3)
    _quiesce(loop, runner)
    assert runner.adapter.sent == []


class FailedSendAdapter(StatusCaptureAdapter):
    def __init__(self):
        super().__init__()
        self.send_attempts = 0

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        with self._lock:
            self.send_attempts += 1
        return SendResult(success=False, error="403 Missing Access")


def test_dead_message_is_a_tombstone_for_its_call(loop):
    """The first send failed: later starts of that call (and a nested call under it) post nothing
    until the call is over; then the state is dropped."""
    adapter = FailedSendAdapter()
    turn, runner, _ = _turn(loop, runner=_StubGatewayRunner(adapter))
    _start(turn, "sa-1", call="call-A", goal="one", count=2)
    assert _wait(lambda: adapter.send_attempts == 1)
    _quiesce(loop, runner)
    _start(turn, "sa-2", call="call-A", goal="two", count=2, index=1)
    _start(turn, "sa-3", call="call-B", goal="grand of one", parent="sa-1", depth=1, count=1)
    _quiesce(loop, runner)
    assert adapter.send_attempts == 1 and adapter.edits == []
    for sid, call, goal in (("sa-3", "call-B", "grand of one"), ("sa-1", "call-A", "one"), ("sa-2", "call-A", "two")):
        _complete(turn, sid, call=call, goal=goal, duration=1)
    _quiesce(loop, runner)
    assert adapter.send_attempts == 1
    assert not getattr(runner, "_subagent_status_calls", {})


def test_expired_tombstone_lets_a_later_start_post_again(loop, monkeypatch):
    """The tombstone is bounded: past its TTL a start of the same call may try a fresh message."""
    monkeypatch.setattr(progress_mod, "_SUBAGENT_STATUS_TOMBSTONE_TTL", -1.0)
    adapter = FailedSendAdapter()
    turn, runner, _ = _turn(loop, runner=_StubGatewayRunner(adapter))
    _start(turn, "sa-1", call="call-A", goal="one", count=2)
    assert _wait(lambda: adapter.send_attempts == 1)
    _quiesce(loop, runner)
    _start(turn, "sa-2", call="call-A", goal="two", count=2, index=1)
    assert _wait(lambda: adapter.send_attempts == 2)
