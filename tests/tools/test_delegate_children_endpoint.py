"""The kept-children row never stores the endpoint URL.

A ``base_url`` can carry a key in its userinfo or query. The row stores what the resumable record stores for the
same child, ``endpoint_identity`` (the provider and a digest of the exact URL), and nothing reads the URL back:
a continue re-resolves the route.
"""

import json
from concurrent.futures import Future

from tests.tools.test_delegate_children import _child, _child_session, _finish, _parent, _rows, db
from tools import delegate_children as dc
from tools.delegate_tool_resumable import endpoint_identity

_URL = "https://svc-user:pw-USERINFO@gateway.example.invalid/v1?api-key=QUERYSECRET"


def test_the_kept_row_stores_the_endpoint_digest_not_the_url(db):
    parent = _parent(db)
    _child_session(db, "kid-url", "parent", "sa-0-url")
    child = _child("kid-url", "sa-0-url")
    with dc.spawn_origin(dc.ORIGIN_MODEL):
        assert dc.note_spawn(parent, child, goal="g", context=None, toolsets=None, role="leaf", model="m",
                             route={"provider": "custom", "model": "m", "base_url": _URL, "api_mode": "chat_completions"})
    future: Future = Future()
    assert dc.worker_started(child, future)
    row = _rows(db.db_path)[0]
    route = json.loads(row["route"])
    assert route == {"provider": "custom", "model": "m", "api_mode": "chat_completions",
                     "endpoint": list(endpoint_identity("custom", _URL))}
    dump = json.dumps(row)
    for secret in ("pw-USERINFO", "QUERYSECRET", "svc-user", "gateway.example.invalid"):
        assert secret not in dump, secret
    _finish(future)


def test_the_real_child_build_writes_no_url_into_the_store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tools import delegate_tool as dt
    handle = SessionDB(db_path=tmp_path / "state.db")
    agents = []
    try:
        parent = AIAgent(session_id="p-desktop", session_db=handle, platform="desktop", api_key="k", base_url=_URL,
                         provider="custom", api_mode="chat_completions", model="m", quiet_mode=True,
                         skip_context_files=True, skip_memory=True, save_trajectories=False,
                         enabled_toolsets=["file"])
        agents.append(parent)
        parent._ensure_db_session()
        with dc.spawn_origin(dc.ORIGIN_MODEL):
            child = dt._build_child_agent(task_index=0, goal="goal", context=None, toolsets=["file"], model=None,
                                          max_iterations=4, task_count=1, parent_agent=parent)
        agents.append(child)
        child._ensure_db_session()
        future: Future = Future()
        assert dc.worker_started(child, future) is not None
        _finish(future)
        (row,) = _rows(tmp_path / "state.db")
        assert "pw-USERINFO" not in json.dumps(row) and "QUERYSECRET" not in json.dumps(row), row["route"]
        assert json.loads(row["route"])["endpoint"] == list(endpoint_identity(child.provider, child.base_url))
        assert "base_url" not in json.loads(row["route"])
    finally:
        for agent in reversed(agents):
            agent.close()
        handle.close()
    for path in tmp_path.glob("state.db*"):  # the store and its journal, as bytes
        blob = path.read_bytes()
        for secret in (b"pw-USERINFO", b"QUERYSECRET"):
            assert secret not in blob, (path.name, secret)
