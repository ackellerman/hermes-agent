"""Behavior tests for the built-in memory → external provider bridge.

The bridge lives behind the MemoryManager interface
(``MemoryManager.notify_memory_tool_write``): the agent loop hands over the raw
built-in memory tool result + args, and the manager decides whether/what to
mirror to external providers. These tests drive that method with a fake
external provider and assert which ``on_memory_write`` calls land.
"""

import json

import pytest

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider


class _RecordingProvider(MemoryProvider):
    """Minimal external provider that records on_memory_write calls."""

    def __init__(self) -> None:
        self.calls = []
        self.evict_calls = []

    @property
    def name(self) -> str:
        return "recording"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        pass

    def get_tool_schemas(self):
        return []

    def shutdown(self) -> None:
        pass

    def on_memory_write(self, action, target, content, metadata=None):
        self.calls.append({
            "action": action,
            "target": target,
            "content": content,
            "metadata": dict(metadata or {}),
        })

    def on_memory_evict(self, content, target, *, metadata=None):
        self.evict_calls.append({
            "content": content,
            "target": target,
            "metadata": dict(metadata or {}),
        })


def _manager_with_provider():
    mgr = MemoryManager()
    provider = _RecordingProvider()
    mgr.add_provider(provider)
    return mgr, provider


def test_notifies_remove_with_old_text_after_success():
    mgr, provider = _manager_with_provider()
    mgr.notify_memory_tool_write(
        json.dumps({"success": True}),
        {"action": "remove", "target": "memory", "old_text": "stale preference entry"},
    )
    assert provider.calls == [
        {
            "action": "remove",
            "target": "memory",
            "content": "",
            "metadata": {"old_text": "stale preference entry"},
        }
    ]






@pytest.mark.parametrize("tool_result", [None, [], object(), "not-json"])
def test_skips_unrecognized_tool_result_shape(tool_result):
    mgr, provider = _manager_with_provider()
    mgr.notify_memory_tool_write(
        tool_result,
        {"action": "add", "target": "memory", "content": "new fact"},
    )
    assert provider.calls == []






def test_build_metadata_callback_is_merged_per_op():
    mgr, provider = _manager_with_provider()
    mgr.notify_memory_tool_write(
        json.dumps({"success": True}),
        {"action": "add", "target": "memory", "content": "fact"},
        build_metadata=lambda: {"session_id": "s1", "tool_name": "memory"},
    )
    assert provider.calls == [
        {
            "action": "add",
            "target": "memory",
            "content": "fact",
            "metadata": {"session_id": "s1", "tool_name": "memory"},
        }
    ]


class _RaisingEvictProvider(_RecordingProvider):
    """External provider whose on_memory_evict raises — must not propagate."""

    def on_memory_evict(self, content, target, *, metadata=None):
        raise RuntimeError("evict boom")


class _BuiltinProvider(_RecordingProvider):
    @property
    def name(self) -> str:
        return "builtin"


# --- Eviction fan-out (terminal dropped-fact shape) -------------------------


def test_evict_fires_on_terminal_done_for_add():
    mgr, provider = _manager_with_provider()
    mgr.notify_memory_tool_write(
        json.dumps({"success": False, "done": True, "error": "consolidation failed"}),
        {"action": "add", "target": "memory", "content": "would-be-dropped fact"},
    )
    # terminal failure is NOT mirrored as a successful write...
    assert provider.calls == []
    # ...but the would-be-dropped fact is handed to the eviction path.
    assert provider.evict_calls == [
        {
            "content": "would-be-dropped fact",
            "target": "memory",
            "metadata": {"eviction_reason": "consolidation failed"},
        }
    ]


def test_evict_fires_for_batched_replace_and_add_with_old_text():
    mgr, provider = _manager_with_provider()
    mgr.notify_memory_tool_write(
        {"success": False, "done": True, "error": "at capacity"},
        {
            "operations": [
                {"action": "replace", "content": "updated fact", "old_text": "old"},
                {"action": "add", "content": "new fact"},
            ]
        },
        build_metadata=lambda: {"session_id": "s1", "tool_name": "memory"},
    )
    assert provider.evict_calls == [
        {
            "content": "updated fact",
            "target": "memory",
            "metadata": {
                "session_id": "s1",
                "tool_name": "memory",
                "eviction_reason": "at capacity",
                "old_text": "old",
            },
        },
        {
            "content": "new fact",
            "target": "memory",
            "metadata": {
                "session_id": "s1",
                "tool_name": "memory",
                "eviction_reason": "at capacity",
            },
        },
    ]


def test_evict_never_fires_for_remove():
    mgr, provider = _manager_with_provider()
    mgr.notify_memory_tool_write(
        {"success": False, "done": True, "error": "boom"},
        {"action": "remove", "target": "memory", "old_text": "stale"},
    )
    # a failed remove leaves the store unchanged — nothing was dropped.
    assert provider.evict_calls == []


@pytest.mark.parametrize("tool_result", [
    {"success": False},                  # no `done` — transient/unknown
    {"success": True, "staged": True},   # staged for approval — not a commit
    {"success": False, "done": False},   # explicit non-terminal failure
    None,
    [],
    object(),
    "not-json",
])
def test_evict_skips_non_terminal_tool_result_shapes(tool_result):
    mgr, provider = _manager_with_provider()
    mgr.notify_memory_tool_write(
        tool_result,
        {"action": "add", "target": "memory", "content": "fact"},
    )
    assert provider.evict_calls == []


def test_evict_exception_is_contained_and_loop_continues():
    mgr = MemoryManager()
    raising = _RaisingEvictProvider()
    recording = _RecordingProvider()
    # bypass the one-external-provider limit to exercise per-provider fault
    # isolation across two non-builtin providers.
    mgr._providers.extend([raising, recording])
    mgr.notify_memory_tool_write(
        {"success": False, "done": True, "error": "cap"},
        {"action": "add", "content": "fact", "target": "memory"},
    )
    # the raising provider's exception is swallowed; the recording provider
    # still received its eviction call.
    assert raising.evict_calls == []
    assert recording.evict_calls == [
        {"content": "fact", "target": "memory", "metadata": {"eviction_reason": "cap"}}
    ]


def test_evict_skips_builtin_provider():
    mgr = MemoryManager()
    builtin = _BuiltinProvider()
    mgr.add_provider(builtin)
    mgr.notify_memory_tool_write(
        {"success": False, "done": True, "error": "cap"},
        {"action": "add", "content": "fact", "target": "memory"},
    )
    assert builtin.evict_calls == []

@pytest.mark.parametrize('batch,operations,previous', [
    (False, [{'action': 'remove', 'old_text': 'Prefers tea'}], ['Prefers tea']),
    (False, [{'action': 'replace', 'old_text': 'Prefers tea', 'content': 'Prefers coffee'}], ['Prefers tea']),
    (True, [{'action': 'remove', 'old_text': 'Prefers tea'}], ['Prefers tea']),
    (True, [{'action': 'replace', 'old_text': 'Prefers tea', 'new_text': 'Prefers coffee'}], ['Prefers tea']),
    (True, [
        {'action': 'add', 'content': 'Uses the blue notebook'},
        {'action': 'replace', 'old_text': 'blue notebook', 'new_text': 'Uses the green notebook'},
        {'action': 'remove', 'old_text': 'green notebook'},
    ], [None, 'Uses the blue notebook', 'Uses the green notebook']),
])
def test_committed_entry_identity_comes_from_locked_store(
    tmp_path, monkeypatch, batch, operations, previous
):
    target = 'memory'
    from contextlib import contextmanager
    from tools import memory_tool_store
    from tools.memory_tool import MemoryStore, memory_tool

    monkeypatch.setattr('tools.memory_tool.get_memory_dir', lambda: tmp_path)
    store = MemoryStore()
    store.load_from_disk()
    store.add(target, 'Prefers tea')
    store.add(target, 'Prefers tea with milk')
    manager, provider = _manager_with_provider()
    locked = False
    lock = store._file_lock
    match = memory_tool_store._find_unique_match

    @contextmanager
    def checked_lock(path):
        nonlocal locked
        with lock(path):
            locked = True
            try:
                yield
            finally:
                locked = False

    def checked_match(entries, old_text):
        assert locked
        assert provider.calls == []
        return match(entries, old_text)

    monkeypatch.setattr(store, '_file_lock', checked_lock)
    monkeypatch.setattr(memory_tool_store, '_find_unique_match', checked_match)
    args = {'target': target, **({'operations': operations} if batch else operations[0])}
    result = memory_tool(store=store, **args)
    assert json.loads(result)['success'] is True
    assert not locked
    assert provider.calls == []
    reloaded = MemoryStore()
    reloaded.load_from_disk()
    assert reloaded._entries_for(target) == store._entries_for(target)

    manager.notify_memory_tool_write(result, args, build_metadata=lambda: {'session_id': 'test'})
    assert [call['action'] for call in provider.calls] == [op['action'] for op in operations]
    assert [call['metadata'].get('previous_content') for call in provider.calls] == previous
    assert all(call['metadata']['session_id'] == 'test' for call in provider.calls)
    assert 'Prefers tea with milk' in store._entries_for(target)


def test_previous_content_cannot_come_from_uncommitted_arguments():
    manager, provider = _manager_with_provider()
    args = {'action': 'remove', 'old_text': 'partial', 'previous_content': 'Untrusted argument'}
    manager.notify_memory_tool_write(
        {'success': True}, args, build_metadata=lambda: {'previous_content': 'Uncommitted metadata'}
    )
    assert provider.calls[0]['metadata'] == {'old_text': 'partial'}
