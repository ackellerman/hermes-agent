"""Default-contract tests for the ``on_memory_evict`` ABC hook.

The hook is optional and must be backward-compatible (04): a provider that
never overrides it inherits a no-op default; a provider that overrides it is
invoked with keyword-only ``metadata``. These tests pin that contract on the
ABC itself.
"""

from agent.memory_provider import MemoryProvider


class _NonOverridingProvider(MemoryProvider):
    """Implements only the abstract core — never overrides on_memory_evict."""

    @property
    def name(self) -> str:
        return "non-overriding"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        pass

    def get_tool_schemas(self):
        return []

    def shutdown(self) -> None:
        pass


class _OverridingProvider(_NonOverridingProvider):
    def __init__(self) -> None:
        self.calls = []

    def on_memory_evict(self, content, target, *, metadata=None):
        self.calls.append((content, target, dict(metadata or {})))


def test_non_overriding_provider_inherits_noop_default():
    provider = _NonOverridingProvider()
    # The manager calls the hook with keyword metadata; the inherited
    # no-op default must accept it and be safe.
    result = provider.on_memory_evict("fact", "memory", metadata={"a": 1})
    assert result is None


def test_overriding_provider_receives_keyword_metadata():
    provider = _OverridingProvider()
    provider.on_memory_evict("fact", "memory", metadata={"eviction_reason": "cap"})
    provider.on_memory_evict("f2", "user")
    assert provider.calls == [
        ("fact", "memory", {"eviction_reason": "cap"}),
        ("f2", "user", {}),
    ]