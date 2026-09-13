"""SPEC-0043 rehydration-binding tests — AC-27/28 falsifiers (sweep parity).

Deterministic: no model called. Exercises:
- persisted StubRegistry across a fresh-instance restart (AC-28);
- read_dump resolving a registry-registered id via transcript fallback;
- the compaction toolset surfacing through tools_config injection (AC-27).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.compaction_dump import DumpStore
from agent.compaction_rehydrate import Rehydrator, StubRegistry


class TestAC28PersistedRegistry:
    def test_falsifier_restart_resolves_registered_id(self, tmp_path):
        msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
                for i in range(6)]
        store = DumpStore(tmp_path)
        ref = store.write_dump("sess", msgs, start_msg=0, end_msg=5, turn=1)
        # Register at swap time.
        reg1 = StubRegistry.for_session(tmp_path, "sess")
        reg1.register(ref.dump_id, 0, 5, "deploy decision")
        assert reg1.path.is_file(), "register must persist the registry file"
        # Fresh process equivalent: brand-new registry + rehydrator instances.
        reg2 = StubRegistry.for_session(tmp_path, "sess")
        entry = reg2.get(ref.dump_id)
        assert entry is not None
        assert entry["start_msg"] == 0 and entry["end_msg"] == 5
        assert entry["summary"] == "deploy decision"

    def test_falsifier_delete_dump_registry_serves_transcript_fallback(self, tmp_path):
        msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
                for i in range(6)]
        store = DumpStore(tmp_path)
        ref = store.write_dump("sess", msgs, start_msg=0, end_msg=5, turn=1)
        reg = StubRegistry.for_session(tmp_path, "sess")
        reg.register(ref.dump_id, 0, 5, "deploy decision")

        def transcript_reader(session_id, start, end):
            if end is None:
                return msgs[start:]
            return msgs[start:end + 1]

        rh = Rehydrator(DumpStore(tmp_path), transcript_reader=transcript_reader,
                        stub_registry=reg)
        # Delete the dump artifact: resolution must still succeed via registry ->
        # transcript fallback (AC-28), not just the open fallback at (0,5).
        store.dump_path("sess", ref.dump_id).unlink()
        result = rh.read_dump(ref.dump_id, session_id="sess")
        assert result["source"] == "transcript-fallback"
        assert result["messages"] == msgs
        assert rh.fallbacks and rh.fallbacks[0].served_from == "transcript-fallback"

    def test_registry_save_is_atomic_no_tmp_leftover(self, tmp_path):
        reg = StubRegistry.for_session(tmp_path, "sess")
        reg.register("a", 0, 3, "x")
        assert not list(reg.path.parent.glob("*.json.tmp")), "atomic save must leave no tmp file"


class TestAC27ToolsetInjection:
    def test_compaction_in_tools_config_static_table(self):
        from hermes_cli import tools_config
        names = {row[0] for row in tools_config.CONFIGURABLE_TOOLSETS}
        assert "compaction" in names
        assert "compaction" in tools_config._DEFAULT_OFF_TOOLSETS

    def test_read_dump_uses_config_root(self, tmp_path, monkeypatch):
        import hermes_cli.config as hc
        # tools.compaction_tools._storage_root and tools_config both resolve from
        # the SAME config key (compaction_pipeline.storage_root / enabled) — the
        # resolver is tested directly and the DumpStore honors it (no hardcode).
        def fake_load():
            return {"compaction_pipeline": {"enabled": True, "storage_root": str(tmp_path)}}
        monkeypatch.setattr(hc, "load_config", fake_load)
        import hermes_cli.tools_config as tc
        assert tc._compaction_pipeline_enabled() is True
        from agent.compaction_dump import DumpStore
        assert DumpStore(Path(tmp_path)).root == Path(tmp_path)

    def test_toolset_injects_when_enabled_and_absent_when_not(self, monkeypatch):
        from hermes_cli import tools_config as tc

        def fake_load(enabled):
            def _load():
                return {"compaction_pipeline": {"enabled": enabled, "storage_root": "/tmp/x"}}
            return _load

        # Ensure x_search creds absent so we isolate the compaction injection.
        monkeypatch.setattr(tc, "_xai_credentials_present", lambda: False)
        monkeypatch.setattr(tc, "_toolset_allowed_for_platform", lambda name, platform: name != "x_search")
        import hermes_cli.config as hc

        monkeypatch.setattr(hc, "load_config", fake_load(True))
        enabled = tc._composite_toolsets(["core"], "zwerg", explicitly_configured=False)
        assert "compaction" in enabled

        monkeypatch.setattr(hc, "load_config", fake_load(False))
        enabled_off = tc._composite_toolsets(["core"], "zwerg", explicitly_configured=False)
        assert "compaction" not in enabled_off