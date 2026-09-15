"""SPEC-0047 W5 — AC-B1/B2 falsifiers: map schema hard gate + defensive slice.

AC-B1: a map-update LLM response carrying id-string edge endpoints
(``{"from": "review-d", ...}``) is REJECTED by ``_parse_map``; the prior map
persists untouched; the raise names the offending edge. Pre-fix, the malformed
map passed ``_parse_map`` (which only checked ``isinstance(dict)`` + covers).

AC-B2: ``slice()`` on an EXISTING map containing id-string edge endpoints
returns episodes/entities normally, drops the unparseable edges, and reports
``map_unparseable_edges >= 1`` without raising. Pre-fix it raised
``ValueError: invalid literal for int() with base 10: 'd'`` — the exact crash
that took down the first production run (session 20260915_024124_a8fa66).

The fixture ``compaction_map_malformed_edges.json`` captures the REAL failing
edge shape from that session's map (ids redacted to the failing shape).
"""

import json
from pathlib import Path

import pytest

from agent.compaction_map import CompactionMap, MapRegressionError

FIXTURE = Path(__file__).parent / "fixtures" / "compaction_map_malformed_edges.json"


def _good_map(end_msg=700):
    return {
        "schema_version": 1,
        "covers": {"start_msg": 0, "end_msg": end_msg},
        "episodes": [{"id": "ep0", "title": "work", "start_msg": 0, "end_msg": end_msg}],
        "entities": [{"id": "doc-work-db", "kind": "design-doc"}],
        "edges": [{"from": [3, 16], "to": [17, 30], "kind": "depends-on"}],
    }


def _malformed_response():
    """The exact failing shape: edges keyed by ENTITY ID strings."""
    return FIXTURE.read_text(encoding="utf-8")


class TestACB1SchemaRejectsIdStringEdges:
    def test_falsifier_malformed_update_rejected_prior_map_persists(self, tmp_path):
        """AC-B1 verbatim: feed the exact malformed response (id-string edges);
        _parse_map rejects it, update() raises naming the edge, and the prior
        map persists byte-identically."""
        cm = CompactionMap(tmp_path, "sess")
        prior = _good_map()
        cm.save(prior)
        prior_bytes = cm.path.read_bytes()

        with pytest.raises(CompactionMap.MapSchemaError) as excinfo:
            cm.update(lambda msgs: _malformed_response(), [],
                      start_msg=0, end_msg=699)
        # The reason NAMES the offending edge (not a bare "invalid map").
        msg = str(excinfo.value)
        assert "edges[" in msg, f"raise must name the offending edge: {msg}"
        assert ("review-d" in msg or "doc-pipeline" in msg), \
            f"raise must cite the offending id-string: {msg}"
        assert "NEVER entity ids" in msg or "entity" in msg.lower()
        # The prior map persists untouched.
        assert cm.path.read_bytes() == prior_bytes

    def test_malformed_map_rejected_at_parse_level(self):
        """Direct: _parse_map itself refuses the real-world malformed map."""
        with pytest.raises(CompactionMap.MapSchemaError):
            CompactionMap._parse_map(_malformed_response())

    def test_error_message_shows_the_wrong_endpoint(self):
        """The counter-example class: a string endpoint is named verbatim."""
        raw = '{"covers": {"start_msg": 0, "end_msg": 5}, "episodes": [], ' \
              '"entities": [], "edges": [{"from": "review-d", "to": [0, 1]}]}'
        with pytest.raises(CompactionMap.MapSchemaError) as excinfo:
            CompactionMap._parse_map(raw)
        assert "review-d" in str(excinfo.value)


class TestACB1CounterExamples:
    """Each malformed shape is rejected with a reason naming the item."""

    def test_id_string_edge_endpoint_rejected(self):
        raw = '{"covers": {"start_msg": 0, "end_msg": 5}, "episodes": [], ' \
              '"entities": [], "edges": [{"from": "doc-pipeline", "to": [0, 1]}]}'
        with pytest.raises(CompactionMap.MapSchemaError) as excinfo:
            CompactionMap._parse_map(raw)
        assert "doc-pipeline" in str(excinfo.value)

    def test_edge_missing_endpoint_rejected(self):
        raw = '{"covers": {"start_msg": 0, "end_msg": 5}, "episodes": [], ' \
              '"entities": [], "edges": [{"from": [0, 1], "kind": "refines"}]}'
        with pytest.raises(CompactionMap.MapSchemaError) as excinfo:
            CompactionMap._parse_map(raw)
        assert "edges[0]" in str(excinfo.value)

    def test_edge_outside_covers_rejected(self):
        raw = '{"covers": {"start_msg": 0, "end_msg": 5}, "episodes": [], ' \
              '"entities": [], "edges": [{"from": [0, 1], "to": [0, 99]}]}'
        with pytest.raises(CompactionMap.MapSchemaError) as excinfo:
            CompactionMap._parse_map(raw)
        assert "outside covers" in str(excinfo.value)

    def test_episode_bad_range_rejected(self):
        raw = '{"covers": {"start_msg": 0, "end_msg": 5}, ' \
              '"episodes": [{"start_msg": 9, "end_msg": 3}], "entities": [], "edges": []}'
        with pytest.raises(CompactionMap.MapSchemaError) as excinfo:
            CompactionMap._parse_map(raw)
        assert "episodes[0]" in str(excinfo.value)

    def test_episode_non_int_range_rejected(self):
        raw = '{"covers": {"start_msg": 0, "end_msg": 5}, ' \
              '"episodes": [{"start_msg": "episode-one", "end_msg": 3}], ' \
              '"entities": [], "edges": []}'
        with pytest.raises(CompactionMap.MapSchemaError) as excinfo:
            CompactionMap._parse_map(raw)
        assert "episodes[0]" in str(excinfo.value)

    def test_entity_empty_id_rejected(self):
        raw = '{"covers": {"start_msg": 0, "end_msg": 5}, "episodes": [], ' \
              '"entities": [{"id": ""}], "edges": []}'
        with pytest.raises(CompactionMap.MapSchemaError) as excinfo:
            CompactionMap._parse_map(raw)
        assert "entities[0]" in str(excinfo.value)

    def test_entity_non_string_id_rejected(self):
        raw = '{"covers": {"start_msg": 0, "end_msg": 5}, "episodes": [], ' \
              '"entities": [{"id": 7}], "edges": []}'
        with pytest.raises(CompactionMap.MapSchemaError) as excinfo:
            CompactionMap._parse_map(raw)
        assert "entities[0]" in str(excinfo.value)

    def test_covers_inverted_rejected(self):
        raw = '{"covers": {"start_msg": 9, "end_msg": 3}, "episodes": [], ' \
              '"entities": [], "edges": []}'
        with pytest.raises(CompactionMap.MapSchemaError):
            CompactionMap._parse_map(raw)

    def test_covers_missing_rejected(self):
        with pytest.raises(ValueError):
            CompactionMap._parse_map('{"episodes": [], "entities": [], "edges": []}')

    def test_map_update_prompt_carries_the_counter_example(self):
        """MAP_UPDATE_PROMPT must show the WRONG shape explicitly."""
        from agent.compaction_map import MAP_UPDATE_PROMPT
        assert '"from": "doc-pipeline"' in MAP_UPDATE_PROMPT
        assert "WRONG" in MAP_UPDATE_PROMPT
        assert "[start_msg, end_msg]" in MAP_UPDATE_PROMPT


class TestACB1CoercionAndPersistence:
    def test_numeric_strings_coerced(self):
        """Numeric strings ARE coerced ("7" -> 7); id-strings never are."""
        raw = '{"covers": {"start_msg": "0", "end_msg": "5"}, ' \
              '"episodes": [{"start_msg": "0", "end_msg": "5"}], ' \
              '"entities": [{"id": "doc"}], ' \
              '"edges": [{"from": ["0", "5"], "to": ["0", "5"]}]}'
        parsed = CompactionMap._parse_map(raw)  # numeric strings pass the gate
        # Downstream consumers int() these safely (slice/retire use int()).
        assert int(parsed["covers"]["end_msg"]) == 5
        assert int(parsed["episodes"][0]["end_msg"]) == 5

    def test_regression_check_still_binds_after_schema_pass(self, tmp_path):
        """A schema-VALID map that shrinks covers still hits the AC-3 gate."""
        cm = CompactionMap(tmp_path, "sess")
        cm.save(_good_map(end_msg=100))
        shrinking = json.dumps({
            "schema_version": 1,
            "covers": {"start_msg": 0, "end_msg": 60},
            "episodes": [{"id": "e", "start_msg": 0, "end_msg": 60}],
            "entities": [], "edges": [],
        })
        with pytest.raises(MapRegressionError):
            cm.update(lambda msgs: shrinking, [], start_msg=0, end_msg=99)
        assert cm.load()["covers"]["end_msg"] == 100

    def test_valid_update_still_lands(self, tmp_path):
        """The happy path is untouched: a conforming update persists."""
        cm = CompactionMap(tmp_path, "sess")
        cm.save(_good_map())
        good = json.dumps(_good_map(end_msg=720))
        new_map = cm.update(lambda msgs: good, [], start_msg=0, end_msg=719)
        assert new_map["covers"]["end_msg"] == 720
        assert cm.load()["covers"]["end_msg"] == 720


class TestACB2SliceSurvivesMalformedEdges:
    def _plant_real_fixture(self, tmp_path):
        cm = CompactionMap(tmp_path, "sess")
        cm.save(json.loads(_malformed_response()))
        return cm

    def test_falsifier_real_map_slice_no_raise_drops_bad_edges(self, tmp_path):
        """AC-B2 verbatim: slice() on the REAL malformed map returns episodes
        and entities, drops every unparseable edge, reports the count, and
        NEVER raises (pre-fix: ValueError 'd')."""
        cm = self._plant_real_fixture(tmp_path)
        result = cm.slice(0, 2_000_000_000)  # must not raise
        assert len(result["episodes"]) >= 10
        assert len(result["entities"]) >= 8
        assert result["edges"] == [], "all edges are id-string: all must drop"
        assert result["map_unparseable_edges"] >= 14, \
            f"expected >= 14 unparseable edges, got {result['map_unparseable_edges']}"
        # Same map against its own covers window: complete verdict intact.
        result2 = cm.slice(0, 700)
        assert result2["complete"] is True
        assert result2["map_unparseable_edges"] == result["map_unparseable_edges"]

    def test_slice_reports_mixed_parseability(self, tmp_path):
        """One good edge survives; each bad one is counted, none raises."""
        m = _good_map()
        m["edges"] = [
            {"from": [3, 16], "to": [17, 30], "kind": "depends-on"},   # good
            {"from": "review-d", "to": [17, 30], "kind": "refines"},   # bad from
            {"from": [3, 16], "to": "doc-work-db", "kind": "refines"},  # bad to
            {"from": [3, 16], "to": [3, 16], "kind": "refines"},        # in-window
            {"from": [900, 910], "to": [920, 930], "kind": "refines"},  # out of window
        ]
        cm = CompactionMap(tmp_path, "sess")
        cm.save(m)
        result = cm.slice(0, 100)
        assert [e.get("kind") for e in result["edges"]] == ["depends-on", "refines"]
        assert result["map_unparseable_edges"] == 2

    def test_slice_partial_string_endpoint_counts_unparseable(self, tmp_path):
        """A 'from' that parses but a 'to' that does not: the edge is
        unparseable as a whole (counted once), not half-kept."""
        m = _good_map()
        m["edges"] = [{"from": [3, 16], "to": "review-d", "kind": "refines"}]
        cm = CompactionMap(tmp_path, "sess")
        cm.save(m)
        result = cm.slice(0, 100)
        assert result["edges"] == []
        assert result["map_unparseable_edges"] == 1

    def test_slice_non_dict_edge_counted(self, tmp_path):
        m = _good_map()
        m["edges"] = ["not-an-edge", {"from": [3, 16], "to": [17, 30]}]
        cm = CompactionMap(tmp_path, "sess")
        cm.save(m)
        result = cm.slice(0, 100)
        assert len(result["edges"]) == 1
        assert result["map_unparseable_edges"] == 1

    def test_slice_clean_map_reports_zero(self, tmp_path):
        cm = CompactionMap(tmp_path, "sess")
        cm.save(_good_map())
        result = cm.slice(0, 100)
        assert result["map_unparseable_edges"] == 0
        assert len(result["edges"]) == 1

    def test_slice_on_empty_map(self, tmp_path):
        cm = CompactionMap(tmp_path, "sess")
        result = cm.slice(0, 100)
        assert result["covers"] == {"start_msg": 0, "end_msg": 0}
        assert result["complete"] is False
        assert result["edges"] == []
        assert result["map_unparseable_edges"] == 0

    def test_slice_numeric_string_endpoints_still_parse(self, tmp_path):
        """Numeric-string endpoints parse defensively (same coercion as D1)."""
        m = _good_map()
        m["edges"] = [{"from": ["3", "16"], "to": ["17", "30"], "kind": "refines"}]
        cm = CompactionMap(tmp_path, "sess")
        cm.save(m)
        result = cm.slice(0, 100)
        assert len(result["edges"]) == 1
        assert result["map_unparseable_edges"] == 0

    def test_retire_survives_malformed_edges(self, tmp_path):
        """The retire path must not crash on id-string edges either (same
        int(endpoint[-1]) trap); malformed edges drop, good ones beyond the
        retired window stay."""
        m = _good_map()
        m["edges"] = [
            {"from": [3, 16], "to": [60, 90], "kind": "depends-on"},  # survives
            {"from": "review-d", "to": [60, 90], "kind": "refines"},  # malformed -> drop
        ]
        cm = CompactionMap(tmp_path, "sess")
        cm.save(m)
        out = cm.retire(0, 30)
        assert len(out["edges"]) == 1
        assert out["edges"][0]["kind"] == "depends-on"


class TestPromptCounterExample:
    def test_update_prompt_warns_against_id_string_edges(self):
        """The D1 counter-example: the prompt shows the WRONG shape."""
        from agent.compaction_map import MAP_UPDATE_PROMPT
        assert "entity ids are NEVER valid endpoints" in MAP_UPDATE_PROMPT
        assert '"doc-pipeline"' in MAP_UPDATE_PROMPT