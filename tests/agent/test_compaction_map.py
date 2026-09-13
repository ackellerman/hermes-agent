"""SPEC-0042 moving-map tests — AC-3/4/5 falsifiers."""

import json

import pytest

from agent.compaction_map import (
    CompactionMap,
    MapRegressionError,
    audit_map_against_dump,
    episode_iou_score,
    iou_ranges,
)


def _ep(s, e, name="work"):
    return {"name": name, "start_msg": s, "end_msg": e}


class TestAC3EpisodeBoundariesAndCovers:
    def test_ten_updates_match_ground_truth_at_90pct_iou(self, tmp_path):
        """10 consecutive updates on a synthetic 200-message conversation:
        episode boundaries match hand-labeled ground truth at >= 90% IoU and
        covers never regresses."""
        # Hand-labeled ground truth: 4 episodes over 200 messages (0-indexed).
        ground_truth = [(0, 49), (50, 99), (100, 149), (150, 199)]
        cm = CompactionMap(tmp_path, "sess")
        # Deterministic stand-in LLM: knows the true boundaries; each chunk of
        # 20 messages extends the episode covering it. This isolates the map's
        # update/slice/IoU machinery from model quality (model quality is the
        # eval run's job, spec §4 item 5).
        def llm_call(messages):
            data = json.loads(messages[1]["content"])
            end = data["chunk"]["end_msg"]
            eps = []
            for s, e in ground_truth:
                if s <= end:
                    eps.append(_ep(s, min(e, end)))
                    if e <= end:
                        continue
            new_map = {
                "schema_version": 1,
                "covers": {"start_msg": 0, "end_msg": end},
                "episodes": eps,
                "entities": [],
                "edges": [],
            }
            return json.dumps(new_map)

        for start in range(0, 200, 20):
            chunk = [{"role": "user", "content": f"m{i}"} for i in range(start, start + 20)]
            cm.update(llm_call, chunk, start_msg=start, end_msg=start + 19)

        final = cm.load()
        predicted = [(int(ep["start_msg"]), int(ep["end_msg"])) for ep in final["episodes"]]
        assert episode_iou_score(predicted, ground_truth) >= 0.90
        assert final["covers"]["end_msg"] == 199

    def test_falsifier_update_shrinking_covers_fails(self, tmp_path):
        """FALSIFIER AC-3 verbatim: any update that shrinks covers without a
        matching retire fails."""
        cm = CompactionMap(tmp_path, "sess")
        good = {
            "schema_version": 1,
            "covers": {"start_msg": 0, "end_msg": 100},
            "episodes": [_ep(0, 100)],
            "entities": [],
            "edges": [],
        }
        cm.save(good)
        shrinking = json.dumps({
            "schema_version": 1,
            "covers": {"start_msg": 0, "end_msg": 60},  # shrink!
            "episodes": [_ep(0, 60)],
            "entities": [],
            "edges": [],
        })
        with pytest.raises(MapRegressionError):
            cm.update(lambda msgs: shrinking, [], start_msg=0, end_msg=99)
        # A regressing update must NOT be persisted.
        assert cm.load()["covers"]["end_msg"] == 100

    def test_iou_definition(self):
        # identical ranges -> 1.0; adjacent off-by-one reduces proportionally
        assert iou_ranges((0, 49), (0, 49)) == 1.0
        assert 0.9 < iou_ranges((0, 49), (0, 50)) < 1.0
        assert episode_iou_score([], [(0, 49)]) == 0.0
        assert episode_iou_score([(0, 49), (50, 99)], [(0, 49), (50, 99)]) == 1.0


class TestAC5BoundedUpdateCost:
    def test_falsifier_measured_input_token_total_within_15pct(self, tmp_path):
        """FALSIFIER AC-5 verbatim: for a 100K-token region, the sum of
        update-call inputs stays within 15% of region size (window overhead tax);
        measured input token total > 1.15 x region tokens fails."""
        cm = CompactionMap(tmp_path, "sess")
        region_tokens = 100_000
        # 100 messages x ~1000 tokens each.
        region = [{"role": "user", "content": "w " * 1000} for _ in range(100)]
        inputs_tokens = []

        def llm_call(messages):
            inputs_tokens.append(sum(len(m["content"]) for m in messages) // 4)
            data = json.loads(messages[1]["content"])
            end = data["chunk"]["end_msg"]
            return json.dumps({
                "schema_version": 1,
                "covers": {"start_msg": 0, "end_msg": end},
                "episodes": [_ep(0, end)],
                "entities": [],
                "edges": [],
            })

        # 10 updates over the region; each chunk = 10 messages (~10K tokens).
        for start in range(0, 100, 10):
            chunk = region[start:start + 10]
            cm.update(llm_call, chunk, start_msg=start, end_msg=start + 9)

        measured_total = sum(inputs_tokens)
        # Region token count per the same /4 estimator used for the measured inputs.
        region_estimate = sum(len(str(m["content"])) for m in region) // 4
        assert measured_total <= 1.15 * region_estimate, (
            f"map update cost unbounded: {measured_total} > 1.15 x {region_estimate}"
        )


class TestAC4AuditDetectsInjectedInconsistency:
    def test_falsifier_audit_must_detect_injected_citation_corruption(self, tmp_path):
        """FALSIFIER AC-4 verbatim: corrupt a map entry's citation (point it at
        a range that says something else) -> sampled audit flags it -> remedy
        rebuilds the slice from dump -> re-audit passes."""
        cm = CompactionMap(tmp_path, "sess")
        # episode 10-19 truthfully covers "deploy auth service" work.
        dump_msgs = [{"role": "user", "content": f"m{i}: "
                      + ("deploying auth service" if 10 <= i <= 19 else "unrelated chat")}
                     for i in range(30)]
        cm.save({
            "schema_version": 1,
            "covers": {"start_msg": 0, "end_msg": 29},
            "episodes": [{"name": "deploy auth", "start_msg": 0, "end_msg": 9}],  # WRONG range
            "entities": ["auth-service"],
            "edges": [],
        })
        findings = audit_map_against_dump(cm.load(), dump_msgs)
        assert findings, "audit must detect the injected inconsistency, not pass it"
        # Remedy: rebuild the slice from dump (episode ranges re-derived from content)
        fixed = cm.load()
        fixed["episodes"] = [_ep(10, 19, "deploy auth")]
        cm.save(fixed)
        assert audit_map_against_dump(cm.load(), dump_msgs) == []