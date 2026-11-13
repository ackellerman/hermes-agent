"""A profile's resumable limits are read from its own home, and reading them never creates one.

``_settings_of`` runs for every installed profile's store when the worktree sweep reads its pins. A profile
removed between the store listing and this read must not come back as an empty home.
"""

import json

from tools import delegate_tool_resumable as r


def test_a_missing_home_yields_the_defaults_and_nothing_is_created(tmp_path):
    gone = tmp_path / "profiles" / "gone"
    gone.parent.mkdir()

    assert r._settings_of(gone) == (r._DEFAULT_TTL_HOURS, r._DEFAULT_MAX_RESUMABLE)

    assert not gone.exists()
    assert list(gone.parent.iterdir()) == []


def test_a_file_where_the_home_should_be_yields_the_defaults_and_stays_a_file(tmp_path):
    not_a_home = tmp_path / "state"
    not_a_home.write_text("x")

    assert r._settings_of(not_a_home) == (r._DEFAULT_TTL_HOURS, r._DEFAULT_MAX_RESUMABLE)

    assert not_a_home.read_text() == "x"


def test_an_existing_home_still_reads_its_own_limits(tmp_path):
    home = tmp_path / "profiles" / "own"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(json.dumps({"delegation": {"resumable_ttl_hours": 5, "max_resumable": 3}}))

    assert r._settings_of(home) == (5.0, 3)
