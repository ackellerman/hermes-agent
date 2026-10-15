"""``hermes profile install`` must work for a name whose profile was deleted.

``delete_profile`` leaves a tombstone so a stray ``logs/`` mkdir cannot resurrect the
name, and every profile-home write refuses a tombstoned home. A distribution install
under that name must therefore clear the tombstone itself — but only once the profile
is published, so a failed install neither resurrects the name nor leaves a partial
profile behind.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli.profile_distribution import (
    DistributionError,
    DistributionManifest,
    install_distribution,
    write_manifest,
)
from hermes_cli.profiles import (
    create_profile,
    delete_profile,
    get_profile_dir,
    list_profiles,
    profile_exists,
)
from hermes_constants import mark_named_profile_deleted, named_profile_is_deleted

NAME = "scholar"


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return tmp_path


def _distribution(root: Path, name: str = NAME) -> Path:
    staged = root / f"dist_{name}"
    staged.mkdir(parents=True, exist_ok=True)
    (staged / "SOUL.md").write_text("I am the distribution.\n")
    (staged / "config.yaml").write_text("model:\n  model: gpt-4\n")
    write_manifest(staged, DistributionManifest(name=name, version="0.1.0"))
    return staged


def _delete(name: str = NAME) -> None:
    with patch("hermes_cli.profiles._cleanup_gateway_service"), patch(
        "hermes_cli.profiles._stop_profile_backends"
    ):
        delete_profile(name, yes=True)


def _listed() -> list[str]:
    return [p.name for p in list_profiles() if not p.is_default]


def _staging_leftovers(profile_dir: Path) -> list[str]:
    return sorted(p.name for p in profile_dir.parent.glob(f".{profile_dir.name}.staging-*"))


def _assert_installed(profile_dir: Path) -> None:
    assert (profile_dir / "SOUL.md").read_text() == "I am the distribution.\n"
    assert not named_profile_is_deleted(profile_dir)
    assert profile_exists(NAME)
    assert NAME in _listed()
    assert _staging_leftovers(profile_dir) == []


def test_install_after_real_delete(profile_env):
    create_profile(NAME, no_alias=True, no_skills=True)
    _delete()
    profile_dir = get_profile_dir(NAME)
    assert named_profile_is_deleted(profile_dir)

    install_distribution(str(_distribution(profile_env)), name=NAME)

    _assert_installed(profile_dir)


def test_install_over_synthetic_tombstone(profile_env):
    profile_dir = get_profile_dir(NAME)
    mark_named_profile_deleted(profile_dir)

    install_distribution(str(_distribution(profile_env)), name=NAME)

    _assert_installed(profile_dir)


def test_partial_dir_with_tombstone_recovers_with_force(profile_env):
    """The leftover of a failed install under a tombstoned name: a partial dir + tombstone."""
    profile_dir = get_profile_dir(NAME)
    profile_dir.mkdir(parents=True)
    (profile_dir / "SOUL.md").write_text("partial\n")
    mark_named_profile_deleted(profile_dir)
    source = str(_distribution(profile_env))

    with pytest.raises(DistributionError, match="already exists"):
        install_distribution(source, name=NAME)
    install_distribution(source, name=NAME, force=True)

    _assert_installed(profile_dir)


def test_ghost_shell_with_tombstone_recovers_with_force(profile_env):
    """The tombstone's own design case: a ``logs/`` shell recreated after the delete."""
    create_profile(NAME, no_alias=True, no_skills=True)
    _delete()
    profile_dir = get_profile_dir(NAME)
    (profile_dir / "logs").mkdir(parents=True)
    assert named_profile_is_deleted(profile_dir)

    install_distribution(str(_distribution(profile_env)), name=NAME, force=True)

    _assert_installed(profile_dir)


def test_failure_after_staging_leaves_no_dir_and_keeps_tombstone(profile_env):
    create_profile(NAME, no_alias=True, no_skills=True)
    _delete()
    profile_dir = get_profile_dir(NAME)

    with patch(
        "hermes_cli.profile_distribution.write_manifest", side_effect=OSError("disk full")
    ), pytest.raises(OSError, match="disk full"):
        install_distribution(str(_distribution(profile_env)), name=NAME)

    assert not profile_dir.exists()
    assert _staging_leftovers(profile_dir) == []
    assert named_profile_is_deleted(profile_dir)
    assert not profile_exists(NAME)
    assert NAME not in _listed()


def test_fresh_name_unaffected(profile_env):
    profile_dir = get_profile_dir(NAME)
    assert not profile_dir.exists()
    assert not named_profile_is_deleted(profile_dir)

    install_distribution(str(_distribution(profile_env)), name=NAME)

    _assert_installed(profile_dir)
