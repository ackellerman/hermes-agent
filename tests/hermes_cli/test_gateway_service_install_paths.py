"""Tests for the gateway unit's LD_LIBRARY_PATH install-path filter.

The filter - drop components inside an install's ``installs/<key>/environments`` generation, keep
everything else - lives in :mod:`hermes_cli.gateway_install_paths`, split out of
``hermes_cli.gateway`` so the facade does not grow. These are its unit tests and the end-to-end
unit-generation tests that read the rendered ``Environment=`` line.
"""

import logging
from pathlib import Path

import pytest

import hermes_cli.gateway as gateway_cli
from hermes_cli import gateway_install_paths


def _venv_generation_lib(installs_root: Path, install_key: str, generation: str = "9fff708538934e1299733edf9ef711c2") -> str:
    """A lib dir inside a dependency generation: the next sync deletes it (#14613's failure)."""
    return str(
        Path(installs_root)
        / install_key
        / "environments"
        / generation
        / "venv/lib/python3.14/site-packages/nvidia/cublas/lib"
    )


class TestUnitDropsInstallEnvironmentLdLibraryPaths:
    """A regenerated unit must not carry a dependency generation's lib dir: ``hermes update``
    replaces the generation, so the path is dead the moment CUDA needs it -- and once baked in,
    every start/restart/status regeneration keeps it (stt-cuda-survives-update, element e3)."""

    def _user_unit(self, monkeypatch, tmp_path, hermes_home: Path) -> str:
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        # The absent-env branch falls back to the installed unit: keep the host's real one out.
        monkeypatch.setattr(
            gateway_cli, "get_systemd_unit_path", lambda system=False: tmp_path / "hermes-gateway.service"
        )
        return gateway_cli.generate_systemd_unit(system=False)

    def test_user_unit_drops_environment_components_built_through_the_default_root(
        self, monkeypatch, tmp_path, caplog
    ):
        from pm.environments import installs_root

        hermes_home = tmp_path / ".hermes"
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        root = installs_root()
        assert root == hermes_home / "installs"

        stale = _venv_generation_lib(root, "9e8f535ea094ebf3")
        # Merely missing on disk: an unmounted volume is not stale, so it stays.
        missing = str(tmp_path / "mnt" / "unmounted-cuda" / "lib")
        keep = "/opt/cuda/lib64"
        monkeypatch.setenv("LD_LIBRARY_PATH", ":".join([stale, keep, missing]))

        with caplog.at_level(logging.INFO, logger="hermes_cli.gateway"):
            unit = self._user_unit(monkeypatch, tmp_path, hermes_home)

        assert not Path(missing).exists()
        assert "environments" not in unit
        assert stale not in unit
        # The surviving value still comes from the installer's inherited env, order intact.
        assert f'Environment="LD_LIBRARY_PATH={keep}:{missing}"' in unit
        drops = [r for r in caplog.records if stale in r.getMessage()]
        assert drops, [r.getMessage() for r in caplog.records]
        assert drops[0].levelno == logging.INFO
        assert "environments" in drops[0].getMessage()

    def test_user_unit_drops_environment_components_built_through_a_custom_root(self, monkeypatch, tmp_path):
        from pm.environments import installs_root

        custom_home = tmp_path / "srv" / "hermes-data"
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        monkeypatch.setenv("HERMES_HOME", str(custom_home))
        root = installs_root()
        assert root == custom_home / "installs"

        stale = _venv_generation_lib(root, "0d5a1c9b7e3f4a26")
        keep = "/opt/cuda/lib64"
        missing = str(tmp_path / "mnt" / "gone" / "lib")
        monkeypatch.setenv("LD_LIBRARY_PATH", ":".join([keep, stale, missing]))

        unit = self._user_unit(monkeypatch, tmp_path, custom_home)

        assert stale not in unit
        assert f'Environment="LD_LIBRARY_PATH={keep}:{missing}"' in unit

    def test_user_unit_drops_environment_components_from_the_installed_unit_fallback(
        self, monkeypatch, tmp_path
    ):
        from pm.environments import installs_root

        hermes_home = tmp_path / ".hermes"
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        stale = _venv_generation_lib(installs_root(), "9e8f535ea094ebf3")
        keep = "/opt/cuda/lib64"
        unit_path = tmp_path / "hermes-gateway.service"
        unit_path.write_text(
            f'[Service]\nEnvironment="LD_LIBRARY_PATH={stale}:{keep}"\n', encoding="utf-8"
        )
        monkeypatch.setattr(gateway_cli, "get_systemd_unit_path", lambda system=False: unit_path)
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)

        unit = gateway_cli.generate_systemd_unit(system=False)

        assert stale not in unit
        assert f'Environment="LD_LIBRARY_PATH={keep}"' in unit

    def test_user_unit_drops_the_line_when_every_component_is_an_environment(
        self, monkeypatch, tmp_path
    ):
        from pm.environments import installs_root

        hermes_home = tmp_path / ".hermes"
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv(
            "LD_LIBRARY_PATH",
            ":".join(
                [
                    _venv_generation_lib(installs_root(), "9e8f535ea094ebf3"),
                    _venv_generation_lib(installs_root(), "0d5a1c9b7e3f4a26"),
                ]
            ),
        )

        unit = self._user_unit(monkeypatch, tmp_path, hermes_home)

        assert "LD_LIBRARY_PATH" not in unit

    def _system_unit(self, monkeypatch, tmp_path, target_home: Path) -> str:
        """``generate_systemd_unit(system=True)`` with the target user faked to *target_home*."""
        monkeypatch.setattr(
            gateway_cli,
            "_system_service_identity",
            lambda run_as_user=None: ("alice", "alice", str(target_home), 1001),
        )
        monkeypatch.setattr(gateway_cli, "_build_service_path_dirs", list)
        monkeypatch.setattr(gateway_cli, "_build_user_local_paths", lambda home, existing: [])
        monkeypatch.setattr(gateway_cli.shutil, "which", lambda name: None)
        monkeypatch.setattr(
            gateway_cli, "get_systemd_unit_path", lambda system=False: tmp_path / "hermes-gateway.service"
        )
        return gateway_cli.generate_systemd_unit(system=True, run_as_user="alice")

    @pytest.mark.parametrize("profile_suffix", ["", "profiles/p"], ids=["native", "profile"])
    def test_system_unit_drops_environment_components_for_the_target_users_home(
        self, monkeypatch, tmp_path, profile_suffix
    ):
        """Under sudo the caller's install root is remapped to the target user's; the filter must
        follow that remap instead of comparing against the caller's ``installs_root()``. A profile
        HERMES_HOME lands on the same native root as the plain one."""
        caller_home = tmp_path / "root"
        target_home = tmp_path / "alice"
        caller_home.mkdir()
        target_home.mkdir()
        monkeypatch.setattr(Path, "home", staticmethod(lambda: caller_home))
        monkeypatch.setenv("HERMES_HOME", str(caller_home / ".hermes" / profile_suffix))

        target_installs = target_home / ".hermes" / "installs"
        caller_installs = caller_home / ".hermes" / "installs"
        stale_target = _venv_generation_lib(target_installs, "9e8f535ea094ebf3")
        remapped_caller = _venv_generation_lib(caller_installs, "9e8f535ea094ebf3")
        keep = "/opt/cuda/lib64"
        missing = str(tmp_path / "mnt" / "gone" / "lib")
        monkeypatch.setenv("LD_LIBRARY_PATH", ":".join([remapped_caller, stale_target, keep, missing]))

        unit = self._system_unit(monkeypatch, tmp_path, target_home)

        assert "environments" not in unit
        assert f'Environment="LD_LIBRARY_PATH={keep}:{missing}"' in unit

    def test_system_unit_drops_environment_components_for_a_subdirectory_hermes_home(
        self, monkeypatch, tmp_path
    ):
        """A HERMES_HOME that is a non-``profiles`` subdirectory of the target's ``~/.hermes`` still
        resolves to the native home: pm writes generations under ``<home>/.hermes/installs``, not
        under ``<home>/.hermes/subdir/installs``, so the target's root has to be anchored on the
        native home rather than on the path's shape."""
        caller_home = tmp_path / "root"
        target_home = tmp_path / "alice"
        caller_home.mkdir()
        target_home.mkdir()
        monkeypatch.setattr(Path, "home", staticmethod(lambda: caller_home))
        monkeypatch.setenv("HERMES_HOME", str(caller_home / ".hermes" / "subdir"))

        stale_target = _venv_generation_lib(target_home / ".hermes" / "installs", "9e8f535ea094ebf3")
        keep = "/opt/cuda/lib64"
        monkeypatch.setenv("LD_LIBRARY_PATH", f"{stale_target}:{keep}")

        unit = self._system_unit(monkeypatch, tmp_path, target_home)

        assert stale_target not in unit
        assert f'Environment="LD_LIBRARY_PATH={keep}"' in unit

    def test_system_unit_drops_a_component_under_the_callers_own_installs_root(
        self, monkeypatch, tmp_path
    ):
        """The remap only rewrites components under ``Path.home()``, so a caller whose installs root
        sits outside its HOME (the account home while HOME is a harness dir) keeps a component the
        remap cannot touch: the caller's root stays a filter candidate of its own."""
        import pm.environments as pm_environments

        caller_home = tmp_path / "root"
        target_home = tmp_path / "alice"
        caller_home.mkdir()
        target_home.mkdir()
        monkeypatch.setattr(Path, "home", staticmethod(lambda: caller_home))
        monkeypatch.setenv("HERMES_HOME", str(caller_home / ".hermes"))
        caller_store = tmp_path / "account-store" / "installs"
        monkeypatch.setattr(pm_environments, "installs_root", lambda: caller_store)

        caller_stale = _venv_generation_lib(caller_store, "0d5a1c9b7e3f4a26")
        target_stale = _venv_generation_lib(target_home / ".hermes" / "installs", "9e8f535ea094ebf3")
        keep = "/opt/cuda/lib64"
        monkeypatch.setenv("LD_LIBRARY_PATH", ":".join([caller_stale, target_stale, keep]))

        roots = gateway_install_paths._ld_library_path_install_roots(system=True, target_home_dir=str(target_home))
        assert roots == [caller_store, target_home / ".hermes" / "installs"]

        unit = self._system_unit(monkeypatch, tmp_path, target_home)

        assert caller_stale not in unit
        assert target_stale not in unit
        assert f'Environment="LD_LIBRARY_PATH={keep}"' in unit

    def test_system_unit_drops_environment_components_under_a_symlinked_hermes_home(
        self, monkeypatch, tmp_path
    ):
        """A ``HERMES_HOME`` that is a symlink to the target's ``~/.hermes`` -- and a caller whose
        HOME is outside that tree, so the remap keeps the path verbatim -- is still the target's
        data root: pm writes the generations under ``<target>/.hermes/installs``. The containment
        test has to resolve both sides as ``hermes_constants._hermes_root_of`` does, or a lexical
        one filters against ``<link>/installs`` and the stale generation survives."""
        caller_home = tmp_path / "root"
        target_home = tmp_path / "alice"
        caller_home.mkdir()
        target_hermes = target_home / ".hermes"
        target_hermes.mkdir(parents=True)
        link = tmp_path / "hermes-link"
        link.symlink_to(target_hermes, target_is_directory=True)
        monkeypatch.setattr(Path, "home", staticmethod(lambda: caller_home))
        monkeypatch.setenv("HERMES_HOME", str(link))

        stale_target = _venv_generation_lib(target_hermes / "installs", "9e8f535ea094ebf3")
        keep = "/opt/cuda/lib64"
        monkeypatch.setenv("LD_LIBRARY_PATH", f"{stale_target}:{keep}")

        unit = self._system_unit(monkeypatch, tmp_path, target_home)

        assert stale_target not in unit
        assert "environments" not in unit
        assert f'Environment="LD_LIBRARY_PATH={keep}"' in unit
        # Self-checking: the target's root is a filter candidate, not just absent from the unit.
        roots = gateway_install_paths._ld_library_path_install_roots(system=True, target_home_dir=str(target_home))
        assert target_hermes / "installs" in roots

    def test_the_environments_dir_itself_and_lookalikes_are_kept(self, monkeypatch, tmp_path):
        """The len-2 boundary: only a path *inside* ``<root>/<key>/environments`` is a generation.
        The dir itself and lookalike names are kept."""
        from pm.environments import installs_root

        hermes_home = tmp_path / ".hermes"
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        root = installs_root()
        boundary = str(root / "9e8f535ea094ebf3" / "environments")
        lookalike = str(root / "9e8f535ea094ebf3" / "environments-backup" / "9fff708538934e1" / "lib")
        other_dir = str(root / "9e8f535ea094ebf3" / "test-environment" / "lib")
        monkeypatch.setenv("LD_LIBRARY_PATH", ":".join([boundary, lookalike, other_dir]))

        unit = self._user_unit(monkeypatch, tmp_path, hermes_home)

        assert f'Environment="LD_LIBRARY_PATH={boundary}:{lookalike}:{other_dir}"' in unit


class TestRemapPathForUser:
    """Unit tests for _remap_path_for_user()."""

    def test_remaps_path_under_current_home(self, monkeypatch, tmp_path):
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "root")
        (tmp_path / "root").mkdir()
        result = gateway_cli._remap_path_for_user(
            str(tmp_path / "root" / ".hermes" / "hermes-agent"),
            str(tmp_path / "alice"),
        )
        assert result == str(tmp_path / "alice" / ".hermes" / "hermes-agent")
