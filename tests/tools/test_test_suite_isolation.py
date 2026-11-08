"""Order-dependence guard for the suite itself: no test may patch a module it never imported.

The canonical runner gives every *file* a fresh interpreter, so a test that leans on
process-global state passes there and fails under a bare ``pytest tests/tools``. Measured on
one host: 61 failed serially, 19 under the runner — 42 of the extra ones order-dependent, and
19 of those share this one mechanism:

    monkeypatch.setattr("plugins.web.firecrawl.provider.Firecrawl", FirecrawlSDK)

pytest does **not** import that dotted path. ``monkeypatch.derive_importpath`` imports the
first segment and ``getattr``-walks the rest, so the patch only lands if something *earlier in
the same process* already imported the leaf. Under per-file isolation the test passes because
its own file imports it somewhere; in a shared process an earlier file leaves ``plugins`` bound
without ``web`` and the fixture dies at setup with

    AttributeError: module 'plugins' has no attribute 'web'

(seen in test_web_managed_routing — 11 errors — test_plugin_skills, test_web_keyless_rescue,
test_web_tools_config and test_telegram_send_message_video_metadata). ``plugins`` is where it
bites: its subpackages are loaded on demand by the plugin loader, so the parent attribute is
whatever the last importer left behind.

The guard is static, so it costs nothing. For every string patch target under ``plugins.``, the
test module containing it must import something inside that target's subpackage (``plugins.web``
for ``plugins.web.firecrawl.provider.Firecrawl``). Importing the subpackage — ``from
plugins.web.exa import provider`` or ``importlib.import_module("plugins.web.firecrawl.provider")``
— binds the attribute for real instead of hoping another test did.

``_KNOWN_UNFIXED`` is debt, not tolerance: one entry per (file, target) still relying on an
import side effect, outside the suite this guard was written for. An entry that goes stale
fails the test, so the list can only shrink.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS = REPO_ROOT / "tests"

#: ``(file, dotted target)`` still patching through an import side effect. Each one passes only
#: under per-file isolation; delete its entry in the same commit that imports the leaf.
_KNOWN_UNFIXED: frozenset[tuple[str, str]] = frozenset({
    ("tests/agent/test_memory_provider_init.py", "plugins.memory.load_memory_provider"),
    ("tests/agent/test_memory_user_id.py", "plugins.memory.load_memory_provider"),
    ("tests/agent/test_per_model_threshold_init_ordering.py", "plugins.context_engine.load_context_engine"),
    ("tests/agent/test_plugin_context_engine_clone.py", "plugins.context_engine.load_context_engine"),
    ("tests/agent/test_plugin_context_engine_init.py", "plugins.context_engine.load_context_engine"),
    ("tests/e2e/test_discord_adapter.py", "plugins.platforms.discord.adapter.cache_image_from_url"),
    ("tests/gateway/test_cron_fire_startup_gate.py", "plugins.cron_providers.chronos.verify.get_fire_verifier"),
    ("tests/gateway/test_cron_fire_webhook.py", "plugins.cron_providers.chronos.verify.get_fire_verifier"),
    ("tests/hermes_cli/test_cron_fire_dashboard.py", "plugins.cron_providers.chronos.verify.get_fire_verifier"),
    ("tests/hermes_cli/test_memory_setup.py", "plugins.memory.find_provider_dir"),
    ("tests/hermes_cli/test_plugins_cmd.py", "plugins.context_engine.discover_context_engines"),
    ("tests/hermes_cli/test_plugins_cmd.py", "plugins.context_engine.load_context_engine"),
    ("tests/hermes_cli/test_web_memory_provider_setup_install.py", "plugins.memory.find_provider_dir"),
    ("tests/pm/test_worker_publication.py", "plugins.memory.find_provider_dir"),
    ("tests/tui_gateway/test_session_agent_cwd.py", "plugins.memory.load_memory_provider"),
    # upstream test files that arrived with the 2026-11-08 sync
    ("tests/agent/test_fork_memory_under_checkpoint_gate.py", "plugins.memory.load_memory_provider"),
    ("tests/agent/test_memory_provider_caller_profile.py", "plugins.memory.load_memory_provider"),
    ("tests/tools/test_browser_cloud_provider_cache.py", "plugins.browser.browser_use.provider.BrowserUseBrowserProvider"),
    ("tests/tools/test_browser_cloud_provider_cache.py", "plugins.browser.browserbase.provider.BrowserbaseBrowserProvider"),
})


def _tree(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:  # a deliberately malformed fixture, not a test module
        return None


def _imported_prefixes(tree: ast.Module) -> set[str]:
    """Every dotted module name the file imports, including ``importlib.import_module("x")``."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Call):
            func = node.func
            called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if called in {"import_module", "__import__"} and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    names.add(first.value)
    return names


def _string_patch_targets(tree: ast.Module) -> list[tuple[str, int]]:
    """``(dotted target, line)`` for string targets under ``plugins.`` (the lazy package)."""
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if called not in {"setattr", "patch", "setitem", "delenv"}:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            if first.value.startswith("plugins."):
                found.append((first.value, node.lineno))
    return found


def _offenders() -> dict[tuple[str, str], list[int]]:
    found: dict[tuple[str, str], list[int]] = {}
    for path in sorted(TESTS.rglob("test_*.py")):
        tree = _tree(path)
        if tree is None:
            continue
        targets = _string_patch_targets(tree)
        if not targets:
            continue
        imported = _imported_prefixes(tree)
        relative = str(path.relative_to(REPO_ROOT))
        for dotted, line in targets:
            subpackage = ".".join(dotted.split(".")[:2])  # plugins.web
            if any(name == subpackage or name.startswith(subpackage + ".") for name in imported):
                continue
            found.setdefault((relative, dotted), []).append(line)
    return found


def test_no_test_patches_a_lazily_bound_subpackage_it_never_imports():
    offenders = _offenders()
    new = sorted(key for key in offenders if key not in _KNOWN_UNFIXED)
    assert not new, (
        "string patch targets that only resolve because an earlier test imported the module "
        "(they pass under the per-file runner and fail in a shared process). Import the leaf "
        "explicitly — importlib.import_module(<module>) — instead of relying on an import side "
        "effect:\n"
        + "\n".join(
            f"  {path}:{','.join(str(line) for line in offenders[(path, target)])} patches "
            f"{target!r} but never imports {'.'.join(target.split('.')[:2])!r}"
            for path, target in new
        )
    )


def test_the_known_unfixed_list_has_no_stale_entries():
    """A fixed file must drop its entry, so the debt list shrinks instead of rotting."""
    offenders = _offenders()
    stale = sorted(key for key in _KNOWN_UNFIXED if key not in offenders)
    assert not stale, f"these entries are fixed — delete them from _KNOWN_UNFIXED: {stale}"
