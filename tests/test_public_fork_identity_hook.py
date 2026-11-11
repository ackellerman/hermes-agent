"""Behaviour contract for the fork pre-push identity gate (`.githooks/pre-push` + helper).

Every push here is a real `git push` into a disposable bare remote under a pytest tmp dir,
through a hook copied from this checkout's `.githooks/` into a disposable clone, with a
synthetic private policy bundle in that clone's common dir. Refusals are asserted on the push
status, on the remote's refs (which must not move) and on a reason phrase that only the gate
prints for that reason. The phrase is searched in the hook's own lines only: git's own
`error:`/`hint:`/`To` lines and the disposable root path are removed first, and the roots
have neutral names, so a path or a test name can never satisfy a needle.

The identity gate fails closed for every non-`sim` author or committer at this checkpoint:
verified-upstream snapshots and reviewed cherry-picks are later elements. All deny-list values
here are synthetic (`Pat Example`, `pat@example.invalid`).
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.platforms("linux"),
    pytest.mark.skipif(shutil.which("git") is None, reason="git is required"),
]

HOOKS_SRC = Path(__file__).resolve().parents[1] / ".githooks"
SIM = ("sim", "sim@hermes-agent.nousresearch.com")
PERSONAL = ("Pat Example", "pat@example.invalid")
UPSTREAM = ("Upstream Dev", "dev@upstream.example")
POLICY = {"version": 1, "names": ["Pat Example"], "emails": ["pat@example.invalid"]}
DATE = "1700000000 +0000"
# Built at runtime so this file's own text never matches the public block pattern.
HOST_PATH = "/" + "mnt" + "/" + "work" + "/projects/x"
GIT_OWN_LINE = re.compile(r"(error|fatal|hint|warning|remote): |To ")

AUTHOR_NOT_SIM = "author is not the fork identity sim"
COMMITTER_NOT_SIM = "committer is not the fork identity sim"
POLICY_MISSING = "private identity policy is missing"


def isolated_env(root: Path) -> dict:
    """Environment for every git process here: no GIT_* leak-in, no ~/.gitconfig, no system."""
    home = root / "home"
    home.mkdir(exist_ok=True)
    gcfg = root / "gitconfig"
    gcfg.touch()
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("GIT_") and k != "HERMES_ALLOW_HOST_PATHS"}
    env.update(HOME=str(home), GIT_CONFIG_GLOBAL=str(gcfg), GIT_CONFIG_NOSYSTEM="1",
               GIT_TERMINAL_PROMPT="0", GIT_AUTHOR_DATE=DATE, GIT_COMMITTER_DATE=DATE)
    return env


def load_guard():
    """Import the helper from this checkout (it is not a package module)."""
    spec = importlib.util.spec_from_file_location("identity_guard", HOOKS_SRC / "identity_guard.py")
    assert spec is not None and spec.loader is not None
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    return guard


class Forge:
    """A disposable local clone + bare remote, with the hook installed in one layout."""

    def __init__(self, root: Path, layout: str = "copied", policy: dict | None = POLICY):
        self.root = root
        self.env = isolated_env(root)
        self.remote = root / "remote.git"
        self.repo = root / "local"
        self.git("init", "-q", "--bare", "-b", "main", str(self.remote), cwd=root)
        self.git("init", "-q", "-b", "main", str(self.repo), cwd=root)
        self.git("remote", "add", "origin", str(self.remote))
        self.common = self.repo / ".git"
        self.install(layout)
        if policy is not None:
            self.write_policy(policy)
        self.base = self.commit("base", [], files={"README.md": "hello\n"}, ref="refs/heads/main")
        self.git("push", "-q", "--no-verify", "origin", "main")

    # ── plumbing ────────────────────────────────────────────────────────────────────────
    def git(self, *args, cwd=None, input=None, env=None, check=True):
        run_env = dict(self.env, **(env or {}))
        res = subprocess.run(["git", *args], cwd=cwd or self.repo, input=input, env=run_env,
                             capture_output=True, timeout=300, check=False)
        if check and res.returncode != 0:
            raise AssertionError(f"setup git {args} failed: {res.stderr.decode(errors='replace')}")
        return res

    def out(self, *args, **kw) -> str:
        return self.git(*args, **kw).stdout.decode().strip()

    def install(self, layout: str):
        if layout == "copied":
            hooks = self.common / "hooks"
            for item in HOOKS_SRC.iterdir():
                dest = hooks / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)
            self.hook_dir = hooks
        elif layout == "tracked":
            shutil.copytree(HOOKS_SRC, self.repo / ".githooks")
            self.git("config", "core.hooksPath", ".githooks")
            self.hook_dir = self.repo / ".githooks"
        else:
            raise ValueError(layout)

    def write_policy(self, policy, *, raw: bytes | None = None, at: Path | None = None):
        pdir = (at or self.common) / "identity-policy"
        pdir.mkdir(exist_ok=True)
        pdir.chmod(0o700)
        pfile = pdir / "policy.json"
        pfile.write_bytes(raw if raw is not None else json.dumps(policy).encode())
        pfile.chmod(0o600)
        return pfile

    def commit(self, msg, parents, author=SIM, committer=SIM, files=None, ref=None) -> str:
        for name, text in (files or {}).items():
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        self.git("add", "-A")
        tree = self.out("write-tree")
        args = ["commit-tree", tree]
        for p in parents:
            args += ["-p", p]
        idents = {"GIT_AUTHOR_NAME": author[0], "GIT_AUTHOR_EMAIL": author[1],
                  "GIT_COMMITTER_NAME": committer[0], "GIT_COMMITTER_EMAIL": committer[1]}
        body = msg if isinstance(msg, bytes) else msg.encode()
        oid = self.out(*args, "-F", "-", input=body, env=idents)
        if ref:
            self.git("update-ref", ref, oid)
        return oid

    def raw_commit(self, raw: bytes, ref: str) -> str:
        oid = self.out("hash-object", "-t", "commit", "-w", "--literally", "--stdin", input=raw)
        self.git("update-ref", ref, oid)
        return oid

    def header(self, parent: str, author=SIM, committer=SIM) -> bytes:
        tree = self.out("rev-parse", f"{parent}^{{tree}}")
        return (f"tree {tree}\nparent {parent}\n"
                f"author {author[0]} <{author[1]}> {DATE}\n"
                f"committer {committer[0]} <{committer[1]}> {DATE}\n").encode()

    def remote_refs(self) -> dict:
        res = self.git("for-each-ref", "--format=%(objectname) %(refname)", cwd=self.remote,
                       env={"GIT_DIR": str(self.remote)})
        pairs = (line.split(" ", 1) for line in res.stdout.decode().splitlines())
        return {ref: oid for oid, ref in pairs}

    def push(self, *refspecs, cwd=None, env=None, config=(), force=False):
        cfg = [arg for kv in config for arg in ("-c", kv)]
        args = [*cfg, "push", "--porcelain", *(["--force"] if force else []), "origin", *refspecs]
        before = self.remote_refs()
        res = self.git(*args, cwd=cwd, env=env, check=False)
        return Pushed(res.returncode, res.stderr.decode(errors="replace"), before)

    def run_hook(self, stdin: bytes, env=None):
        hook = self.hook_dir / "pre-push"
        res = subprocess.run([str(hook), "origin", str(self.remote)], cwd=self.repo, input=stdin,
                             env=dict(self.env, **(env or {})), capture_output=True, timeout=300,
                             check=False)
        return res.returncode, res.stderr.decode(errors="replace")

    def said(self, stderr: str) -> str:
        """What the hook itself printed: no git lines, and the disposable root path masked."""
        text = stderr.replace(str(self.root), "<root>")
        return "\n".join(ln for ln in text.splitlines() if not GIT_OWN_LINE.match(ln))


class Pushed:
    def __init__(self, rc, stderr, before):
        self.rc, self.stderr, self.before = rc, stderr, before


PRIVATE_WORDS = [re.compile(rf"\b{re.escape(w)}\b") for w in PERSONAL[0].lower().split()]


def assert_no_private(text: str, said: str | None = None):
    """No full deny-list value anywhere in stderr; and, in what the hook itself printed (`said`:
    git's own lines dropped, root masked), not even one word of the synthetic name. Word-bounded,
    so the hook's fixed text ("pattern") cannot satisfy it; neither word occurs in that text."""
    low = text.lower()
    for value in (PERSONAL[0].lower(), PERSONAL[1].lower(), "pat\nexample"):
        assert value not in low, f"diagnostic disclosed a private deny-list value: {text}"
    said_low = (said or "").lower()
    for word in PRIVATE_WORDS if said is not None else ():
        assert not word.search(said_low), f"diagnostic disclosed a private name word: {said}"


def assert_refused(forge: Forge, res: Pushed, needle: str):
    assert res.rc != 0, (f"push was ACCEPTED (rc 0); remote refs moved from {res.before} to "
                         f"{forge.remote_refs()}; stderr:\n{res.stderr}")
    assert forge.remote_refs() == res.before, "a refused push still moved the remote"
    said = forge.said(res.stderr)
    assert needle in said, f"refused, but not for {needle!r}; the hook said:\n{said}"
    assert_no_private(res.stderr, said)


def assert_hook_refused(forge: Forge, rc: int, stderr: str, needle: str):
    """For crafted stdin that a real push cannot produce (the hook is run directly)."""
    assert rc != 0, f"the hook accepted crafted input:\n{stderr}"
    said = forge.said(stderr)
    assert needle in said, f"refused, but not for {needle!r}; the hook said:\n{said}"


def assert_accepted(forge: Forge, res: Pushed, ref: str, oid: str):
    assert res.rc == 0, f"push refused (rc {res.rc}):\n{res.stderr}"
    assert forge.remote_refs().get(ref) == oid


@pytest.fixture
def root(tmp_path_factory):
    """A disposable root with a neutral name (never the test's name)."""
    return tmp_path_factory.mktemp("r")


@pytest.fixture
def forge(root):
    return Forge(root)


def tool_dir(root: Path, python3: str | None) -> Path:
    """A PATH holding only the tools the hook needs, with `python3` absent or replaced."""
    tools = ["git", "bash", "sh", "env", "cat", "md5sum", "sha256sum", "cut", "mktemp", "rm",
             "sort", "wc", "tr", "timeout", "sed", "awk", "grep", "head", "dirname", "uname"]
    found = {t: shutil.which(t) for t in tools}
    missing = [t for t, p in found.items() if p is None]
    if missing:
        pytest.skip(f"host lacks {missing}")
    bindir = root / "bin"
    bindir.mkdir()
    for name, path in found.items():
        assert path is not None
        (bindir / name).symlink_to(path)
    if python3 is not None:
        fake = bindir / "python3"
        fake.write_text(python3)
        fake.chmod(0o755)
    return bindir


# ── the defect: a personal author hidden behind a sim merge ─────────────────────────────────
@pytest.mark.parametrize("layout", ["copied", "tracked"])
def test_hidden_merge_personal_author_refuses(root, layout):
    f = Forge(root, layout=layout)
    mainline = f.commit("mainline", [f.base], files={"a.txt": "a\n"})
    side = f.commit("side work", [f.base], author=PERSONAL, committer=PERSONAL,
                    files={"b.txt": "b\n"})
    merge = f.commit("merge side", [mainline, side], ref="refs/heads/main")
    assert merge
    assert_refused(f, f.push("main"), AUTHOR_NOT_SIM)


@pytest.mark.parametrize("layout", ["copied", "tracked"])
def test_sim_push_passes(root, layout):
    f = Forge(root, layout=layout)
    one = f.commit("one", [f.base], files={"a.txt": "a\n"})
    side = f.commit("side", [f.base], files={"b.txt": "b\n"})
    merge = f.commit("merge", [one, side], ref="refs/heads/main")
    assert_accepted(f, f.push("main"), "refs/heads/main", merge)


def test_personal_author_on_second_parent_of_octopus_refuses(forge):
    a = forge.commit("a", [forge.base], files={"a.txt": "a\n"})
    b = forge.commit("b", [forge.base], files={"b.txt": "b\n"})
    c = forge.commit("c", [forge.base], author=PERSONAL, files={"c.txt": "c\n"})
    forge.commit("octopus", [a, b, c], ref="refs/heads/main")
    assert_refused(forge, forge.push("main"), AUTHOR_NOT_SIM)


@pytest.mark.parametrize("committer", [PERSONAL, UPSTREAM, ("Sim", SIM[1]), ("sim", "sim@example.invalid")])
def test_non_sim_committer_refuses(forge, committer):
    forge.commit("work", [forge.base], committer=committer, files={"a.txt": "a\n"},
                 ref="refs/heads/main")
    assert_refused(forge, forge.push("main"), COMMITTER_NOT_SIM)


def test_upstream_author_fails_closed_at_this_checkpoint(forge):
    forge.commit("upstream work", [forge.base], author=UPSTREAM, files={"a.txt": "a\n"},
                 ref="refs/heads/main")
    assert_refused(forge, forge.push("main"), AUTHOR_NOT_SIM)


# ── messages ───────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("message", [
    "credit to Pat Example for the idea",
    "thanks PAT   example!",
    "mail PAT@Example.INVALID.",
    "pair\n\nCo-authored-by: Pat\nExample <x@y.example>",
])
def test_message_deny_list_hit_refuses_without_disclosure(forge, message):
    forge.commit(message, [forge.base], files={"a.txt": "a\n"}, ref="refs/heads/main")
    assert_refused(forge, forge.push("main"), "message matches the private deny-list")


def test_message_tokens_are_word_bounded_not_substrings(forge):
    oid = forge.commit("Pat Examples and Pat Exampleville; xpat@example.invalid is someone else",
                       [forge.base], files={"a.txt": "a\n"}, ref="refs/heads/main")
    assert_accepted(forge, forge.push("main"), "refs/heads/main", oid)


def test_message_public_block_pattern_refuses_even_with_host_path_override(forge):
    forge.commit(f"see {HOST_PATH}", [forge.base], files={"a.txt": "a\n"}, ref="refs/heads/main")
    assert_refused(forge, forge.push("main", env={"HERMES_ALLOW_HOST_PATHS": "1"}),
                   "message matches the public block pattern")


# ── extra headers: mergetag, gpgsig and any other header after committer ──────────────────
def _mergetag(base: str, tagger: tuple, note: str) -> bytes:
    return (f"mergetag object {base}\n type commit\n tag v1\n"
            f" tagger {tagger[0]} <{tagger[1]}> {DATE}\n \n {note}\n").encode()


def _gpgsig(comment: bytes) -> bytes:
    return (b"gpgsig -----BEGIN PGP SIGNATURE-----\n " + comment
            + b"\n \n iQEzBAABCAAdFiEE\n -----END PGP SIGNATURE-----\n")


@pytest.mark.parametrize("extra,needle", [
    (lambda b: _mergetag(b, PERSONAL, "release"), "'mergetag' header matches the private deny-list"),
    (lambda b: _mergetag(b, SIM, "tagged for Pat Example"),
     "'mergetag' header matches the private deny-list"),
    (lambda b: _gpgsig(b"Comment: pat@example.invalid"),
     "'gpgsig' header matches the private deny-list"),
    (lambda b: _gpgsig(b"Comment: \xff\xfe"), "'gpgsig' header is not valid UTF-8"),
], ids=["mergetag-tagger", "mergetag-message", "gpgsig-email", "gpgsig-not-utf8"])
def test_extra_header_text_is_audited(forge, extra, needle):
    forge.raw_commit(forge.header(forge.base) + extra(forge.base) + b"\nmerge\n",
                     "refs/heads/main")
    assert_refused(forge, forge.push("main"), needle)


def test_clean_mergetag_and_gpgsig_pass(forge):
    raw = (forge.header(forge.base) + _mergetag(forge.base, SIM, "release")
           + _gpgsig(b"Comment: signed") + b"\nmerge\n")
    oid = forge.raw_commit(raw, "refs/heads/main")
    assert_accepted(forge, forge.push("main"), "refs/heads/main", oid)


@pytest.mark.parametrize("line", [b"pat@example.invalid x", b"Pat Example x"])
def test_unparseable_header_never_echoes_its_token(forge, line):
    forge.raw_commit(forge.header(forge.base) + line + b"\n\nmsg\n", "refs/heads/main")
    assert_refused(forge, forge.push("main"), "malformed header: an unparseable header line")


@pytest.mark.parametrize("key,extra", [
    ("encoding", b"encoding UTF-8\nencoding UTF-8\n"),
    ("gpgsig", _gpgsig(b"Comment: one") + _gpgsig(b"Comment: two")),
])
def test_duplicated_non_core_header_refuses(forge, key, extra):
    forge.raw_commit(forge.header(forge.base) + extra + b"\nmsg\n", "refs/heads/main")
    assert_refused(forge, forge.push("main"), f"malformed header: duplicate or misplaced '{key}' header")


@pytest.mark.parametrize("extra,needle", [
    (b"pat example\n", "an unknown header matches the private deny-list"),
    (b"pat one\npat two\n", "malformed header: duplicate or misplaced an unknown header"),
], ids=["single", "duplicated"])
def test_unknown_header_key_is_never_printed(forge, extra, needle):
    """A key-shaped token that is not a git header name may be a word of a private value."""
    forge.raw_commit(forge.header(forge.base) + extra + b"\nmsg\n", "refs/heads/main")
    assert_refused(forge, forge.push("main"), needle)


def test_continuation_line_without_a_header_refuses(forge):
    forge.raw_commit(forge.header(forge.base) + b" x\n\nmsg\n", "refs/heads/main")
    assert_refused(forge, forge.push("main"),
                   "malformed header: a continuation line without a header")


# ── refs: multi-ref, new ref, update range, tags, deletion ─────────────────────────────────
def test_multi_ref_push_refuses_all_refs_when_one_is_bad(forge):
    good = forge.commit("good", [forge.base], files={"a.txt": "a\n"}, ref="refs/heads/main")
    forge.commit("bad", [forge.base], author=PERSONAL, files={"b.txt": "b\n"},
                 ref="refs/heads/side")
    res = forge.push("main", "side")
    assert_refused(forge, res, AUTHOR_NOT_SIM)
    assert forge.remote_refs().get("refs/heads/main") != good


def test_new_ref_audits_full_ancestry_without_tracking_ref_exclusion(forge):
    bad = forge.commit("published personal", [forge.base], author=PERSONAL,
                       files={"a.txt": "a\n"}, ref="refs/heads/main")
    forge.git("push", "-q", "--no-verify", "origin", "main")  # already on the remote
    forge.commit("feature", [bad], files={"b.txt": "b\n"}, ref="refs/heads/feature")
    assert_refused(forge, forge.push("feature"), AUTHOR_NOT_SIM)


def test_update_audits_only_commits_beyond_the_advertised_old_tip(forge):
    bad = forge.commit("published personal", [forge.base], author=PERSONAL,
                       files={"a.txt": "a\n"}, ref="refs/heads/main")
    forge.git("push", "-q", "--no-verify", "origin", "main")
    tip = forge.commit("next", [bad], files={"b.txt": "b\n"}, ref="refs/heads/main")
    assert_accepted(forge, forge.push("main"), "refs/heads/main", tip)


def test_lightweight_tag_on_sim_commit_passes(forge):
    oid = forge.commit("tagged", [forge.base], files={"a.txt": "a\n"}, ref="refs/tags/v1")
    assert_accepted(forge, forge.push("refs/tags/v1"), "refs/tags/v1", oid)


def test_annotated_tag_refuses_at_this_checkpoint(forge):
    forge.commit("tagged", [forge.base], files={"a.txt": "a\n"}, ref="refs/heads/main")
    forge.git("tag", "-a", "v1", "-m", "release", "main",
              env={"GIT_COMMITTER_NAME": SIM[0], "GIT_COMMITTER_EMAIL": SIM[1]})
    assert_refused(forge, forge.push("refs/tags/v1"), "is an annotated tag; annotated tags are refused")


def test_non_commit_object_refuses(forge):
    blob = forge.out("hash-object", "-w", "--stdin", input=b"payload\n")
    assert_refused(forge, forge.push(f"{blob}:refs/tags/blob"), f"{blob} is a blob, not a commit")


def test_deletion_passes_with_valid_policy(forge):
    forge.git("push", "-q", "--no-verify", "origin", f"{forge.base}:refs/heads/old")
    res = forge.push(":refs/heads/old")
    assert res.rc == 0, res.stderr
    assert "refs/heads/old" not in forge.remote_refs()


def test_deletion_only_push_still_validates_policy(forge):
    forge.git("push", "-q", "--no-verify", "origin", f"{forge.base}:refs/heads/old")
    shutil.rmtree(forge.common / "identity-policy")
    assert_refused(forge, forge.push(":refs/heads/old"), POLICY_MISSING)


def test_empty_input_still_validates_policy(forge):
    shutil.rmtree(forge.common / "identity-policy")
    rc, err = forge.run_hook(b"")
    assert_hook_refused(forge, rc, err, POLICY_MISSING)


# ── caps, limits and missing data ──────────────────────────────────────────────────────────
def test_commit_cap_refuses(forge):
    tip = forge.base
    for i in range(3):
        tip = forge.commit(f"c{i}", [tip], files={f"{i}.txt": "x\n"})
    forge.git("update-ref", "refs/heads/main", tip)
    assert_refused(forge, forge.push("main", config=["identityguard.maxCommits=2"]),
                   "more than 2 commits to audit exceeds the commit cap")


def test_time_limit_refuses(forge):
    forge.commit("one", [forge.base], files={"a.txt": "a\n"}, ref="refs/heads/main")
    res = forge.push("main", config=["identityguard.timeoutSeconds=0.000001"])
    assert_refused(forge, res, "time limit was exceeded")


@pytest.mark.parametrize("knob,needle", [
    ("identityguard.maxCommits=abc", "identityguard.maxCommits must be a positive integer"),
    ("identityguard.maxCommits=0", "identityguard.maxCommits must be a positive integer"),
    ("identityguard.timeoutSeconds=-1", "identityguard.timeoutSeconds must be a positive number"),
    ("identityguard.timeoutSeconds=nan", "identityguard.timeoutSeconds must be a positive number"),
])
def test_malformed_limit_knob_refuses(forge, knob, needle):
    forge.commit("one", [forge.base], files={"a.txt": "a\n"}, ref="refs/heads/main")
    assert_refused(forge, forge.push("main", config=[knob]), needle)


@pytest.mark.parametrize("knobs,expected", [
    ({"maxCommits": "999999999", "timeoutSeconds": "99999"}, (250_000, 180.0)),
    ({"maxCommits": "7", "timeoutSeconds": "2.5"}, (7, 2.5)),
    ({}, (250_000, 180.0)),
], ids=["above-default", "below-default", "unset"])
def test_limit_knobs_can_only_lower_the_defaults(root, monkeypatch, knobs, expected):
    env = isolated_env(root)
    for key in [k for k in os.environ if k.startswith("GIT_")]:
        monkeypatch.delenv(key)
    for key in ("HOME", "GIT_CONFIG_GLOBAL", "GIT_CONFIG_NOSYSTEM"):
        monkeypatch.setenv(key, env[key])
    repo = root / "k"
    subprocess.run(["git", "init", "-q", str(repo)], env=env, check=True, timeout=60)
    for name, value in knobs.items():
        subprocess.run(["git", "-C", str(repo), "config", f"identityguard.{name}", value],
                       env=env, check=True, timeout=60)
    monkeypatch.chdir(repo)
    guard = load_guard()
    assert guard.read_limits(guard.Clock(guard.TIME_LIMIT, time.monotonic())) == expected


def test_missing_parent_object_refuses(forge):
    a = forge.commit("a", [forge.base], files={"a.txt": "a\n"})
    forge.commit("b", [a], files={"b.txt": "b\n"}, ref="refs/heads/feature")
    loose = forge.common / "objects" / a[:2] / a[2:]
    loose.chmod(0o600)
    loose.unlink()
    assert_refused(forge, forge.push("feature"), "missing or unreadable")


def test_unknown_advertised_old_tip_refuses(forge, root):
    other = root / "other"
    forge.git("clone", "-q", str(forge.remote), str(other), cwd=root)
    (other / "z.txt").write_text("z\n")
    forge.git("add", "-A", cwd=other)
    forge.git("-c", f"user.name={SIM[0]}", "-c", f"user.email={SIM[1]}", "commit", "-q",
              "-m", "elsewhere", cwd=other)
    forge.git("push", "-q", "--no-verify", "origin", "main", cwd=other)
    forge.commit("mine", [forge.base], files={"a.txt": "a\n"}, ref="refs/heads/main")
    assert_refused(forge, forge.push("main", force=True),
                   "is not a commit in this repository; fetch first (git fetch origin)")


def test_shallow_clone_refuses(forge, root):
    forge.commit("two", [forge.base], files={"a.txt": "a\n"}, ref="refs/heads/main")
    forge.git("push", "-q", "--no-verify", "origin", "main")
    shallow = root / "shallow"
    forge.git("clone", "-q", "--depth=1", f"file://{forge.remote}", str(shallow), cwd=root)
    shutil.copytree(forge.common / "hooks", shallow / ".git" / "hooks", dirs_exist_ok=True)
    forge.write_policy(POLICY, at=shallow / ".git")
    (shallow / "c.txt").write_text("c\n")
    forge.git("add", "-A", cwd=shallow)
    forge.git("-c", f"user.name={SIM[0]}", "-c", f"user.email={SIM[1]}", "commit", "-q",
              "-m", "three", cwd=shallow)
    res = forge.push("main", cwd=shallow)
    assert_refused(forge, res, "this is a shallow repository")


# ── raw metadata and malformed objects ─────────────────────────────────────────────────────
def test_non_utf8_message_refuses(forge):
    # commit-tree transcodes invalid bytes as Latin-1, so write the raw object.
    forge.raw_commit(forge.header(forge.base) + b"\nbad bytes \xff\xfe here\n", "refs/heads/main")
    assert_refused(forge, forge.push("main"), "message is not valid UTF-8")


@pytest.mark.parametrize("mutate,needle", [
    (lambda h: h + b"\nmessage with \x00 nul\n", "message contains a NUL byte"),
    (lambda h: h.replace(b"author sim <sim@hermes-agent.nousresearch.com>",
                         b"author sim sim@hermes-agent.nousresearch.com") + b"\nmsg\n",
     "malformed or missing author header"),
    (lambda h: h + h.splitlines(keepends=True)[2] + b"\nmsg\n",
     "malformed header: duplicate or misplaced 'author' header"),
    (lambda h: b"".join(l for l in h.splitlines(keepends=True) if not l.startswith(b"committer"))
     + b"\nmsg\n", "malformed or missing committer header"),
    (lambda h: h + b"encoding ISO-8859-1\n\nmsg\n", "declares a non-UTF-8 encoding header"),
], ids=["nul", "author-shape", "author-twice", "no-committer", "latin1-encoding"])
def test_malformed_commit_object_refuses(forge, mutate, needle):
    forge.raw_commit(mutate(forge.header(forge.base)), "refs/heads/main")
    assert_refused(forge, forge.push("main"), needle)


@pytest.mark.parametrize("stdin", [
    b"refs/heads/main abc refs/heads/main " + b"0" * 40 + b"\n",
    b"refs/heads/main " + b"g" * 40 + b" refs/heads/main " + b"0" * 40 + b"\n",
    b"refs/heads/main " + b"1" * 40 + b" refs/heads/main\n",
    b"refs/heads/main " + b"1" * 40 + b" main " + b"0" * 40 + b"\n",
    b"refs/heads/\xff " + b"1" * 40 + b" refs/heads/main " + b"0" * 40 + b"\n",
    b"refs/heads/main  " + b"1" * 40 + b" refs/heads/main " + b"0" * 40 + b"\n",
])
def test_malformed_hook_input_refuses(forge, stdin):
    rc, err = forge.run_hook(stdin)
    assert_hook_refused(forge, rc, err, "malformed pre-push input (line 1")


# ── private policy bundle ──────────────────────────────────────────────────────────────────
def _symlinked_file(f: Forge):
    real = f.root / "elsewhere.json"
    real.write_text(json.dumps(POLICY))
    real.chmod(0o600)
    pfile = f.common / "identity-policy" / "policy.json"
    pfile.unlink()
    pfile.symlink_to(real)


def _symlinked_dir(f: Forge):
    real = f.root / "elsewhere"
    shutil.move(str(f.common / "identity-policy"), real)
    (f.common / "identity-policy").symlink_to(real)


POLICY_BREAKAGE = {
    "missing": (lambda f: shutil.rmtree(f.common / "identity-policy"), POLICY_MISSING),
    "symlinked file": (_symlinked_file, "policy.json is a symlink"),
    "symlinked dir": (_symlinked_dir, "identity-policy is a symlink"),
    "group-writable file": (lambda f: (f.common / "identity-policy" / "policy.json").chmod(0o620),
                            "policy.json is group- or world-writable"),
    "world-writable dir": (lambda f: (f.common / "identity-policy").chmod(0o707),
                           "identity-policy is group- or world-writable"),
    "duplicate keys": (lambda f: f.write_policy(None, raw=(
        b'{"version": 1, "names": ["Pat Example"], "emails": ["pat@example.invalid"], '
        b'"emails": ["other@example.invalid"]}')), "has a duplicate JSON key"),
    "unknown key": (lambda f: f.write_policy({**POLICY, "snapshots": []}),
                    "has unknown key(s) ['snapshots']"),
    "other version": (lambda f: f.write_policy({**POLICY, "version": 2}), "version is not 1"),
    "bool version": (lambda f: f.write_policy({**POLICY, "version": True}), "version is not 1"),
    "no names": (lambda f: f.write_policy({"version": 1, "emails": POLICY["emails"]}),
                 "lacks key(s) ['names']"),
    "empty emails": (lambda f: f.write_policy({**POLICY, "emails": []}),
                     "'emails' must be a list of 1..32 entries"),
    "single-word name": (lambda f: f.write_policy({**POLICY, "names": ["Pat"]}),
                         "names[0] is not a full name"),
    "bad email": (lambda f: f.write_policy({**POLICY, "emails": ["not-an-email"]}),
                  "emails[0] is not an email"),
    "invalid json": (lambda f: f.write_policy(None, raw=b'{"version": 1, '),
                     "is not valid UTF-8 JSON"),
    "not utf-8": (lambda f: f.write_policy(None, raw=b'{"version": 1, "names": ["\xff"]}'),
                  "is not valid UTF-8 JSON"),
}


@pytest.mark.parametrize("breakage", sorted(POLICY_BREAKAGE))
def test_broken_policy_refuses_a_clean_sim_push(forge, breakage):
    forge.commit("clean", [forge.base], files={"a.txt": "a\n"}, ref="refs/heads/main")
    breaker, needle = POLICY_BREAKAGE[breakage]
    breaker(forge)
    assert_refused(forge, forge.push("main"), needle)


def test_policy_owner_is_checked_on_the_opened_dir_and_file(root):
    """Unprivileged tests cannot chown (and this host has no subuid user namespaces), so the
    ownership rule is driven through the real loader with an injected euid on a real tree."""
    guard = load_guard()
    common = root / "common"
    pdir = common / "identity-policy"
    pdir.mkdir(parents=True)
    pdir.chmod(0o700)
    pfile = pdir / "policy.json"
    pfile.write_text(json.dumps(POLICY))
    pfile.chmod(0o600)
    me, other = os.geteuid(), os.geteuid() + 1
    assert guard.load_policy(str(common), me) == (POLICY["names"], POLICY["emails"])
    # the opened directory (fstat after O_NOFOLLOW open)
    with pytest.raises(guard.GuardError, match=r"identity-policy is owned by uid"):
        guard.load_policy(str(common), other)
    # the opened file, inside a directory that passed for its owner
    dfd = guard.open_policy_dir(str(common), me)
    try:
        with pytest.raises(guard.GuardError, match=r"policy\.json is owned by uid"):
            guard.read_policy_file(dfd, str(pfile), other)
    finally:
        os.close(dfd)
    # the pre-open lstat: a symlinked dir is never opened, whatever the euid
    moved = root / "moved"
    pdir.rename(moved)
    pdir.symlink_to(moved)
    with pytest.raises(guard.GuardError, match=r"identity-policy is a symlink"):
        guard.open_policy_dir(str(common), me)


def test_run_checks_policy_ownership_against_the_process_euid(root, monkeypatch):
    """run() itself must pass os.geteuid() to the loader: with a different euid, a real
    0700/0600 policy tree owned by this user is refused."""
    env = isolated_env(root)
    for key in [k for k in os.environ if k.startswith("GIT_")]:
        monkeypatch.delenv(key)
    for key in ("HOME", "GIT_CONFIG_GLOBAL", "GIT_CONFIG_NOSYSTEM"):
        monkeypatch.setenv(key, env[key])
    repo = root / "e"
    subprocess.run(["git", "init", "-q", str(repo)], env=env, check=True, timeout=60)
    pdir = repo / ".git" / "identity-policy"
    pdir.mkdir()
    pdir.chmod(0o700)
    (pdir / "policy.json").write_text(json.dumps(POLICY))
    (pdir / "policy.json").chmod(0o600)
    monkeypatch.chdir(repo)
    guard = load_guard()
    other = os.geteuid() + 1
    monkeypatch.setattr(guard.os, "geteuid", lambda: other)
    with pytest.raises(guard.GuardError, match=r"identity-policy is owned by uid"):
        guard.run(["--block-patterns", str(HOOKS_SRC / "patterns" / "block.ere")], b"")


def test_policy_beside_tracked_hook_is_not_trusted(root):
    f = Forge(root, layout="tracked", policy=None)
    f.write_policy(POLICY, at=f.repo / ".githooks")
    f.commit("clean", [f.base], files={"a.txt": "a\n"}, ref="refs/heads/main")
    assert_refused(f, f.push("main"), POLICY_MISSING)


def test_linked_worktree_uses_common_dir_hooks_and_policy(forge, root):
    wt = root / "wt"
    forge.git("worktree", "add", "-q", "-b", "topic", str(wt), forge.base)
    (wt / "w.txt").write_text("w\n")
    forge.git("add", "-A", cwd=wt)
    forge.git("-c", f"user.name={SIM[0]}", "-c", f"user.email={SIM[1]}", "commit", "-q",
              "-m", "from worktree", cwd=wt)
    tip = forge.out("rev-parse", "HEAD", cwd=wt)
    assert_accepted(forge, forge.push("topic", cwd=wt), "refs/heads/topic", tip)


def test_policy_only_in_linked_worktree_refuses(root):
    f = Forge(root, policy=None)
    wt = root / "wt"
    f.git("worktree", "add", "-q", "-b", "topic", str(wt), f.base)
    f.write_policy(POLICY, at=Path(f.out("rev-parse", "--absolute-git-dir", cwd=wt)))
    f.write_policy(POLICY, at=wt)
    (wt / "w.txt").write_text("w\n")
    f.git("add", "w.txt", cwd=wt)
    f.git("-c", f"user.name={SIM[0]}", "-c", f"user.email={SIM[1]}", "commit", "-q",
          "-m", "from worktree", cwd=wt)
    assert_refused(f, f.push("topic", cwd=wt), POLICY_MISSING)


# ── helper discovery and integrity ─────────────────────────────────────────────────────────
def test_missing_helper_refuses(forge):
    (forge.hook_dir / "identity_guard.py").unlink(missing_ok=True)
    forge.commit("clean", [forge.base], files={"a.txt": "a\n"}, ref="refs/heads/main")
    assert_refused(forge, forge.push("main"), "identity helper not found beside this hook")


def test_modified_helper_refuses(forge):
    helper = forge.hook_dir / "identity_guard.py"
    with helper.open("a") as fh:
        fh.write("\n# locally modified\n")
    forge.commit("clean", [forge.base], files={"a.txt": "a\n"}, ref="refs/heads/main")
    assert_refused(forge, forge.push("main"), "does not match its published sha256")


def test_missing_python_refuses_with_install_diagnostic(forge, root):
    bindir = tool_dir(root, python3=None)
    forge.commit("clean", [forge.base], files={"a.txt": "a\n"}, ref="refs/heads/main")
    res = forge.push("main", env={"PATH": str(bindir)})
    assert_refused(forge, res, "Python 3.10 or newer is required")


def test_python_older_than_310_refuses_with_install_diagnostic(forge, root):
    # Stands in for an old interpreter: it fails the hook's version probe like Python < 3.10.
    bindir = tool_dir(root, python3="#!/bin/sh\nexit 1\n")
    forge.commit("clean", [forge.base], files={"a.txt": "a\n"}, ref="refs/heads/main")
    res = forge.push("main", env={"PATH": str(bindir)})
    assert_refused(forge, res, "Python 3.10 or newer is required")


# ── host-path content gate stays separate ──────────────────────────────────────────────────
def test_host_path_content_still_refuses(forge):
    forge.commit("add notes", [forge.base], files={"notes.md": f"lives in {HOST_PATH}\n"},
                 ref="refs/heads/main")
    assert_refused(forge, forge.push("main"), "the push adds host-path / personal-info literals")


def test_host_path_override_bypasses_only_the_content_gate(forge):
    oid = forge.commit("add notes", [forge.base], files={"notes.md": f"lives in {HOST_PATH}\n"},
                       ref="refs/heads/main")
    res = forge.push("main", env={"HERMES_ALLOW_HOST_PATHS": "1"})
    assert_accepted(forge, res, "refs/heads/main", oid)


def test_host_path_override_never_bypasses_identity(forge):
    forge.commit("personal", [forge.base], author=PERSONAL, committer=PERSONAL,
                 files={"a.txt": "a\n"}, ref="refs/heads/main")
    res = forge.push("main", env={"HERMES_ALLOW_HOST_PATHS": "1"})
    assert_refused(forge, res, AUTHOR_NOT_SIM)
