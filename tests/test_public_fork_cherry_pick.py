"""Behaviour contract for e3 of the fork pre-push identity gate: TARS-reviewed cherry-pick
records (`.githooks/pre-push` + `.githooks/identity_guard.py`, policy schema v3), and the
receiver's own old-OID check under a real push race.

A fork cherry-pick of an upstream commit keeps the upstream author, so the identity gate would
refuse it. It passes only when an exact pick record in the private bundle names its full
candidate OID and the full OID of the upstream commit it was picked from, the candidate is
committed by sim, and its raw author name and email equal the recorded upstream commit's. A
`(cherry picked from commit ...)` trailer, an equal author or an equal patch is never proof:
any other object, however alike, refuses.

Every push is a real `git push` into a disposable bare remote (the `Forge` of the e1 contract)
through the hook copied from this checkout; the canonical upstream is the e2 contract's local
bare stand-in, and the picks are made by a real `git cherry-pick -x`. Deny-list values are
synthetic (`Pat Example`, `pat@example.invalid`).
"""

from __future__ import annotations

import json
from itertools import count

import pytest

from tests.test_public_fork_identity_hook import (
    DATE,
    PERSONAL,
    SIM,
    UPSTREAM,
    Forge,
    assert_accepted,
    assert_refused,
    root as root,  # noqa: PLC0414 -- the self-alias is load-bearing: it re-exports the fixture
)
from tests.test_public_fork_upstream_snapshot import (
    CANONICAL,
    NOT_VERIFIED,
    REWRITE,
    SOURCE,
    VERIFIED_AT,
    Upstream,
    _local_commits,
    bundle,
    snapshot,
)

pytestmark = [pytest.mark.platforms("linux")]

PICK_AUTHOR = "author does not equal the author of the upstream commit its pick record names"
PICK_COMMITTER = "committer is not the fork identity sim; a recorded cherry-pick is committed by sim"
TRAILER_HINT = "a trailer is not proof"
OTHER = ("Other Dev", "other@upstream.example")
_SEQ = count()


def pick_record(candidate: str, upstream_commit: str, **over) -> dict:
    return {"candidate": candidate, "upstream_commit": upstream_commit, "repository": CANONICAL,
            "source": SOURCE, "ref": "refs/heads/main", "verified_at": VERIFIED_AT,
            "note": "candidate and upstream diffs and messages compared by TARS: clean pick",
            **over}


def commit_on(forge: Forge, parent: str, files: dict, msg: str, author=SIM, committer=SIM) -> str:
    """A commit whose tree is `parent`'s plus `files`, built in a private index."""
    index = forge.root / f"index-{next(_SEQ)}"
    env = {"GIT_INDEX_FILE": str(index)}
    forge.git("read-tree", parent, env=env)
    for name, text in files.items():
        blob = forge.out("hash-object", "-w", "--stdin", input=text.encode())
        forge.git("update-index", "--add", "--cacheinfo", f"100644,{blob},{name}", env=env)
    tree = forge.out("write-tree", env=env)
    idents = {"GIT_AUTHOR_NAME": author[0], "GIT_AUTHOR_EMAIL": author[1],
              "GIT_COMMITTER_NAME": committer[0], "GIT_COMMITTER_EMAIL": committer[1]}
    return forge.out("commit-tree", tree, "-p", parent, "-F", "-", input=msg.encode(), env=idents)


def publish_upstream(forge: Forge, up: Upstream, oid: str) -> str:
    """Make `oid` the stand-in upstream's main and fetch it back (`refs/remotes/upstream/*`)."""
    forge.git("push", "-q", "--no-verify", "--force", str(up.bare), f"{oid}:refs/heads/main")
    up.tip = oid
    up.fetch()
    return oid


def cherry_pick(forge: Forge, onto: str, source: str, committer=SIM) -> str:
    """A real `git cherry-pick -x` of `source` onto `onto` in a throwaway linked worktree."""
    wt = forge.root / f"pick-{next(_SEQ)}"
    forge.git("worktree", "add", "-q", "--detach", str(wt), onto)
    forge.git("cherry-pick", "-x", source, cwd=wt,
              env={"GIT_COMMITTER_NAME": committer[0], "GIT_COMMITTER_EMAIL": committer[1]})
    oid = forge.out("rev-parse", "HEAD", cwd=wt)
    forge.git("worktree", "remove", "--force", str(wt))
    return oid


def variant(forge: Forge, oid: str, old: bytes, new: bytes) -> str:
    """Another commit object: the raw bytes of `oid` with one edit."""
    raw = forge.git("cat-file", "commit", oid).stdout
    assert old in raw, (old, raw)
    return forge.out("hash-object", "-t", "commit", "-w", "--stdin", input=raw.replace(old, new, 1))


class Scene:
    """The fork has one sim commit on the base; upstream has one fix by its own author; the
    candidate is the fix cherry-picked onto the fork commit by sim."""

    def __init__(self, forge: Forge, up: Upstream, upstream_author=UPSTREAM):
        self.forge, self.up = forge, up
        self.source = publish_upstream(forge, up, commit_on(
            forge, forge.base, {"up.txt": "upstream fix\n"}, "upstream fix\n",
            author=upstream_author, committer=upstream_author))
        self.fork = commit_on(forge, forge.base, {"a.txt": "fork work\n"}, "fork work\n")
        self.candidate = cherry_pick(forge, self.fork, self.source)

    def policy(self, picks=None, pins=None):
        picks = [pick_record(self.candidate, self.source)] if picks is None else picks
        self.forge.write_policy(bundle([snapshot(*(pins or [self.source]))], pick_records=picks))

    def push(self, oid: str):
        self.forge.git("update-ref", "refs/heads/main", oid)
        return self.forge.push("main")


@pytest.fixture
def forge(root):
    return Forge(root, policy=bundle())


@pytest.fixture
def up(forge):
    return Upstream(forge)


@pytest.fixture
def scene(forge, up):
    return Scene(forge, up)


# ── a reviewed pick passes ─────────────────────────────────────────────────────────────────
def test_reviewed_upstream_cherry_pick_passes(scene):
    raw = scene.forge.git("cat-file", "commit", scene.candidate).stdout
    assert f"author {UPSTREAM[0]} <{UPSTREAM[1]}>".encode() in raw
    assert f"committer {SIM[0]} <{SIM[1]}>".encode() in raw
    scene.policy()
    assert_accepted(scene.forge, scene.push(scene.candidate), "refs/heads/main", scene.candidate)


def test_reviewed_pick_of_an_open_upstream_pr_passes(scene):
    """PR evidence: the source is a canonical `refs/pull/<n>/head`, pinned by no snapshot."""
    scene.forge.write_policy(bundle(pick_records=[
        pick_record(scene.candidate, scene.source, ref="refs/pull/4242/head")]))
    assert_accepted(scene.forge, scene.push(scene.candidate), "refs/heads/main", scene.candidate)


def test_reviewed_pick_behind_a_sim_merge_passes(scene):
    merge = commit_on(scene.forge, scene.candidate, {"m.txt": "m\n"}, "more fork work\n")
    scene.policy()
    assert_accepted(scene.forge, scene.push(merge), "refs/heads/main", merge)


# ── no record, or a record for another object: refuses ────────────────────────────────────
def test_unrecorded_cherry_pick_refuses_and_names_candidate_and_claimed_source(scene):
    scene.policy(picks=[])
    res = scene.push(scene.candidate)
    assert_refused(scene.forge, res, NOT_VERIFIED)
    said = scene.forge.said(res.stderr)
    assert scene.candidate in said and scene.source in said and TRAILER_HINT in said, said
    assert "TARS" in said and "never writes approval" in said, said
    assert REWRITE not in said, f"a record-based refusal printed the rewrite remedy:\n{said}"


def test_forged_cherry_pick_trailer_refuses(scene):
    """Upstream author, sim committer and a trailer naming the recorded source: not proof."""
    forged = commit_on(scene.forge, scene.fork, {"other.txt": "not the fix\n"},
                       f"upstream fix\n\n(cherry picked from commit {scene.source})\n",
                       author=UPSTREAM)
    scene.policy()
    res = scene.push(forged)
    assert_refused(scene.forge, res, NOT_VERIFIED)
    said = scene.forge.said(res.stderr)
    assert forged in said and TRAILER_HINT in said, said


@pytest.mark.parametrize("edit", ["message", "tree", "committer-date"])
def test_changed_candidate_refuses(scene, edit):
    """An amended pick is another object; the record of the reviewed one does not carry over.
    `committer-date` is the unreviewed lookalike: every byte but one timestamp is the same."""
    picked_tree = scene.forge.out("rev-parse", f"{scene.candidate}^{{tree}}")
    fork_tree = scene.forge.out("rev-parse", f"{scene.fork}^{{tree}}")
    old, new = {
        "message": (b"upstream fix\n", b"upstream fix, adjusted\n"),
        "tree": (f"tree {picked_tree}\n".encode(), f"tree {fork_tree}\n".encode()),
        "committer-date": (f"{SIM[1]}> {DATE}".encode(), f"{SIM[1]}> 1700000001 +0000".encode()),
    }[edit]
    changed = variant(scene.forge, scene.candidate, old, new)
    assert changed != scene.candidate
    scene.policy()
    assert_refused(scene.forge, scene.push(changed), NOT_VERIFIED)


def test_record_naming_another_upstream_commit_refuses(scene):
    """The record's source changed to an upstream commit by someone else."""
    other = commit_on(scene.forge, scene.forge.base, {"up.txt": "upstream fix\n"}, "upstream fix\n",
                      author=OTHER, committer=OTHER)
    scene.forge.write_policy(bundle([snapshot(scene.source, other)],
                                    pick_records=[pick_record(scene.candidate, other)]))
    res = scene.push(scene.candidate)
    assert_refused(scene.forge, res, PICK_AUTHOR)
    said = scene.forge.said(res.stderr)
    assert scene.candidate in said and other in said and "TARS" in said, said


def test_pick_of_a_changed_upstream_source_refuses(scene):
    """Upstream replaced the reviewed source; a re-pick of the replacement is not reviewed."""
    changed = publish_upstream(scene.forge, scene.up, commit_on(
        scene.forge, scene.forge.base, {"up.txt": "upstream fix, reworked\n"}, "upstream fix\n",
        author=UPSTREAM, committer=UPSTREAM))
    repick = cherry_pick(scene.forge, scene.fork, changed)
    scene.policy(pins=[scene.source, changed])
    assert_refused(scene.forge, scene.push(repick), NOT_VERIFIED)


def test_removing_the_pick_record_revokes_approval(scene):
    scene.policy()
    scene.forge.git("push", "-q", "--no-verify", "origin", f"{scene.candidate}:refs/heads/probe")
    scene.policy(picks=[])
    assert_refused(scene.forge, scene.push(scene.candidate), NOT_VERIFIED)


def test_recorded_pick_committed_by_someone_else_refuses(scene):
    picked = cherry_pick(scene.forge, scene.fork, scene.source, committer=UPSTREAM)
    scene.policy(picks=[pick_record(picked, scene.source)])
    res = scene.push(picked)
    assert_refused(scene.forge, res, PICK_COMMITTER)
    assert REWRITE in scene.forge.said(res.stderr)


def test_recorded_pick_author_on_the_deny_list_refuses(forge, up):
    scene = Scene(forge, up, upstream_author=("Someone Else", PERSONAL[1]))
    scene.policy()
    assert_refused(forge, scene.push(scene.candidate), "author email matches the private deny-list")


def test_recorded_pick_message_on_the_deny_list_refuses(scene):
    reworded = variant(scene.forge, scene.candidate, b"upstream fix\n", b"upstream fix by Pat Example\n")
    scene.policy(picks=[pick_record(reworded, scene.source)])
    assert_refused(scene.forge, scene.push(reworded), "message matches the private deny-list")


# ── the v3 bundle: strict schema, exact records ────────────────────────────────────────────
def _picks(*recs):
    return lambda s: bundle([snapshot(s.source)], pick_records=[r(s) for r in recs])


BROKEN_PICKS = {
    "v2 bundle": (lambda s: {k: v for k, v in bundle(version=2).items() if k != "pick_records"},
                  "version is not 3"),
    "empty policy": (lambda s: {}, "version is not 3"),
    "no pick_records key": (lambda s: {k: v for k, v in bundle().items() if k != "pick_records"},
                            "lacks key(s) ['pick_records']"),
    "pick_records not a list": (lambda s: bundle(pick_records={}), "'pick_records' must be a list"),
    "arbitrary author allowlist": (lambda s: bundle(allowed_authors=[UPSTREAM[1]]),
                                   "has unknown key(s) ['allowed_authors']"),
    "record author field": (_picks(lambda s: pick_record(s.candidate, s.source, author="*")),
                            "pick_records[0] must have exactly the fields"),
    "record without note": (_picks(lambda s: {k: v for k, v in pick_record(s.candidate, s.source).items()
                                              if k != "note"}), "pick_records[0] must have exactly the fields"),
    "empty note": (_picks(lambda s: pick_record(s.candidate, s.source, note=" ")),
                   "pick_records[0] note must be a non-empty"),
    "wildcard candidate": (_picks(lambda s: pick_record("*", s.source)),
                           "pick_records[0] candidate is not a full object id"),
    "short candidate": (_picks(lambda s: pick_record(s.candidate[:12], s.source)),
                        "pick_records[0] candidate is not a full object id"),
    "short upstream commit": (_picks(lambda s: pick_record(s.candidate, s.source[:12])),
                              "pick_records[0] upstream_commit is not a full object id"),
    "candidate is its own source": (_picks(lambda s: pick_record(s.candidate, s.candidate)),
                                    "pick_records[0] names one commit as both candidate and upstream commit"),
    "duplicate record": (_picks(lambda s: pick_record(s.candidate, s.source),
                                lambda s: pick_record(s.candidate, s.source)),
                         "duplicate or conflicting pick records"),
    "conflicting records": (_picks(lambda s: pick_record(s.candidate, s.source),
                                   lambda s: pick_record(s.candidate, s.forge.base)),
                            "duplicate or conflicting pick records"),
    "candidate missing locally": (_picks(lambda s: pick_record("1" * 40, s.source)),
                                  "pick candidate 1111111111111111111111111111111111111111 is missing, "
                                  "not a commit in this repository"),
    "upstream commit is a tree": (_picks(lambda s: pick_record(
        s.candidate, s.forge.out("rev-parse", f"{s.source}^{{tree}}"))),
        "is a tree, not a commit in this repository"),
    "tracking ref": (_picks(lambda s: pick_record(s.candidate, s.source, ref="refs/remotes/upstream/main")),
                     "pick_records[0] ref must be a refs/heads/, refs/tags/ or refs/pull/<n>/head ref"),
    "pull merge ref": (_picks(lambda s: pick_record(s.candidate, s.source, ref="refs/pull/7/merge")),
                       "pick_records[0] ref must be a refs/heads/, refs/tags/ or refs/pull/<n>/head ref"),
    "remote name source": (_picks(lambda s: pick_record(s.candidate, s.source, source="upstream")),
                           "pick_records[0] source is not an approved authenticated URL"),
    "local path source": (_picks(lambda s: pick_record(s.candidate, s.source, source=str(s.up.bare))),
                          "pick_records[0] source is not an approved authenticated URL"),
    "other repository": (_picks(lambda s: pick_record(s.candidate, s.source, repository="someone/hermes-agent")),
                         "pick_records[0] repository is not the canonical NousResearch/hermes-agent"),
    "bad time": (_picks(lambda s: pick_record(s.candidate, s.source, verified_at="2026-11-12")),
                 "pick_records[0] verified_at must be"),
}


@pytest.mark.parametrize("case", sorted(BROKEN_PICKS))
def test_broken_v3_bundle_refuses_a_clean_sim_push(scene, case):
    make, needle = BROKEN_PICKS[case]
    scene.forge.write_policy(make(scene))
    assert_refused(scene.forge, scene.push(scene.fork), needle)


def _write_compact(forge: Forge, policy: dict, pad: int = 0) -> int:
    raw = json.dumps(policy, separators=(",", ":")).encode() + b" " * pad
    forge.write_policy(None, raw=raw)
    return len(raw)


@pytest.mark.parametrize("n,needle", [
    (256, None), (257, "'pick_records' must be a list of at most 256 records"),
], ids=["256-records", "257-records"])
def test_pick_record_cap(forge, n, needle):
    picks = [pick_record(oid, forge.base, source="git@github.com:NousResearch/hermes-agent", note="n")
             for oid in _local_commits(forge, n)]
    assert _write_compact(forge, bundle(pick_records=picks)) < 256 * 1024, \
        "the cap case must not hit the bundle size limit instead"
    clean = forge.commit("clean", [forge.base], files={"a.txt": "a\n"}, ref="refs/heads/main")
    res = forge.push("main")
    if needle is None:
        assert_accepted(forge, res, "refs/heads/main", clean)
    else:
        assert_refused(forge, res, needle)


def test_bundle_over_the_size_limit_refuses(forge):
    size = _write_compact(forge, bundle(), pad=256 * 1024)
    assert size > 256 * 1024
    forge.commit("clean", [forge.base], files={"a.txt": "a\n"}, ref="refs/heads/main")
    assert_refused(forge, forge.push("main"), "private identity policy exceeds 262144 bytes")


# ── a real race: the receiver checks the advertised old OID, even on a force push ─────────
def _racing_hook(forge: Forge, racer: str) -> None:
    """Wrap the real hook: run it unchanged, then (only if it passed) move the bare remote's
    `main` the way a concurrent pusher would, after git advertised the old OID and before git
    sends its update."""
    hook = forge.hook_dir / "pre-push"
    hook.rename(forge.hook_dir / "pre-push.guard")
    hook.write_text("#!/bin/sh\n"
                    '"$(dirname "$0")/pre-push.guard" "$@" || exit 1\n'
                    f'git --git-dir="{forge.remote}" update-ref refs/heads/main {racer}\n')
    hook.chmod(0o755)


@pytest.mark.parametrize("force", [False, True], ids=["plain", "force"])
def test_receiver_refuses_an_update_whose_advertised_old_oid_moved(forge, force):
    racer = forge.commit("concurrent push", [forge.base], files={"r.txt": "r\n"})
    forge.git("push", "-q", "--no-verify", "origin", f"{racer}:refs/heads/racer")
    mine = forge.commit("mine", [forge.base], files={"a.txt": "a\n"}, ref="refs/heads/main")
    _racing_hook(forge, racer)
    assert forge.remote_refs()["refs/heads/main"] == forge.base
    res = forge.git("push", "--porcelain", *(["--force"] if force else []), "origin", "main",
                    check=False)
    out, err = res.stdout.decode(), res.stderr.decode(errors="replace")
    assert "pre-push identity: passed" in err, f"the real guard did not run and pass:\n{err}"
    assert res.returncode != 0, f"the raced push was accepted:\n{out}\n{err}"
    assert "[remote rejected] (failed to update ref)" in out, out
    assert f"is at {racer} but expected {forge.base}" in err, err
    assert forge.remote_refs()["refs/heads/main"] == racer != mine


def test_racing_hook_without_a_race_passes(forge):
    """Control: the wrapper itself does not refuse when the remote stays where it was."""
    mine = forge.commit("mine", [forge.base], files={"a.txt": "a\n"}, ref="refs/heads/main")
    _racing_hook(forge, forge.base)
    assert_accepted(forge, forge.push("main"), "refs/heads/main", mine)


def test_cherry_pick_helpers_build_a_real_pick(scene):
    """The fixture's candidate is what `git cherry-pick -x` writes: source author, sim
    committer, the trailer, the fork commit as parent and the source's change on its tree."""
    show = scene.forge.out("show", "-s", "--format=%an <%ae>|%cn <%ce>|%P|%B", scene.candidate)
    author, committer, parent, body = show.split("|", 3)
    assert (author, committer, parent) == (f"{UPSTREAM[0]} <{UPSTREAM[1]}>", f"{SIM[0]} <{SIM[1]}>",
                                           scene.fork)
    assert f"(cherry picked from commit {scene.source})" in body
    assert scene.forge.out("cat-file", "blob", f"{scene.candidate}:up.txt") == "upstream fix"
