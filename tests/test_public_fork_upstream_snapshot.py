"""Behaviour contract for e2 of the fork pre-push identity gate: pinned upstream snapshots and
annotated tags (`.githooks/pre-push` + `.githooks/identity_guard.py`, policy schema v2).

A verified upstream commit keeps its own author and committer when it is reachable from an
exact snapshot tip that TARS pinned in the private bundle; an annotated tag passes when every
tag object in its peel chain is tagged by sim or is named by an exact tag record. Everything
else that is not sim still refuses, and verified upstream objects are still scanned against
the deny-list.

Every push is a real `git push` into a disposable bare remote (the `Forge` of the e1 contract),
with the hook copied from this checkout. The canonical upstream is stood in for by a second
disposable bare repository whose history is fetched into the clone; records name the public
canonical repository `NousResearch/hermes-agent`, as TARS's real records do, and the test
never reaches the network. Deny-list values are synthetic (`Pat Example`,
`pat@example.invalid`).
"""

from __future__ import annotations

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

pytestmark = [pytest.mark.platforms("linux")]

CANONICAL = "NousResearch/hermes-agent"
SOURCE = "https://github.com/NousResearch/hermes-agent.git"
VERIFIED_AT = "2026-11-12T09:30:00Z"
NOT_VERIFIED = "author is not the fork identity sim and the commit is not in a verified upstream snapshot"
TAG_UNRECORDED = "tagger is not the fork identity sim and no tag record names this exact tag object"
REWRITE = "rewrite the unpublished commits"


def snapshot(*tips: str, **over) -> dict:
    return {"tips": list(tips), "repository": CANONICAL, "source": SOURCE,
            "ref": "refs/heads/main", "verified_at": VERIFIED_AT,
            "note": "fetched from the canonical upstream and compared by TARS", **over}


def tag_record(oid: str, name: str = "v1", **over) -> dict:
    return {"tag": oid, "repository": CANONICAL, "source": SOURCE, "ref": f"refs/tags/{name}",
            "verified_at": VERIFIED_AT, "note": "tag object compared by TARS", **over}


def bundle(snaps=(), recs=(), **over) -> dict:
    return {"version": 2, "names": [PERSONAL[0]], "emails": [PERSONAL[1]],
            "snapshots": list(snaps), "tag_records": list(recs), **over}


class Upstream:
    """Stand-in for the canonical upstream: a disposable bare repository whose non-sim history
    starts at the fork's base and is fetched into the fork clone (`refs/remotes/upstream/*`)."""

    def __init__(self, forge: Forge):
        self.f = forge
        self.bare = forge.root / "upstream.git"
        forge.git("init", "-q", "--bare", "-b", "main", str(self.bare), cwd=forge.root)
        forge.git("push", "-q", "--no-verify", str(self.bare), f"{forge.base}:refs/heads/main")
        self.tip = forge.base

    def git(self, *args, **kw):
        return self.f.git(*args, cwd=self.bare, **kw)

    def out(self, *args, **kw) -> str:
        return self.git(*args, **kw).stdout.decode().strip()

    def commit(self, msg="upstream work", author=UPSTREAM, committer=UPSTREAM, date=DATE,
               parent: str | None = None) -> str:
        parent = parent or self.tip
        idents = {"GIT_AUTHOR_NAME": author[0], "GIT_AUTHOR_EMAIL": author[1],
                  "GIT_COMMITTER_NAME": committer[0], "GIT_COMMITTER_EMAIL": committer[1],
                  "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
        tree = self.out("rev-parse", f"{parent}^{{tree}}")
        oid = self.out("commit-tree", tree, "-p", parent, "-F", "-", input=msg.encode(), env=idents)
        return self.advance(oid)

    def raw_commit(self, raw: bytes) -> str:
        return self.advance(self.out("hash-object", "-t", "commit", "-w", "--literally", "--stdin",
                                     input=raw))

    def advance(self, oid: str) -> str:
        self.git("update-ref", f"refs/heads/kept/{oid}", oid)  # every commit stays fetchable
        self.git("update-ref", "refs/heads/main", oid)
        self.tip = oid
        return oid

    def tag(self, name: str, target: str, tagger=UPSTREAM, msg="upstream release") -> str:
        self.git("tag", "-a", name, "-m", msg, target,
                 env={"GIT_COMMITTER_NAME": tagger[0], "GIT_COMMITTER_EMAIL": tagger[1]})
        return self.out("rev-parse", f"refs/tags/{name}")

    def fetch(self):
        self.f.git("fetch", "-q", str(self.bare), "+refs/heads/*:refs/remotes/upstream/*",
                   "+refs/tags/*:refs/tags/*")


@pytest.fixture
def forge(root):
    return Forge(root, policy=bundle())


@pytest.fixture
def up(forge):
    return Upstream(forge)


def published(forge: Forge, up: Upstream, *, pins=None, records=()) -> None:
    """Fetch the stand-in upstream and install a bundle pinning `pins` (default: its tip)."""
    up.fetch()
    forge.write_policy(bundle([snapshot(*(pins or [up.tip]))], records))


# ── verified upstream commits ──────────────────────────────────────────────────────────────
def test_verified_upstream_commit_passes(forge, up):
    up.commit("upstream one")
    tip = up.commit("upstream two")
    published(forge, up)
    forge.git("update-ref", "refs/heads/main", tip)
    assert_accepted(forge, forge.push("main"), "refs/heads/main", tip)


def test_sim_merge_of_verified_upstream_passes(forge, up):
    tip = up.commit("upstream one")
    published(forge, up)
    mine = forge.commit("fork work", [forge.base], files={"a.txt": "a\n"})
    merge = forge.commit("merge upstream", [mine, tip], ref="refs/heads/main")
    assert_accepted(forge, forge.push("main"), "refs/heads/main", merge)


def test_upstream_commit_reachable_only_from_a_tracking_ref_refuses(forge, up):
    """A fetched `upstream` remote-tracking ref is a mutable local name, not approval."""
    tip = up.commit("upstream one")
    up.fetch()
    forge.git("update-ref", "refs/heads/main", tip)
    res = forge.push("main")
    assert_refused(forge, res, NOT_VERIFIED)
    said = forge.said(res.stderr)
    assert tip in said and "TARS" in said and "never writes approval" in said, said
    assert REWRITE not in said, f"a record-based refusal printed the rewrite remedy:\n{said}"


def test_unverified_lookalike_of_a_pinned_commit_refuses(forge, up):
    """Same tree, parent, identities and message as the pinned commit, another object."""
    real = up.commit("upstream one")
    published(forge, up)
    lookalike = up.commit("upstream one", date="1700000001 +0000", parent=forge.base)
    assert lookalike != real
    up.fetch()
    forge.git("update-ref", "refs/heads/main", lookalike)
    assert_refused(forge, forge.push("main"), NOT_VERIFIED)


def test_upstream_commit_beyond_the_pinned_tip_refuses(forge, up):
    pinned = up.commit("upstream one")
    later = up.commit("upstream two")
    published(forge, up, pins=[pinned])
    forge.git("update-ref", "refs/heads/main", later)
    assert_refused(forge, forge.push("main"), NOT_VERIFIED)


def test_pin_changed_to_an_unrelated_upstream_commit_refuses(forge, up):
    first = up.commit("upstream one")
    other = up.commit("upstream elsewhere", parent=forge.base)
    published(forge, up, pins=[other])
    forge.git("update-ref", "refs/heads/main", first)
    assert_refused(forge, forge.push("main"), NOT_VERIFIED)


def test_removing_the_snapshot_revokes_approval(forge, up):
    tip = up.commit("upstream one")
    published(forge, up)
    forge.write_policy(bundle())
    forge.git("update-ref", "refs/heads/main", tip)
    assert_refused(forge, forge.push("main"), NOT_VERIFIED)


@pytest.mark.parametrize("pin,needle", [
    (lambda f, up: "1" * 40, "is missing, not a commit in this repository"),
    (lambda f, up: f.out("rev-parse", f"{up.tip}^{{tree}}"), "is a tree, not a commit in this repository"),
], ids=["missing", "tree"])
def test_snapshot_tip_that_is_not_a_local_commit_refuses(forge, up, pin, needle):
    tip = up.commit("upstream one")
    up.fetch()
    forge.write_policy(bundle([snapshot(tip, pin(forge, up))]))
    forge.git("update-ref", "refs/heads/main", tip)
    assert_refused(forge, forge.push("main"), needle)


@pytest.mark.parametrize("author,committer,msg,needle", [
    (("Someone Else", PERSONAL[1]), UPSTREAM, "work", "author email matches the private deny-list"),
    ((PERSONAL[0], "other@upstream.example"), UPSTREAM, "work",
     "author name matches the private deny-list"),
    (UPSTREAM, ("Another Dev", PERSONAL[1]), "work", "committer email matches the private deny-list"),
    (UPSTREAM, UPSTREAM, "thanks Pat Example", "message matches the private deny-list"),
], ids=["author-email", "author-name-other-email", "committer-email", "message-name"])
def test_verified_upstream_deny_list_collision_refuses(forge, up, author, committer, msg, needle):
    tip = up.commit(msg, author=author, committer=committer)
    published(forge, up)
    forge.git("update-ref", "refs/heads/main", tip)
    res = forge.push("main")
    assert_refused(forge, res, needle)
    assert REWRITE not in forge.said(res.stderr), "verified upstream history cannot be rewritten"


def test_verified_upstream_non_utf8_author_name_refuses(forge, up):
    tree = up.out("rev-parse", f"{up.tip}^{{tree}}")
    tip = up.raw_commit(f"tree {tree}\nparent {up.tip}\n".encode()
                        + b"author Upstr\xffeam Dev <dev@upstream.example> " + DATE.encode()
                        + f"\ncommitter {UPSTREAM[0]} <{UPSTREAM[1]}> {DATE}\n\nwork\n".encode())
    published(forge, up)
    forge.git("update-ref", "refs/heads/main", tip)
    assert_refused(forge, forge.push("main"), "author header is not valid UTF-8")


def test_verified_upstream_commit_with_an_empty_email_passes(forge, up):
    """Canonical upstream history has `Name <> time tz` identities, which git itself accepts."""
    tree = up.out("rev-parse", f"{up.tip}^{{tree}}")
    tip = up.raw_commit(f"tree {tree}\nparent {up.tip}\nauthor Upstream Dev <> {DATE}\n"
                        f"committer Upstream Bot <> {DATE}\n\nwork\n".encode())
    published(forge, up)
    forge.git("update-ref", "refs/heads/main", tip)
    assert_accepted(forge, forge.push("main"), "refs/heads/main", tip)


def test_fork_commit_with_an_empty_email_refuses(forge):
    tree = forge.out("rev-parse", f"{forge.base}^{{tree}}")
    forge.raw_commit(f"tree {tree}\nparent {forge.base}\nauthor {SIM[0]} <> {DATE}\n"
                     f"committer {SIM[0]} <{SIM[1]}> {DATE}\n\nwork\n".encode(), "refs/heads/main")
    assert_refused(forge, forge.push("main"), NOT_VERIFIED)


def test_personal_fork_commit_still_gets_the_rewrite_remedy(forge, up):
    up.commit("upstream one")
    published(forge, up)
    forge.commit("mine", [forge.base], author=PERSONAL, files={"a.txt": "a\n"}, ref="refs/heads/main")
    res = forge.push("main")
    assert_refused(forge, res, "author email matches the private deny-list")
    assert REWRITE in forge.said(res.stderr)


# ── annotated tags ─────────────────────────────────────────────────────────────────────────
def test_verified_annotated_upstream_tag_passes(forge, up):
    tip = up.commit("upstream release")
    tag = up.tag("v1", tip)
    published(forge, up, records=[tag_record(tag)])
    assert_accepted(forge, forge.push("refs/tags/v1"), "refs/tags/v1", tag)


def test_unrecorded_upstream_tag_refuses(forge, up):
    tip = up.commit("upstream release")
    tag = up.tag("v1", tip)
    published(forge, up)
    res = forge.push("refs/tags/v1")
    assert_refused(forge, res, TAG_UNRECORDED)
    said = forge.said(res.stderr)
    assert tag in said and "TARS" in said, said
    assert REWRITE not in said, said


def test_lookalike_tag_with_the_recorded_name_refuses(forge, up):
    tip = up.commit("upstream release")
    real = up.tag("v1", tip)
    up.fetch()
    up.git("tag", "-d", "v1")
    lookalike = up.tag("v1", tip, msg="upstream release (re-tagged)")
    assert lookalike != real
    published(forge, up, records=[tag_record(real)])
    assert forge.out("rev-parse", "refs/tags/v1") == lookalike
    assert_refused(forge, forge.push("refs/tags/v1"), TAG_UNRECORDED)


def test_tag_record_naming_another_tag_refuses(forge, up):
    tip = up.commit("upstream release")
    tag = up.tag("v1", tip)
    published(forge, up, records=[tag_record(tag, name="v2")])
    assert_refused(forge, forge.push("refs/tags/v1"), "the tag record for this tag object names a different tag")


def test_recorded_tag_of_an_unverified_commit_refuses(forge, up):
    tip = up.commit("upstream release")
    tag = up.tag("v1", tip)
    up.fetch()
    forge.write_policy(bundle(recs=[tag_record(tag)]))
    assert_refused(forge, forge.push("refs/tags/v1"), NOT_VERIFIED)


@pytest.mark.parametrize("tagger,msg,needle", [
    (("Someone Else", PERSONAL[1]), "release", "tagger email matches the private deny-list"),
    ((PERSONAL[0], "other@upstream.example"), "release", "tagger name matches the private deny-list"),
    (UPSTREAM, "release by Pat Example", "tag message matches the private deny-list"),
], ids=["tagger-email", "tagger-name", "message"])
def test_recorded_tag_deny_list_collision_refuses(forge, up, tagger, msg, needle):
    tip = up.commit("upstream release")
    tag = up.tag("v1", tip, tagger=tagger, msg=msg)
    published(forge, up, records=[tag_record(tag)])
    assert_refused(forge, forge.push("refs/tags/v1"), needle)


def test_personal_tagger_on_sim_commit_refuses(forge):
    forge.commit("tagged", [forge.base], files={"a.txt": "a\n"}, ref="refs/heads/main")
    forge.git("tag", "-a", "v1", "-m", "release", "main",
              env={"GIT_COMMITTER_NAME": UPSTREAM[0], "GIT_COMMITTER_EMAIL": UPSTREAM[1]})
    assert_refused(forge, forge.push("refs/tags/v1"), TAG_UNRECORDED)


def test_tag_of_a_tag_peels_recursively_and_passes(forge):
    forge.commit("tagged", [forge.base], files={"a.txt": "a\n"}, ref="refs/heads/main")
    sim = {"GIT_COMMITTER_NAME": SIM[0], "GIT_COMMITTER_EMAIL": SIM[1]}
    forge.git("tag", "-a", "inner", "-m", "inner", "main", env=sim)
    forge.git("tag", "-a", "outer", "-m", "outer", "refs/tags/inner", env=sim)
    outer = forge.out("rev-parse", "refs/tags/outer")
    assert_accepted(forge, forge.push("refs/tags/outer"), "refs/tags/outer", outer)


def test_inner_tag_of_a_nested_chain_is_audited(forge):
    forge.commit("tagged", [forge.base], files={"a.txt": "a\n"}, ref="refs/heads/main")
    forge.git("tag", "-a", "inner", "-m", "inner", "main",
              env={"GIT_COMMITTER_NAME": UPSTREAM[0], "GIT_COMMITTER_EMAIL": UPSTREAM[1]})
    forge.git("tag", "-a", "outer", "-m", "outer", "refs/tags/inner",
              env={"GIT_COMMITTER_NAME": SIM[0], "GIT_COMMITTER_EMAIL": SIM[1]})
    assert_refused(forge, forge.push("refs/tags/outer"), TAG_UNRECORDED)


def _sim_tag(forge: Forge, name: str, target: str) -> str:
    forge.git("tag", "-a", name, "-m", "release", target,
              env={"GIT_COMMITTER_NAME": SIM[0], "GIT_COMMITTER_EMAIL": SIM[1]})
    return forge.out("rev-parse", f"refs/tags/{name}")


def test_tag_chain_deeper_than_the_cap_refuses(forge):
    target = forge.commit("tagged", [forge.base], files={"a.txt": "a\n"}, ref="refs/heads/main")
    for depth in range(17):
        target = _sim_tag(forge, f"t{depth}", target)
    assert_refused(forge, forge.push("refs/tags/t16"), "the tag chain is deeper than 16")


@pytest.mark.parametrize("target,kind", [
    (lambda f: f.out("hash-object", "-w", "--stdin", input=b"payload\n"), "blob"),
    (lambda f: f.out("rev-parse", f"{f.base}^{{tree}}"), "tree"),
    (lambda f: _sim_tag(f, "inner", f.out("hash-object", "-w", "--stdin", input=b"x\n")), "blob"),
], ids=["blob", "tree", "tag-of-blob"])
def test_tag_of_a_non_commit_refuses(forge, target, kind):
    _sim_tag(forge, "v1", target(forge))
    assert_refused(forge, forge.push("refs/tags/v1"), f"peels to a {kind}, not a commit")


def _raw_tag(forge: Forge, body: bytes) -> str:
    oid = forge.out("hash-object", "-t", "tag", "-w", "--literally", "--stdin", input=body)
    forge.git("update-ref", "refs/tags/v1", oid)
    return oid


@pytest.mark.parametrize("make,needle", [
    (lambda b: f"object {b}\ntype commit\ntag v1\n\nno tagger\n".encode(),
     "malformed tag: missing or malformed tagger header"),
    (lambda b: f"object {b}\ntype commit\ntag v1\ntagger sim sim@x {DATE}\n\nm\n".encode(),
     "malformed tag: missing or malformed tagger header"),
    (lambda b: f"object {b}\ntype tree\ntag v1\ntagger {SIM[0]} <{SIM[1]}> {DATE}\n\nm\n".encode(),
     "malformed tag: the type header does not match the tagged object"),
    (lambda b: f"object {b}\ntype commit\ntag v1\ntagger {SIM[0]} <{SIM[1]}> {DATE}\n\n".encode()
     + b"bad \xff\n", "tag message is not valid UTF-8"),
    (lambda b: f"object {b}\ntype commit\ntag v1\ntagger {SIM[0]} <{SIM[1]}> {DATE}\n\n".encode()
     + b"nul \x00\n", "tag message contains a NUL byte"),
    (lambda b: f"object {b}\ntype commit\ntag \ntagger {SIM[0]} <{SIM[1]}> {DATE}\n\nm\n".encode(),
     "malformed tag: bad object, type or tag header"),
], ids=["no-tagger", "tagger-shape", "type-mismatch", "not-utf8", "nul", "empty-name"])
def test_malformed_tag_object_refuses(forge, make, needle):
    _raw_tag(forge, make(forge.base))
    assert_refused(forge, forge.push("refs/tags/v1"), needle)


# ── a mergetag embeds a tag object: same tagger rule in fork commits ───────────────────────
def _merge_with_tag(forge: Forge, tag: str) -> str:
    """A sim merge whose `mergetag` header embeds the real tag object `tag`, as `git merge`
    writes it for a signed tag (one leading space per continuation line)."""
    body = forge.git("cat-file", "tag", tag).stdout
    target = forge.out("rev-parse", f"{tag}^{{commit}}")
    mine = forge.commit("fork work", [forge.base], files={"m.txt": "m\n"})
    embedded = b"mergetag " + body.rstrip(b"\n").replace(b"\n", b"\n ") + b"\n"
    head = forge.header(mine).replace(b"author ", f"parent {target}\nauthor ".encode(), 1)
    return forge.raw_commit(head + embedded + b"\nmerge tag\n", "refs/heads/main")


def test_mergetag_from_an_unrecorded_upstream_tag_refuses(forge, up):
    tag = up.tag("v1", up.commit("upstream release"))
    published(forge, up)
    _merge_with_tag(forge, tag)
    assert_refused(forge, forge.push("main"), f"mergetag {TAG_UNRECORDED}")


def test_mergetag_from_a_recorded_upstream_tag_passes(forge, up):
    tag = up.tag("v1", up.commit("upstream release"))
    published(forge, up, records=[tag_record(tag)])
    merge = _merge_with_tag(forge, tag)
    assert_accepted(forge, forge.push("main"), "refs/heads/main", merge)


# ── the v2 bundle: strict schema, exact records ────────────────────────────────────────────
def _with(snap=None, rec=None, **top):
    def build(forge: Forge, tip: str, tag: str) -> dict:
        snaps = [snapshot(tip, **(snap or {}))] if snap is not False else []
        recs = [tag_record(tag, **(rec or {}))] if rec is not None else []
        return bundle(snaps, recs, **top)
    return build


BROKEN_BUNDLE = {
    "v1 bundle": (lambda f, tip, tag: {"version": 1, "names": [PERSONAL[0]], "emails": [PERSONAL[1]]},
                  "version is not 2"),
    "no snapshots key": (lambda f, tip, tag: {k: v for k, v in bundle().items() if k != "snapshots"},
                         "lacks key(s) ['snapshots']"),
    "snapshots not a list": (_with(snapshots={}), "'snapshots' must be a list"),
    "duplicate snapshot tip": (lambda f, tip, tag: bundle([snapshot(tip), snapshot(tip, ref="refs/heads/next")]),
                               "duplicate snapshot tip"),
    "tip twice in one record": (lambda f, tip, tag: bundle([snapshot(tip, tip)]), "duplicate snapshot tip"),
    "conflicting snapshots": (lambda f, tip, tag: bundle([snapshot(tip), snapshot(f.base)]),
                              "conflicting snapshot records"),
    "duplicate tag record": (lambda f, tip, tag: bundle(recs=[tag_record(tag), tag_record(tag)]),
                             "duplicate or conflicting tag records"),
    "conflicting tag records": (lambda f, tip, tag: bundle(recs=[
        tag_record(tag), tag_record(tag, note="another review")]), "duplicate or conflicting tag records"),
    "two objects for one tag ref": (lambda f, tip, tag: bundle(recs=[
        tag_record(tag), tag_record(f.out("rev-parse", "refs/tags/v2"))]),
        "duplicate or conflicting tag records"),
    "snapshot unknown field": (_with(snap={"approved": True}), "snapshots[0] must have exactly the fields"),
    "snapshot missing field": (lambda f, tip, tag: bundle([{k: v for k, v in snapshot(tip).items() if k != "note"}]),
                               "snapshots[0] must have exactly the fields"),
    "no tips": (lambda f, tip, tag: bundle([snapshot()]), "snapshots[0] 'tips' must be a list of 1.."),
    "short tip": (lambda f, tip, tag: bundle([snapshot(tip[:12])]), "snapshots[0] tips[0] is not a full object id"),
    "other repository": (_with(snap={"repository": "someone/hermes-agent"}),
                         "snapshots[0] repository is not the canonical NousResearch/hermes-agent"),
    "local path source": (lambda f, tip, tag: bundle([snapshot(tip, source=str(f.root / "upstream.git"))]),
                          "snapshots[0] source is not an approved authenticated URL"),
    "remote name source": (_with(snap={"source": "upstream"}),
                           "snapshots[0] source is not an approved authenticated URL"),
    "plain http source": (_with(snap={"source": "http://github.com/NousResearch/hermes-agent.git"}),
                          "snapshots[0] source is not an approved authenticated URL"),
    "fork source": (_with(snap={"source": "https://github.com/someone/hermes-agent.git"}),
                    "snapshots[0] source is not an approved authenticated URL"),
    "tracking ref": (_with(snap={"ref": "refs/remotes/upstream/main"}),
                     "snapshots[0] ref must be a refs/heads/ or refs/tags/ ref"),
    "bad time": (_with(snap={"verified_at": "yesterday"}), "snapshots[0] verified_at must be"),
    "empty note": (_with(snap={"note": ""}), "snapshots[0] note must be a non-empty"),
    "tag record unknown field": (_with(rec={"approved": True}), "tag_records[0] must have exactly the fields"),
    "tag record branch ref": (_with(rec={"ref": "refs/heads/v1"}), "tag_records[0] ref must be a refs/tags/ ref"),
    "tag record names a commit": (lambda f, tip, tag: bundle(recs=[tag_record(tip)]),
                                  "is a commit, not a tag object in this repository"),
}


@pytest.mark.parametrize("case", sorted(BROKEN_BUNDLE))
def test_broken_v2_bundle_refuses_a_clean_sim_push(forge, up, case):
    tip = up.commit("upstream one")
    tag = up.tag("v1", tip)
    up.tag("v2", tip, msg="other")
    up.fetch()
    make, needle = BROKEN_BUNDLE[case]
    forge.write_policy(make(forge, tip, tag))
    forge.commit("clean", [forge.base], files={"a.txt": "a\n"}, ref="refs/heads/main")
    assert_refused(forge, forge.push("main"), needle)


def test_ssh_source_is_an_approved_authenticated_url(forge, up):
    tip = up.commit("upstream one")
    up.fetch()
    forge.write_policy(bundle([snapshot(tip, source="git@github.com:NousResearch/hermes-agent.git")]))
    forge.git("update-ref", "refs/heads/main", tip)
    assert_accepted(forge, forge.push("main"), "refs/heads/main", tip)
