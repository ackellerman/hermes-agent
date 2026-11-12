"""pre-push identity gate: audit the raw metadata of every commit a push would publish.

Run by `.githooks/pre-push` (same directory) BEFORE the host-path content gate, with the
pre-push stdin and `--block-patterns <verified block.ere>`. Exit 0 passes; anything else refuses.

WHY: the content gate reads diffs, not author/committer headers or commit messages, and it
skips history it believes a remote already has. A personal identity on the hidden parent of a
merge therefore went public through it. This gate enumerates, per pushed ref, every commit
reachable through ALL parents (`<new> ^<advertised old>`; a new ref audits its whole ancestry,
never excluding tracking refs), and checks each raw commit object:

  * author AND committer must be byte-exact `sim <sim@hermes-agent.nousresearch.com>`, unless
    the commit is VERIFIED UPSTREAM: reachable from an exact snapshot tip that TARS pinned in
    the private bundle. A verified upstream commit keeps its own author and committer. Every
    other identity fails closed (reviewed cherry-pick records are a later element);
  * headers must be well formed and UTF-8, the message UTF-8 without NUL bytes;
  * every class, verified upstream included, is scanned with the public block pattern and the
    private deny-list. A header (author, committer, tagger) refuses when its email is the
    operator email token or its name is the operator's full name (whatever email goes with
    it); messages refuse on the full name (case-insensitive, word-bounded phrase) or the email
    token, never on substrings. Every header after committer (mergetag, gpgsig, ...) must be
    UTF-8 and its whole text, continuation lines included, is scanned like the message;
    matched values are never printed.

An annotated tag is peeled recursively. Every tag object in the chain must be well formed
(object, type, tag name, tagger), its declared type must be the tagged object's real type, its
tagger and message obey the header/message rules above, and its tagger must be sim unless an
exact tag record names that tag object (and its tag name). The chain must end at a commit,
whose ancestry is audited like a branch. A `mergetag` in a fork commit embeds a tag object and
obeys the same tag rules; in a verified upstream commit it is only scanned.

The private bundle lives ONLY in `<git common dir>/identity-policy/policy.json` (never a
worktree or hook-directory copy). Schema v2 is exactly
`{"version": 2, "names": [...], "emails": [...], "snapshots": [...], "tag_records": [...]}`:
`names`/`emails` are the non-empty deny-list; a snapshot record is
`{"tips": [<full commit oid>, ...], "repository": "NousResearch/hermes-agent", "source": <https
or ssh URL of that repository on github.com>, "ref": "refs/heads/..|refs/tags/..",
"verified_at": "YYYY-MM-DDTHH:MM:SSZ", "note": <TARS's verification note>}`; a tag record has
the same provenance fields with `"tag": <full tag-object oid>` instead of `tips` and a
`refs/tags/` ref. Every pinned tip must be a local commit and every recorded tag a local tag
object. A missing, malformed, duplicate-keyed, other-version, symlinked, foreign-owned or
group/world-writable bundle refuses, and so do duplicate or conflicting records. The check
never writes approval: TARS verifies the canonical upstream independently and edits the bundle.

Limits: 250,000 commits and 180 s for the whole gate, fail closed. `git -c
identityguard.maxCommits=N` / `identityguard.timeoutSeconds=S` can only LOWER them (tests use
this); a malformed value refuses. Threat model: ordinary-operation accidents; `--no-verify`
and a replaced hook bypass any local hook. `HERMES_ALLOW_HOST_PATHS=1` never reaches this gate.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import threading
import time
from datetime import datetime

SIM_NAME = b"sim"
SIM_EMAIL = b"sim@hermes-agent.nousresearch.com"
MAX_COMMITS = 250_000
TIME_LIMIT = 180.0
POLICY_VERSION = 2
POLICY_MAX_BYTES = 64 * 1024
POLICY_MAX_ENTRIES = 32
MAX_RECORDS = 256
MAX_TIPS = 32
MAX_TAG_DEPTH = 16
CANONICAL_REPO = "NousResearch/hermes-agent"
# Authenticated transports to the canonical repository only: never a local path, a remote
# name, file://, plain http:// or git://, another host or another owner's fork.
APPROVED_SOURCE = re.compile(r"(?:https://github\.com/|ssh://git@github\.com/|git@github\.com:)"
                             r"NousResearch/hermes-agent(?:\.git)?")
SNAPSHOT_FIELDS = {"tips", "repository", "source", "ref", "verified_at", "note"}
TAG_RECORD_FIELDS = {"tag", "repository", "source", "ref", "verified_at", "note"}
REF_PART = r"[A-Za-z0-9_][A-Za-z0-9._-]*"
SNAPSHOT_REF = re.compile(rf"refs/(?:heads|tags)/{REF_PART}(?:/{REF_PART})*")
TAG_REF = re.compile(rf"refs/tags/{REF_PART}(?:/{REF_PART})*")
UTC_TIME = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
FULL_OID = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
NOT_VERIFIED = "is not the fork identity sim and the commit is not in a verified upstream snapshot"
TAG_UNRECORDED = ("tagger is not the fork identity sim and no tag record names this exact tag "
                  "object")
MAX_STDIN = 16 * 1024 * 1024
MAX_REPORTED = 20
OBJECT_HEX = {"sha1": 40, "sha256": 64}
PREFIX = "pre-push identity:"

GIT_ENV = dict(os.environ, GIT_NO_LAZY_FETCH="1", GIT_NO_REPLACE_OBJECTS="1",
               GIT_TERMINAL_PROMPT="0", GIT_PAGER="cat", LC_ALL="C")
# The email may be empty (`Name <> ...`), as git itself allows: canonical upstream history has it.
IDENT_RE = re.compile(rb"(author|committer|tagger) ([^<>\n]+) <([^<>\n]*)> ([0-9]+) ([+-][0-9]{4})")
EMAIL_SHAPE = re.compile(r"[^\s@<>]+@[^\s@<>]+\.[^\s@<>]+")
UTF8_NAMES = {b"utf-8", b"utf8"}
KNOWN_HEADERS = {b"tree", b"parent", b"author", b"committer", b"encoding", b"mergetag", b"gpgsig",
                 b"gpgsig-sha256"}


class GuardError(Exception):
    """A refusal; the message must never contain a private deny-list value."""


class Clock:
    def __init__(self, limit: float, start: float):
        self.limit = limit
        self.deadline = start + limit

    def remaining(self) -> float:
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise GuardError(f"the {self.limit:g}s time limit was exceeded; refusing (fail closed)")
        return left


def git(args: list[str], clock: Clock, data: bytes | None = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["git", *args], input=data, capture_output=True, env=GIT_ENV,
                              timeout=clock.remaining(), check=False)
    except subprocess.TimeoutExpired:
        raise GuardError(f"the {clock.limit:g}s time limit was exceeded during git "
                         f"{args[0]}; refusing (fail closed)") from None


# ── limits ─────────────────────────────────────────────────────────────────────────────────
def _knob(name: str, clock: Clock) -> str | None:
    res = git(["config", "--get", f"identityguard.{name}"], clock)
    if res.returncode == 1:
        return None
    if res.returncode != 0:
        raise GuardError(f"cannot read identityguard.{name}; refusing (fail closed)")
    return res.stdout.decode(errors="replace").strip()


def read_limits(clock: Clock) -> tuple[int, float]:
    """Return (max commits, seconds). Knobs only lower the defaults; garbage refuses."""
    max_commits, seconds = MAX_COMMITS, TIME_LIMIT
    raw = _knob("maxCommits", clock)
    if raw is not None:
        if not re.fullmatch(r"[0-9]+", raw) or int(raw) <= 0:
            raise GuardError("identityguard.maxCommits must be a positive integer; refusing")
        max_commits = min(max_commits, int(raw))
    raw = _knob("timeoutSeconds", clock)
    if raw is not None:
        try:
            value = float(raw)
        except ValueError:
            value = math.nan
        if not math.isfinite(value) or value <= 0:
            raise GuardError("identityguard.timeoutSeconds must be a positive number; refusing")
        seconds = min(seconds, value)
    return max_commits, seconds


# ── private policy bundle ──────────────────────────────────────────────────────────────────
def check_node_type(path, st: os.stat_result, want: str) -> None:
    """Pre-open lstat check: no symlink, and the right type (never open a FIFO or device)."""
    if stat.S_ISLNK(st.st_mode):
        raise GuardError(f"private identity policy path {path} is a symlink; refusing")
    if want == "dir" and not stat.S_ISDIR(st.st_mode):
        raise GuardError(f"private identity policy path {path} is not a directory; refusing")
    if want == "file" and not stat.S_ISREG(st.st_mode):
        raise GuardError(f"private identity policy path {path} is not a regular file; refusing")


def check_private_node(path, st: os.stat_result, *, euid: int, want: str) -> None:
    """Post-open fstat check of what was actually opened: type, owner and mode."""
    check_node_type(path, st, want)
    if st.st_uid != euid:
        raise GuardError(f"private identity policy path {path} is owned by uid {st.st_uid}, "
                         f"not by the pushing user (uid {euid}); refusing")
    if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise GuardError(f"private identity policy path {path} is group- or world-writable "
                         f"(mode {stat.S_IMODE(st.st_mode):o}); refusing")


def common_dir(clock: Clock) -> str:
    res = git(["rev-parse", "--path-format=absolute", "--git-common-dir"], clock)
    path = os.fsdecode(res.stdout.rstrip(b"\n"))
    if res.returncode != 0 or not path or not os.path.isabs(path):
        raise GuardError("cannot resolve an absolute git common dir for the private identity "
                         "policy; refusing (fail closed)")
    return path


def _no_duplicate_keys(pairs):
    keys = [k for k, _ in pairs]
    if len(set(keys)) != len(keys):
        raise GuardError("private identity policy has a duplicate JSON key; refusing")
    return dict(pairs)


def _reject_constant(token):
    raise GuardError(f"private identity policy contains the non-JSON constant {token}; refusing")


def _entries(policy: dict, key: str) -> list[str]:
    values = policy[key]
    if not isinstance(values, list) or not 1 <= len(values) <= POLICY_MAX_ENTRIES:
        raise GuardError(f"private identity policy '{key}' must be a list of 1..{POLICY_MAX_ENTRIES} "
                         "entries; refusing")
    for i, value in enumerate(values):
        if not isinstance(value, str) or not value or len(value) > 254 or not value.isprintable():
            raise GuardError(f"private identity policy {key}[{i}] is not a usable string; refusing")
    if len({v.casefold() for v in values}) != len(values):
        raise GuardError(f"private identity policy '{key}' has duplicate entries; refusing")
    return values


class Bundle:
    """The validated private bundle. Record values other than object ids are never printed."""

    def __init__(self, names: list[str], emails: list[str], tips: frozenset[str],
                 tag_records: dict[str, str]):
        self.names, self.emails = names, emails
        self.tips = tips                # every pinned snapshot tip
        self.tag_records = tag_records  # exact tag-object oid -> the recorded refs/tags/<name>


def _records(policy: dict, key: str) -> list:
    values = policy[key]
    if not isinstance(values, list) or len(values) > MAX_RECORDS:
        raise GuardError(f"private identity policy '{key}' must be a list of at most {MAX_RECORDS} "
                         "records; refusing")
    return values


def _oid(value, label: str) -> str:
    if not isinstance(value, str) or not FULL_OID.fullmatch(value):
        raise GuardError(f"private identity policy {label} is not a full object id; refusing")
    return value


def _provenance(rec, label: str, fields: set, ref_re: re.Pattern, ref_kind: str) -> None:
    """The review provenance every record carries; the values themselves are never printed."""
    if not isinstance(rec, dict) or set(rec) != fields:
        raise GuardError(f"private identity policy {label} must have exactly the fields "
                         f"{sorted(fields)}; refusing")
    if rec["repository"] != CANONICAL_REPO:
        raise GuardError(f"private identity policy {label} repository is not the canonical "
                         f"{CANONICAL_REPO}; refusing")
    if not isinstance(rec["source"], str) or not APPROVED_SOURCE.fullmatch(rec["source"]):
        raise GuardError(f"private identity policy {label} source is not an approved authenticated "
                         f"URL of {CANONICAL_REPO} (https:// or ssh to github.com); a local path, "
                         "a remote name or a tracking ref is not a source; refusing")
    if not isinstance(rec["ref"], str) or not ref_re.fullmatch(rec["ref"]) or ".." in rec["ref"]:
        raise GuardError(f"private identity policy {label} ref must be a {ref_kind} ref on the "
                         "canonical source; refusing")
    try:
        ok = bool(UTC_TIME.fullmatch(rec["verified_at"]))
        ok = ok and bool(datetime.strptime(rec["verified_at"], "%Y-%m-%dT%H:%M:%SZ"))
    except (TypeError, ValueError):
        ok = False
    if not ok:
        raise GuardError(f"private identity policy {label} verified_at must be a UTC time like "
                         "2026-01-31T12:00:00Z; refusing")
    note = rec["note"]
    if (not isinstance(note, str) or not note.strip() or len(note) > 2000
            or not all(c.isprintable() or c in "\n\t" for c in note)):
        raise GuardError(f"private identity policy {label} note must be a non-empty review note; "
                         "refusing")


def _snapshot_tips(policy: dict) -> frozenset[str]:
    tips: set[str] = set()
    fetches: dict[tuple, tuple] = {}
    for i, rec in enumerate(_records(policy, "snapshots")):
        label = f"snapshots[{i}]"
        _provenance(rec, label, SNAPSHOT_FIELDS, SNAPSHOT_REF, "refs/heads/ or refs/tags/")
        values = rec["tips"]
        if not isinstance(values, list) or not 1 <= len(values) <= MAX_TIPS:
            raise GuardError(f"private identity policy {label} 'tips' must be a list of 1..{MAX_TIPS} "
                             "object ids; refusing")
        for n, value in enumerate(values):
            oid = _oid(value, f"{label} tips[{n}]")
            if oid in tips:
                raise GuardError(f"private identity policy has a duplicate snapshot tip ({oid}); "
                                 "refusing")
            tips.add(oid)
        fetch, pinned = (rec["ref"], rec["verified_at"]), tuple(sorted(values))
        if fetches.setdefault(fetch, pinned) != pinned:
            raise GuardError("private identity policy has conflicting snapshot records (one fetch of "
                             f"one ref recorded with different tips, {label}); refusing")
    return frozenset(tips)


def _tag_records(policy: dict) -> dict[str, str]:
    by_oid: dict[str, str] = {}
    by_ref: dict[str, str] = {}
    for i, rec in enumerate(_records(policy, "tag_records")):
        label = f"tag_records[{i}]"
        _provenance(rec, label, TAG_RECORD_FIELDS, TAG_REF, "refs/tags/")
        oid = _oid(rec["tag"], f"{label} tag")
        if oid in by_oid or by_ref.setdefault(rec["ref"], oid) != oid:
            raise GuardError(f"private identity policy has duplicate or conflicting tag records "
                             f"({label}); refusing")
        by_oid[oid] = rec["ref"]
    return by_oid


def parse_policy(raw: bytes) -> Bundle:
    try:
        text = raw.decode("utf-8")
        policy = json.loads(text, object_pairs_hook=_no_duplicate_keys,
                            parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError):
        raise GuardError("private identity policy is not valid UTF-8 JSON; refusing") from None
    if not isinstance(policy, dict):
        raise GuardError("private identity policy must be a JSON object; refusing")
    version = policy.get("version")
    if type(version) is not int or version != POLICY_VERSION:
        raise GuardError(f"private identity policy version is not {POLICY_VERSION}; this hook "
                         f"accepts exactly schema v{POLICY_VERSION}; refusing")
    expected = {"version", "names", "emails", "snapshots", "tag_records"}
    if set(policy) - expected:
        raise GuardError(f"private identity policy has unknown key(s) "
                         f"{sorted(set(policy) - expected)}; refusing (schema v{POLICY_VERSION})")
    if expected - set(policy):
        raise GuardError(f"private identity policy lacks key(s) {sorted(expected - set(policy))}; "
                         "refusing")
    names, emails = _entries(policy, "names"), _entries(policy, "emails")
    for i, name in enumerate(names):
        words = name.split()
        if " ".join(words) != name or len(words) < 2 or not all(any(c.isalpha() for c in w) for w in words):
            raise GuardError(f"private identity policy names[{i}] is not a full name; refusing")
    for i, email in enumerate(emails):
        if not EMAIL_SHAPE.fullmatch(email):
            raise GuardError(f"private identity policy emails[{i}] is not an email; refusing")
    return Bundle(names, emails, _snapshot_tips(policy), _tag_records(policy))


def open_policy_dir(common: str, euid: int) -> int:
    """Open <common>/identity-policy without following a symlink; return the checked dir fd."""
    pdir = os.path.join(common, "identity-policy")
    try:
        check_node_type(pdir, os.lstat(pdir), "dir")
        dfd = os.open(pdir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except FileNotFoundError:
        raise GuardError(f"private identity policy is missing: {pdir}/policy.json\n  The operator "
                         "installs it in the common git dir (never in a worktree); refusing") from None
    except OSError as exc:
        raise GuardError(f"private identity policy dir {pdir} is unreadable ({exc.strerror}); "
                         "refusing") from None
    try:
        check_private_node(pdir, os.fstat(dfd), euid=euid, want="dir")
    except GuardError:
        os.close(dfd)
        raise
    return dfd


def read_policy_file(dfd: int, pfile: str, euid: int) -> bytes:
    """Read policy.json inside the opened dir; the owner/mode check is on the opened file."""
    try:
        check_node_type(pfile, os.stat("policy.json", dir_fd=dfd, follow_symlinks=False), "file")
        ffd = os.open("policy.json", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dfd)
    except FileNotFoundError:
        raise GuardError(f"private identity policy is missing: {pfile}; refusing") from None
    except OSError as exc:
        raise GuardError(f"private identity policy {pfile} is unreadable ({exc.strerror}); "
                         "refusing") from None
    try:
        check_private_node(pfile, os.fstat(ffd), euid=euid, want="file")
        with os.fdopen(ffd, "rb", closefd=False) as fh:
            raw = fh.read(POLICY_MAX_BYTES + 1)
    finally:
        os.close(ffd)
    if len(raw) > POLICY_MAX_BYTES:
        raise GuardError(f"private identity policy exceeds {POLICY_MAX_BYTES} bytes; refusing")
    return raw


def load_bundle(common: str, euid: int) -> Bundle:
    dfd = open_policy_dir(common, euid)
    try:
        raw = read_policy_file(dfd, os.path.join(common, "identity-policy", "policy.json"), euid)
    finally:
        os.close(dfd)
    return parse_policy(raw)


def load_policy(common: str, euid: int) -> tuple[list[str], list[str]]:
    """The deny-list view of the bundle: (full names, emails)."""
    bundle = load_bundle(common, euid)
    return bundle.names, bundle.emails


def check_records(bundle: Bundle, hexlen: int, clock: Clock) -> None:
    """Every pinned tip must be a local commit and every recorded tag a local tag object."""
    for oid in sorted(bundle.tips | set(bundle.tag_records)):
        if len(oid) != hexlen:
            raise GuardError(f"private identity policy names {oid}, which is not a full object id "
                             "of this repository's object format; refusing")
    types = object_types(set(bundle.tips) | set(bundle.tag_records), clock)
    for oid, want, what in ([(t, "commit", "snapshot tip") for t in sorted(bundle.tips)]
                            + [(t, "tag", "tag record") for t in sorted(bundle.tag_records)]):
        kind = types.get(oid, "missing")
        if kind != want:
            desc = "missing" if kind == "missing" else f"a {kind}"
            raise GuardError(f"private identity policy {what} {oid} is {desc}, not a "
                             f"{'commit' if want == 'commit' else 'tag object'} in this repository. "
                             "TARS fetches the canonical upstream, re-verifies the record and "
                             "updates the private bundle; refusing")


# ── matching ───────────────────────────────────────────────────────────────────────────────
class Matchers:
    def __init__(self, names: list[str], emails: list[str], block_file: str):
        self.names = [re.compile(r"(?<!\w)" + r"\s+".join(map(re.escape, n.split())) + r"(?!\w)",
                                 re.IGNORECASE) for n in names]
        self.emails = [re.compile(r"(?<![\w.%+-])" + re.escape(e) + r"(?![\w%+-]|\.\w)",
                                  re.IGNORECASE) for e in emails]
        try:
            with open(block_file, "rb") as fh:
                lines = [ln for ln in fh.read().decode("utf-8").splitlines() if ln]
            self.block = re.compile("|".join(f"(?:{ln})" for ln in lines))
        except (OSError, UnicodeDecodeError, re.error) as exc:
            raise GuardError(f"cannot load the public block pattern {block_file} ({exc}); "
                             "refusing") from None
        if not lines:
            raise GuardError("the public block pattern is empty; refusing")

    def header(self, role: str, name: str, email: str) -> list[str]:
        """An identity header identifies the operator by the email token, or by the full name
        whatever email goes with it (a name field may also carry the email)."""
        found = []
        if any(r.search(name) for r in self.names) or any(r.search(name) for r in self.emails):
            found.append(f"{role} name matches the private deny-list")
        if any(r.search(email) for r in self.emails):
            found.append(f"{role} email matches the private deny-list")
        for field, value in (("name", name), ("email", email)):
            if self.block.search(value):
                found.append(f"{role} {field} matches the public block pattern")
        return found

    def text(self, field: str, text: str) -> list[str]:
        """Free text (a message or an extra header's whole value): full names and emails."""
        found = []
        if any(r.search(text) for r in self.names) or any(r.search(text) for r in self.emails):
            found.append(f"{field} matches the private deny-list")
        if self.block.search(text):
            found.append(f"{field} matches the public block pattern")
        return found


# ── raw object audit ───────────────────────────────────────────────────────────────────────
def _ident(role: str, line: bytes | None, match: Matchers,
           malformed: str | None = None) -> tuple[list[str], bool | None]:
    """Audit one identity header; return (problems, whether it is byte-exact sim), the latter
    None when the header is unusable (already a problem)."""
    m = IDENT_RE.fullmatch(line or b"")
    if not m or m.group(1).decode() != role:
        return [malformed or f"malformed or missing {role} header"], None
    name, email = m.group(2), m.group(3)
    try:
        texts = (name.decode("utf-8"), email.decode("utf-8"))
    except UnicodeDecodeError:
        return [f"{role} header is not valid UTF-8"], None
    return match.header(role, *texts), name == SIM_NAME and email == SIM_EMAIL


def _headers(head: bytes, hexlen: int,
             match: Matchers) -> tuple[list[str], bytes | None, bytes | None, list[bytes]]:
    """Validate header structure; return (problems, author line, committer line, embedded
    tag objects of the mergetag headers)."""
    lines = head.split(b"\n")
    oid = rb"[0-9a-f]{%d}" % hexlen
    if not re.fullmatch(rb"tree " + oid, lines[0]):
        return ["malformed tree header"], None, None, []
    i = 1
    while i < len(lines) and lines[i].startswith(b"parent "):
        if not re.fullmatch(rb"parent " + oid, lines[i]):
            return ["malformed parent header"], None, None, []
        i += 1
    author = lines[i] if i < len(lines) and lines[i].startswith(b"author ") else None
    i += author is not None
    committer = lines[i] if i < len(lines) and lines[i].startswith(b"committer ") else None
    i += committer is not None
    problems, mergetags = _extra_headers(lines[i:], match)
    return problems, author, committer, mergetags


def _extra_headers(lines: list[bytes], match: Matchers) -> tuple[list[str], list[bytes]]:
    """Every header after committer (encoding, mergetag, gpgsig, ...), with its continuation
    lines, must be well formed and UTF-8, and its whole text is scanned like a message: a
    mergetag embeds a tagger and a tag message, a signature block is free text. Also return
    each mergetag's embedded tag object, byte-exact (git indents its lines by one space)."""
    problems, seen, blocks = [], set(), []
    for line in lines:
        if line.startswith(b" "):
            if not blocks:
                problems.append("malformed header: a continuation line without a header")
                blocks.append(("a header continuation line", []))
            blocks[-1][1].append(line[1:])
            continue
        key = line.split(b" ", 1)[0]
        if not re.fullmatch(rb"[a-z][a-z0-9-]*", key):
            label = "an unparseable header"  # never echo the token: it may be a private value
            problems.append("malformed header: an unparseable header line")
        else:
            # Only a known git header name is ever printed: any other key-shaped token may be a
            # word of a private value.
            label = f"'{key.decode()}' header" if key in KNOWN_HEADERS else "an unknown header"
            if key in seen or key in {b"tree", b"parent", b"author", b"committer"}:
                problems.append(f"malformed header: duplicate or misplaced {label}")
        seen.add(key)
        if key == b"encoding" and line[len(key) + 1:].lower() not in UTF8_NAMES:
            problems.append("declares a non-UTF-8 encoding header")
        blocks.append((label, [line]))
    mergetags = []
    for label, parts in blocks:
        whole = b"\n".join(parts)
        if label == "'mergetag' header":
            mergetags.append(whole[len(b"mergetag "):] + b"\n")
        try:
            text = whole.decode("utf-8")
        except UnicodeDecodeError:
            problems.append(f"{label} is not valid UTF-8")
            continue
        problems += match.text(label, text)
    return problems, mergetags


def _message(msg: bytes, field: str, match: Matchers) -> list[str]:
    if b"\x00" in msg:
        return [f"{field} contains a NUL byte"]
    try:
        return match.text(field, msg.decode("utf-8"))
    except UnicodeDecodeError:
        return [f"{field} is not valid UTF-8"]


def audit_commit(raw: bytes, hexlen: int, match: Matchers, *, upstream: bool = False,
                 tag_records: dict[str, str] | None = None, fmt: str = "sha1") -> list[str]:
    """Problems of one raw commit. A verified upstream commit (`upstream`) keeps its own
    identities; a fork commit must be sim and its mergetags obey the tag rules."""
    sep = raw.find(b"\n\n")
    if sep < 0:
        return ["malformed commit: no header/message separator"]
    head, msg = raw[:sep], raw[sep + 2:]
    if b"\x00" in head:
        return ["malformed commit: NUL byte in the headers"]
    problems, author, committer, mergetags = _headers(head, hexlen, match)
    for role, line in (("author", author), ("committer", committer)):
        found, is_sim = _ident(role, line, match)
        problems += found
        if is_sim is False and not upstream:
            problems.append(f"{role} {NOT_VERIFIED}")
    problems += _message(msg, "message", match)
    if not upstream:
        for embedded in mergetags:
            oid = hashlib.new(fmt, b"tag %d\x00" % len(embedded) + embedded).hexdigest()
            found, _target, _kind, _recorded = audit_tag(embedded, hexlen, match, tag_records or {},
                                                         oid)
            problems += [f"mergetag {p}" for p in found]
    return problems


def audit_tag(raw: bytes, hexlen: int, match: Matchers, tag_records: dict[str, str],
              oid: str) -> tuple[list[str], str | None, str | None, bool]:
    """Problems of one raw tag object `oid`; also (tagged oid, declared type, recorded?).
    The tagged oid is None when the object/type/tag headers are unusable."""
    sep = raw.find(b"\n\n")
    if sep < 0 or b"\x00" in raw[:sep]:
        return ["malformed tag: no header/message separator or a NUL byte in the headers"], \
            None, None, False
    head, msg = raw[:sep], raw[sep + 2:]
    lines = head.split(b"\n") + [b"", b""]
    obj = re.fullmatch(rb"object ([0-9a-f]{%d})" % hexlen, lines[0])
    kind = re.fullmatch(rb"type (commit|tree|blob|tag)", lines[1])
    name = re.fullmatch(rb"tag ([^\x00-\x20\x7f]+)", lines[2])
    if not (obj and kind and name):
        return ["malformed tag: bad object, type or tag header"], None, None, False
    lines = lines[:-2]
    tagger = lines[3] if len(lines) > 3 and lines[3].startswith(b"tagger ") else None
    problems, _ = _extra_headers(lines[3 + (tagger is not None):], match)
    found, is_sim = _ident("tagger", tagger, match, "malformed tag: missing or malformed tagger header")
    problems += found
    record = tag_records.get(oid)
    try:
        tag_name = name.group(1).decode("utf-8")
    except UnicodeDecodeError:
        problems.append("malformed tag: the tag name is not valid UTF-8")
        tag_name = None
    else:
        problems += match.text("tag name", tag_name)
    if record is not None and record != f"refs/tags/{tag_name}":
        problems.append("the tag record for this tag object names a different tag")
    if is_sim is False and record is None:
        problems.append(TAG_UNRECORDED)
    problems += _message(msg, "tag message", match)
    return problems, obj.group(1).decode(), kind.group(1).decode(), record is not None


def read_objects(oids: list[str], clock: Clock):
    """Yield (oid, type, bytes) for each oid via one `cat-file --batch`, under the deadline."""
    if not oids:
        return
    proc = subprocess.Popen(["git", "cat-file", "--batch"], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=GIT_ENV)
    killer = threading.Timer(clock.remaining(), proc.kill)
    killer.start()

    def feed():
        try:
            proc.stdin.write("".join(f"{o}\n" for o in oids).encode())
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass

    writer = threading.Thread(target=feed, daemon=True)  # health: allow HX012 -- standalone hook subprocess, no profile scope
    writer.start()
    try:
        for n, oid in enumerate(oids):
            if n % 1024 == 0:
                clock.remaining()
            header = proc.stdout.readline().split()
            if len(header) != 3 or header[0].decode() != oid:
                clock.remaining()
                raise GuardError(f"commit {oid} is missing or unreadable; refusing (fail closed)")
            size = int(header[2])
            body = proc.stdout.read(size + 1)
            if len(body) != size + 1:
                clock.remaining()
                raise GuardError(f"commit {oid} could not be read completely; refusing")
            yield oid, header[1].decode(), body[:-1]
    finally:
        killer.cancel()
        proc.kill()
        proc.wait()
        writer.join(timeout=5)


# ── refs and traversal ─────────────────────────────────────────────────────────────────────
def parse_input(data: bytes, hexlen: int) -> list[tuple[str, str, str, str]]:
    if len(data) > MAX_STDIN:
        raise GuardError("pre-push input is larger than expected; refusing")
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    oid_re = re.compile(f"[0-9a-f]{{{hexlen}}}")
    updates = []
    for n, raw in enumerate(lines, 1):
        try:
            parts = raw.decode("utf-8").split(" ")
        except UnicodeDecodeError:
            raise GuardError(f"malformed pre-push input (line {n}: not UTF-8); refusing") from None
        if len(parts) != 4 or not all(parts) or any(not p.isprintable() for p in parts):
            raise GuardError(f"malformed pre-push input (line {n}: expected 4 fields); refusing")
        local_ref, local_oid, remote_ref, remote_oid = parts
        if not oid_re.fullmatch(local_oid) or not oid_re.fullmatch(remote_oid):
            raise GuardError(f"malformed pre-push input (line {n}: object ids must be {hexlen} "
                             "lowercase hex digits); refusing")
        if not remote_ref.startswith("refs/"):
            raise GuardError(f"malformed pre-push input (line {n}: remote ref outside refs/); "
                             "refusing")
        updates.append((local_ref, local_oid, remote_ref, remote_oid))
    return updates


def object_types(oids: set[str], clock: Clock) -> dict[str, str]:
    if not oids:
        return {}
    res = git(["cat-file", "--batch-check"], clock, "".join(f"{o}\n" for o in sorted(oids)).encode())
    if res.returncode != 0:
        raise GuardError("cannot inspect the pushed objects; refusing (fail closed)")
    types = {}
    for line in res.stdout.decode(errors="replace").splitlines():
        parts = line.split()
        types[parts[0]] = parts[1] if len(parts) == 3 else "missing"
    return types


def peel(remote_ref: str, local_oid: str, kind: str, hexlen: int, match: Matchers,
         tag_records: dict[str, str], clock: Clock) -> tuple[str | None, list]:
    """Follow an annotated tag chain to its commit, auditing every tag object on the way.
    Return (peeled commit or None when the chain is malformed, [(tag oid, problems, class)])."""
    bad, target = [], local_oid
    for _depth in range(MAX_TAG_DEPTH + 1):
        if kind != "tag":
            break
        _oid, _kind, raw = next(read_objects([target], clock))
        problems, tagged, declared, recorded = audit_tag(raw, hexlen, match, tag_records, target)
        actual = None
        if tagged is not None:
            actual = object_types({tagged}, clock).get(tagged, "missing")
            if actual == "missing":
                raise GuardError(f"{remote_ref} -> {local_oid}: the tagged object {tagged} is missing "
                                 "locally; refusing")
            if actual != declared:
                problems.append("malformed tag: the type header does not match the tagged object")
        if problems:
            bad.append((target, problems, "upstream" if recorded else "fork"))
        if actual is None or actual != declared:
            return None, bad
        target, kind = tagged, actual
    else:
        raise GuardError(f"{remote_ref} -> {local_oid}: the tag chain is deeper than "
                         f"{MAX_TAG_DEPTH}; refusing")
    if kind != "commit":
        verb = "is" if target == local_oid else "peels to"
        raise GuardError(f"{remote_ref} -> {local_oid} {verb} a {kind}, not a commit; refusing")
    return target, bad


def resolve_refs(updates, zero: str, remote: str, hexlen: int, match: Matchers,
                 tag_records: dict[str, str], clock: Clock):
    """Return ([(remote ref, pushed commit, advertised old)], tag problems) for the updates."""
    wanted = {u[1] for u in updates if u[1] != zero} | {u[3] for u in updates if u[3] != zero}
    types = object_types(wanted, clock)
    targets, bad = [], []
    for _local_ref, local_oid, remote_ref, remote_oid in updates:
        if local_oid == zero:
            continue
        kind = types.get(local_oid, "missing")
        if kind == "missing":
            raise GuardError(f"{remote_ref} -> {local_oid} is missing locally; refusing")
        commit, tag_bad = peel(remote_ref, local_oid, kind, hexlen, match, tag_records, clock)
        bad += tag_bad
        if remote_oid != zero and types.get(remote_oid, "missing") not in ("commit", "tag"):
            raise GuardError(f"the remote's current {remote_ref} ({remote_oid}) is not a commit "
                             f"in this repository; fetch first (git fetch {remote}) and retry")
        if commit is not None:
            targets.append((remote_ref, commit, remote_oid))
    return targets, bad


def _rev_list(revs: list[str], cap: int, what: str, clock: Clock) -> list[str]:
    res = git(["rev-list", f"--max-count={cap + 1}", "--stdin"], clock,
              "".join(f"{r}\n" for r in revs).encode())
    if res.returncode != 0:
        first = res.stderr.decode(errors="replace").strip().splitlines()[:1]
        raise GuardError(f"the history of {what} cannot be traversed (missing or unreadable "
                         f"objects: {' '.join(first)}); refusing")
    return res.stdout.decode().split()


def enumerate_commits(targets, zero: str, cap: int, clock: Clock) -> list[str]:
    seen: dict[str, None] = {}
    for remote_ref, commit, remote_oid in targets:
        revs = [commit] + ([f"^{remote_oid}"] if remote_oid != zero else [])
        for oid in _rev_list(revs, cap, f"{remote_ref} -> {commit}", clock):
            seen[oid] = None
        if len(seen) > cap:
            raise GuardError(f"more than {cap} commits to audit exceeds the commit cap; refusing "
                             "(fail closed)")
    return list(seen)


def unverified_commits(targets, zero: str, tips: frozenset[str], cap: int,
                       clock: Clock) -> set[str] | None:
    """The pushed commits NOT reachable from any pinned snapshot tip (None: no snapshots, so
    every commit is unverified). Only exact pinned tips count: no ref is ever consulted."""
    if not tips:
        return None
    outside: set[str] = set()
    for remote_ref, commit, remote_oid in targets:
        revs = [commit] + ([f"^{remote_oid}"] if remote_oid != zero else [])
        revs += [f"^{t}" for t in sorted(tips)]
        outside.update(_rev_list(revs, cap, f"{remote_ref} -> {commit}", clock))
    return outside


def run(argv: list[str], stdin: bytes) -> int:
    if len(argv) < 2 or argv[0] != "--block-patterns":
        raise GuardError("usage: identity_guard.py --block-patterns <block.ere> <remote> [<url>]")
    block_file, remote = argv[1], (argv[2] if len(argv) > 2 else "origin")
    start = time.monotonic()
    cap, seconds = read_limits(Clock(TIME_LIMIT, start))
    clock = Clock(seconds, start)
    bundle = load_bundle(common_dir(clock), os.geteuid())
    match = Matchers(bundle.names, bundle.emails, block_file)
    fmt = git(["rev-parse", "--show-object-format"], clock).stdout.decode().strip()
    if fmt not in OBJECT_HEX:
        raise GuardError("unknown repository object format; refusing")
    hexlen = OBJECT_HEX[fmt]
    zero = "0" * hexlen
    check_records(bundle, hexlen, clock)
    updates = parse_input(stdin, hexlen)
    shallow = git(["rev-parse", "--is-shallow-repository"], clock)
    if shallow.returncode != 0 or shallow.stdout.strip() != b"false":
        raise GuardError("this is a shallow repository, so the full ancestry cannot be audited; "
                         "unshallow it (git fetch --unshallow) and retry")
    targets, bad = resolve_refs(updates, zero, remote, hexlen, match, bundle.tag_records, clock)
    t_enum = time.monotonic()
    commits = enumerate_commits(targets, zero, cap, clock)
    outside = unverified_commits(targets, zero, bundle.tips, cap, clock)
    t_audit = time.monotonic()
    upstream_count = 0
    for oid, kind, raw in read_objects(commits, clock):
        upstream = outside is not None and oid not in outside
        upstream_count += upstream
        problems = ([f"object is a {kind}"] if kind != "commit" else
                    audit_commit(raw, hexlen, match, upstream=upstream,
                                 tag_records=bundle.tag_records, fmt=fmt))
        if problems:
            bad.append((oid, problems, "upstream" if upstream else "fork"))
    clock.remaining()
    done = time.monotonic()
    print(f"{PREFIX} {len(commits)} commits enumerated in {t_audit - t_enum:.2f}s, audited in "
          f"{done - t_audit:.2f}s (total {done - start:.2f}s)", file=sys.stderr)
    if not bad:
        print(f"{PREFIX} passed ({len(commits)} commits: {len(commits) - upstream_count} sim, "
              f"{upstream_count} verified upstream)", file=sys.stderr)
        return 0
    report(bad, len(commits))
    return 1


def _needs_record(problems: list[str]) -> bool:
    """True when the only problems are a missing approval (no deny-list hit, nothing malformed)."""
    return all(p.endswith(NOT_VERIFIED) or p.endswith(TAG_UNRECORDED) for p in problems)


def report(bad, total: int) -> None:
    err = sys.stderr
    print(f"{PREFIX} REFUSED — {len(bad)} objects fail the identity/metadata audit "
          f"({total} commits audited):", file=err)
    for oid, problems, _cls in bad[:MAX_REPORTED]:
        print(f"  {oid}  {'; '.join(problems)}", file=err)
    if len(bad) > MAX_REPORTED:
        print(f"  ... and {len(bad) - MAX_REPORTED} more objects", file=err)
    approve = [oid for oid, problems, _cls in bad if _needs_record(problems)]
    fork = [oid for oid, problems, cls in bad if cls == "fork" and not _needs_record(problems)]
    upstream = [oid for oid, problems, cls in bad if cls == "upstream" and not _needs_record(problems)]
    print("  Matched values are not printed; inspect an object privately with\n"
          "    git cat-file -p <object>", file=err)
    if approve:
        print(f"  Not approved ({len(approve)}): {' '.join(approve[:MAX_REPORTED])}\n"
              "  Upstream history passes only through the private bundle: TARS independently fetches\n"
              f"  the canonical {CANONICAL_REPO} over an authenticated transport, verifies these\n"
              "  exact objects there, and records the verified tip (a snapshot record) or the exact\n"
              "  tag object (a tag record) in <git common dir>/identity-policy/policy.json. This\n"
              "  check never writes approval. Fork work is authored, committed and tagged as sim.",
              file=err)
    if fork:
        print("  Fork commits must be authored AND committed as sim <sim@hermes-agent.nousresearch.com>,\n"
              "  and no name, email or message may carry a private deny-list or host-path value:\n"
              "  rewrite the unpublished commits (identity and message), or re-create the\n"
              "  unpublished tags, before pushing again.", file=err)
    if upstream:
        print("  Verified upstream objects hit the deny-list or are malformed. Published upstream\n"
              "  history is not rewritten here and there is no bypass: stop for operator and TARS\n"
              "  investigation.", file=err)
    print("  HERMES_ALLOW_HOST_PATHS=1 does not bypass this gate.", file=err)


def main() -> int:
    try:
        return run(sys.argv[1:], sys.stdin.buffer.read(MAX_STDIN + 1))
    except GuardError as exc:
        print(f"{PREFIX} REFUSED — {exc}\n  (HERMES_ALLOW_HOST_PATHS=1 does not bypass this gate)",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
