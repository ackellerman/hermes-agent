"""pre-push identity gate: audit the raw metadata of every commit a push would publish.

Run by `.githooks/pre-push` (same directory) BEFORE the host-path content gate, with the
pre-push stdin and `--block-patterns <verified block.ere>`. Exit 0 passes; anything else refuses.

WHY: the content gate reads diffs, not author/committer headers or commit messages, and it
skips history it believes a remote already has. A personal identity on the hidden parent of a
merge therefore went public through it. This gate enumerates, per pushed ref, every commit
reachable through ALL parents (`<new> ^<advertised old>`; a new ref audits its whole ancestry,
never excluding tracking refs), and checks each raw commit object:

  * author AND committer must be byte-exact `sim <sim@hermes-agent.nousresearch.com>`. This
    checkpoint has no verified-upstream snapshot or reviewed cherry-pick records, so it fails
    closed for every other identity;
  * headers must be well formed and UTF-8, the message UTF-8 without NUL bytes;
  * names, emails and the whole message are scanned with the public block pattern and the
    private deny-list. Headers refuse on the operator email token; messages refuse on the full
    name (case-insensitive, word-bounded phrase) or the email token, never on substrings.
    Every header after committer (mergetag, gpgsig, ...) must be UTF-8 and its whole text,
    continuation lines included, is scanned like the message. Matched values are never printed.

The private deny-list lives ONLY in `<git common dir>/identity-policy/policy.json` (never a
worktree or hook-directory copy): schema v1 is exactly
`{"version": 1, "names": ["<full name>", ...], "emails": ["<email>", ...]}`, both lists
non-empty. A missing, malformed, duplicate-keyed, symlinked, foreign-owned or group/world-
writable bundle refuses. Annotated tags and other non-commit objects refuse at this checkpoint.

Limits: 250,000 commits and 180 s for the whole gate, fail closed. `git -c
identityguard.maxCommits=N` / `identityguard.timeoutSeconds=S` can only LOWER them (tests use
this); a malformed value refuses. Threat model: ordinary-operation accidents; `--no-verify`
and a replaced hook bypass any local hook. `HERMES_ALLOW_HOST_PATHS=1` never reaches this gate.
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
import subprocess
import sys
import threading
import time

SIM_NAME = b"sim"
SIM_EMAIL = b"sim@hermes-agent.nousresearch.com"
MAX_COMMITS = 250_000
TIME_LIMIT = 180.0
POLICY_VERSION = 1
POLICY_MAX_BYTES = 64 * 1024
POLICY_MAX_ENTRIES = 32
MAX_STDIN = 16 * 1024 * 1024
MAX_REPORTED = 20
OBJECT_HEX = {"sha1": 40, "sha256": 64}
PREFIX = "pre-push identity:"

GIT_ENV = dict(os.environ, GIT_NO_LAZY_FETCH="1", GIT_NO_REPLACE_OBJECTS="1",
               GIT_TERMINAL_PROMPT="0", GIT_PAGER="cat", LC_ALL="C")
IDENT_RE = re.compile(rb"(author|committer) ([^<>\n]+) <([^<>\n]+)> ([0-9]+) ([+-][0-9]{4})")
EMAIL_SHAPE = re.compile(r"[^\s@<>]+@[^\s@<>]+\.[^\s@<>]+")
UTF8_NAMES = {b"utf-8", b"utf8"}


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


def parse_policy(raw: bytes) -> tuple[list[str], list[str]]:
    try:
        text = raw.decode("utf-8")
        policy = json.loads(text, object_pairs_hook=_no_duplicate_keys,
                            parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError):
        raise GuardError("private identity policy is not valid UTF-8 JSON; refusing") from None
    if not isinstance(policy, dict):
        raise GuardError("private identity policy must be a JSON object; refusing")
    expected = {"version", "names", "emails"}
    if set(policy) - expected:
        raise GuardError(f"private identity policy has unknown key(s) "
                         f"{sorted(set(policy) - expected)}; refusing (schema v{POLICY_VERSION})")
    if expected - set(policy):
        raise GuardError(f"private identity policy lacks key(s) {sorted(expected - set(policy))}; "
                         "refusing")
    version = policy["version"]
    if type(version) is not int or version != POLICY_VERSION:
        raise GuardError(f"private identity policy version is not {POLICY_VERSION}; this hook "
                         f"accepts exactly schema v{POLICY_VERSION}; refusing")
    names, emails = _entries(policy, "names"), _entries(policy, "emails")
    for i, name in enumerate(names):
        words = name.split()
        if " ".join(words) != name or len(words) < 2 or not all(any(c.isalpha() for c in w) for w in words):
            raise GuardError(f"private identity policy names[{i}] is not a full name; refusing")
    for i, email in enumerate(emails):
        if not EMAIL_SHAPE.fullmatch(email):
            raise GuardError(f"private identity policy emails[{i}] is not an email; refusing")
    return names, emails


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


def load_policy(common: str, euid: int) -> tuple[list[str], list[str]]:
    dfd = open_policy_dir(common, euid)
    try:
        raw = read_policy_file(dfd, os.path.join(common, "identity-policy", "policy.json"), euid)
    finally:
        os.close(dfd)
    return parse_policy(raw)


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

    def header(self, field: str, value: str) -> list[str]:
        found = []
        if any(r.search(value) for r in self.emails):
            found.append(f"{field} matches the private deny-list")
        if self.block.search(value):
            found.append(f"{field} matches the public block pattern")
        return found

    def text(self, field: str, text: str) -> list[str]:
        """Free text (a message or an extra header's whole value): full names and emails."""
        found = []
        if any(r.search(text) for r in self.names) or any(r.search(text) for r in self.emails):
            found.append(f"{field} matches the private deny-list")
        if self.block.search(text):
            found.append(f"{field} matches the public block pattern")
        return found

    def message(self, text: str) -> list[str]:
        return self.text("message", text)


# ── raw commit audit ───────────────────────────────────────────────────────────────────────
def _ident(role: str, line: bytes | None, match: Matchers) -> list[str]:
    m = IDENT_RE.fullmatch(line or b"")
    if not m or m.group(1).decode() != role:
        return [f"malformed or missing {role} header"]
    name, email = m.group(2), m.group(3)
    found = []
    try:
        texts = (name.decode("utf-8"), email.decode("utf-8"))
    except UnicodeDecodeError:
        return [f"{role} header is not valid UTF-8"]
    if name != SIM_NAME or email != SIM_EMAIL:
        found.append(f"{role} is not the fork identity sim (non-sim identities fail closed at "
                     "this checkpoint)")
    found += match.header(f"{role} name", texts[0]) + match.header(f"{role} email", texts[1])
    return found


def _headers(head: bytes, hexlen: int,
             match: Matchers) -> tuple[list[str], bytes | None, bytes | None]:
    """Validate header structure; return (problems, author line, committer line)."""
    lines = head.split(b"\n")
    oid = rb"[0-9a-f]{%d}" % hexlen
    if not re.fullmatch(rb"tree " + oid, lines[0]):
        return ["malformed tree header"], None, None
    i = 1
    while i < len(lines) and lines[i].startswith(b"parent "):
        if not re.fullmatch(rb"parent " + oid, lines[i]):
            return ["malformed parent header"], None, None
        i += 1
    author = lines[i] if i < len(lines) and lines[i].startswith(b"author ") else None
    i += author is not None
    committer = lines[i] if i < len(lines) and lines[i].startswith(b"committer ") else None
    i += committer is not None
    return _extra_headers(lines[i:], match), author, committer


def _extra_headers(lines: list[bytes], match: Matchers) -> list[str]:
    """Every header after committer (encoding, mergetag, gpgsig, ...), with its continuation
    lines, must be well formed and UTF-8, and its whole text is scanned like a message: a
    mergetag embeds a tagger and a tag message, a signature block is free text."""
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
            label = f"'{key.decode()}' header"
            if key in seen or key in {b"tree", b"parent", b"author", b"committer"}:
                problems.append(f"malformed header: duplicate or misplaced {label}")
        seen.add(key)
        if key == b"encoding" and line[len(key) + 1:].lower() not in UTF8_NAMES:
            problems.append("declares a non-UTF-8 encoding header")
        blocks.append((label, [line]))
    for label, parts in blocks:
        try:
            text = b"\n".join(parts).decode("utf-8")
        except UnicodeDecodeError:
            problems.append(f"{label} is not valid UTF-8")
            continue
        problems += match.text(label, text)
    return problems


def audit_commit(raw: bytes, hexlen: int, match: Matchers) -> list[str]:
    sep = raw.find(b"\n\n")
    if sep < 0:
        return ["malformed commit: no header/message separator"]
    head, msg = raw[:sep], raw[sep + 2:]
    if b"\x00" in head:
        return ["malformed commit: NUL byte in the headers"]
    problems, author, committer = _headers(head, hexlen, match)
    problems += _ident("author", author, match) + _ident("committer", committer, match)
    if b"\x00" in msg:
        problems.append("message contains a NUL byte")
    else:
        try:
            problems += match.message(msg.decode("utf-8"))
        except UnicodeDecodeError:
            problems.append("message is not valid UTF-8")
    return problems


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

    writer = threading.Thread(target=feed, daemon=True)
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


def check_ref_objects(updates, zero: str, remote: str, clock: Clock) -> None:
    wanted = {u[1] for u in updates if u[1] != zero} | {u[3] for u in updates if u[3] != zero}
    types = object_types(wanted, clock)
    for _local_ref, local_oid, remote_ref, remote_oid in updates:
        if local_oid == zero:
            continue
        kind = types.get(local_oid, "missing")
        if kind == "tag":
            raise GuardError(f"{remote_ref} -> {local_oid} is an annotated tag; annotated tags "
                             "are refused at this checkpoint (push a lightweight tag)")
        if kind == "missing":
            raise GuardError(f"{remote_ref} -> {local_oid} is missing locally; refusing")
        if kind != "commit":
            raise GuardError(f"{remote_ref} -> {local_oid} is a {kind}, not a commit; refusing")
        if remote_oid != zero and types.get(remote_oid, "missing") != "commit":
            raise GuardError(f"the remote's current {remote_ref} ({remote_oid}) is not a commit "
                             f"in this repository; fetch first (git fetch {remote}) and retry")


def enumerate_commits(updates, zero: str, cap: int, clock: Clock) -> list[str]:
    seen: dict[str, None] = {}
    for _local_ref, local_oid, remote_ref, remote_oid in updates:
        if local_oid == zero:
            continue
        args = ["rev-list", f"--max-count={cap + 1}", local_oid]
        if remote_oid != zero:
            args.append(f"^{remote_oid}")
        res = git([*args, "--"], clock)
        if res.returncode != 0:
            first = res.stderr.decode(errors="replace").strip().splitlines()[:1]
            raise GuardError(f"the history of {remote_ref} -> {local_oid} cannot be traversed "
                             f"(missing or unreadable objects: {' '.join(first)}); refusing")
        for oid in res.stdout.decode().split():
            seen[oid] = None
        if len(seen) > cap:
            raise GuardError(f"more than {cap} commits to audit exceeds the commit cap; refusing "
                             "(fail closed)")
    return list(seen)


def run(argv: list[str], stdin: bytes) -> int:
    if len(argv) < 2 or argv[0] != "--block-patterns":
        raise GuardError("usage: identity_guard.py --block-patterns <block.ere> <remote> [<url>]")
    block_file, remote = argv[1], (argv[2] if len(argv) > 2 else "origin")
    start = time.monotonic()
    cap, seconds = read_limits(Clock(TIME_LIMIT, start))
    clock = Clock(seconds, start)
    names, emails = load_policy(common_dir(clock), os.geteuid())
    match = Matchers(names, emails, block_file)
    fmt = git(["rev-parse", "--show-object-format"], clock).stdout.decode().strip()
    if fmt not in OBJECT_HEX:
        raise GuardError("unknown repository object format; refusing")
    hexlen = OBJECT_HEX[fmt]
    zero = "0" * hexlen
    updates = parse_input(stdin, hexlen)
    shallow = git(["rev-parse", "--is-shallow-repository"], clock)
    if shallow.returncode != 0 or shallow.stdout.strip() != b"false":
        raise GuardError("this is a shallow repository, so the full ancestry cannot be audited; "
                         "unshallow it (git fetch --unshallow) and retry")
    check_ref_objects(updates, zero, remote, clock)
    t_enum = time.monotonic()
    commits = enumerate_commits(updates, zero, cap, clock)
    t_audit = time.monotonic()
    bad: list[tuple[str, list[str]]] = []
    for oid, kind, raw in read_objects(commits, clock):
        problems = [f"object is a {kind}"] if kind != "commit" else audit_commit(raw, hexlen, match)
        if problems:
            bad.append((oid, problems))
    clock.remaining()
    done = time.monotonic()
    print(f"{PREFIX} {len(commits)} commits enumerated in {t_audit - t_enum:.2f}s, audited in "
          f"{done - t_audit:.2f}s (total {done - start:.2f}s)", file=sys.stderr)
    if not bad:
        print(f"{PREFIX} passed ({len(commits)} commits, all sim)", file=sys.stderr)
        return 0
    report(bad, len(commits))
    return 1


def report(bad, total: int) -> None:
    err = sys.stderr
    print(f"{PREFIX} REFUSED — {len(bad)} of {total} commits fail the identity/metadata audit:",
          file=err)
    for oid, problems in bad[:MAX_REPORTED]:
        print(f"  {oid}  {'; '.join(problems)}", file=err)
    if len(bad) > MAX_REPORTED:
        print(f"  ... and {len(bad) - MAX_REPORTED} more commits", file=err)
    print("  Fork commits must be authored AND committed as sim <sim@hermes-agent.nousresearch.com>,\n"
          "  and no name, email or message may carry a private deny-list or host-path value.\n"
          "  Matched values are not printed; inspect a commit privately with\n"
          "    git log -1 --format='%an <%ae> | %cn <%ce>%n%B' <commit>\n"
          "  then rewrite the unpublished commits (identity and message) before pushing again.\n"
          "  HERMES_ALLOW_HOST_PATHS=1 does not bypass this gate.", file=err)


def main() -> int:
    try:
        return run(sys.argv[1:], sys.stdin.buffer.read(MAX_STDIN + 1))
    except GuardError as exc:
        print(f"{PREFIX} REFUSED — {exc}\n  (HERMES_ALLOW_HOST_PATHS=1 does not bypass this gate)",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
