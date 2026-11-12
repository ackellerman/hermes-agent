# Pre-push identity gate: private policy bundle (operator guide)

`pre-push` runs two gates on every push, in this order:

1. **Identity** (`identity_guard.py`). It audits the raw author, committer and message of every
   commit each pushed ref would publish, through all parents, and every annotated tag object.
   It has no override.
2. **Host-path content.** This is the older diff scan. `HERMES_ALLOW_HOST_PATHS=1` skips this
   gate only.

This file covers the private bundle that the identity gate reads: its schema, how TARS
bootstraps and reviews it, the limits, and how to recover from a refusal. It never contains a
private value, and nothing here authorizes a deployment, a push or a history rewrite.

## Status: not deployed

This hook is shipped and tested, but it is **not deployed**. Until a separately authorized
deployment, the copy that runs in the fork's `.git/hooks/` is the older content-only gate, which
does not check identities. The e1 and e2 checkpoints of this work are never deployed, because e1
refuses every non-sim identity and would block every push of real upstream history. Only the
complete hook is deployable: `pre-push`, `identity_guard.py` and `patterns/` from one commit,
together with a bootstrapped bundle. See [Deployment](#deployment).

## What the gate accepts

Each commit that a pushed ref would publish belongs to exactly one class:

| Class | Author | Committer | How it is approved |
|---|---|---|---|
| Fork work, including merges | `sim <sim@hermes-agent.nousresearch.com>`, byte for byte | the same | nothing more is needed |
| Verified upstream | any | any | reachable from an exact snapshot tip in the bundle |
| Reviewed cherry-pick | equal to the recorded upstream commit's author, byte for byte (name and email) | sim | an exact pick record |

Any other identity refuses. Every class, verified upstream included, is also scanned:

- The name and email fields of every identity header are matched against the private deny-list
  and against the public block pattern (`patterns/block.ere`).
- The whole message, and every header after the committer (`mergetag`, `gpgsig`, ...), is
  scanned for the full name (case-insensitive, word-bounded) and the email (as a complete token).

A deny-list hit always refuses, and the diagnostic never prints the matched value. Headers and
messages must be well-formed UTF-8 with no NUL bytes.

Annotated tags are peeled recursively, to a depth of at most 16. Every tag object in the chain
must be well formed, its declared type must equal the real type of the object it tags, and its
tagger must be sim unless an exact tag record names that tag object. The chain must end at a
commit, and that commit's ancestry is audited like a branch. A `mergetag` inside a fork commit
obeys the same tag rules.

## Where the bundle lives

The bundle is read only from `<git common dir>/identity-policy/policy.json`, where the common
dir is the output of `git rev-parse --path-format=absolute --git-common-dir`. That path is the
same from every linked worktree and is independent of where the hook was copied. A copy inside a
worktree or beside the hook is ignored, and the push refuses as "missing".

The gate opens the directory and the file without following symlinks, then checks what it
actually opened:

- `identity-policy/` must be a directory owned by the pushing user.
- `policy.json` must be a regular file owned by the pushing user.
- Any symlink, any other owner, or any group- or world-writable bit refuses. Group or world
  read/execute bits and a setgid bit do not (a setgid `2700` directory passes). Bootstrap with
  `0700` for the directory and `0600` for the file (below), which is what TARS should leave.

The parent common dir itself is not checked. The fork's `.git` is group-writable (setgid
`2775`), and anyone who replaced `identity-policy/` there would own the replacement and fail
the owner check anyway.

## Schema v3 (exact)

The bundle is one JSON object. It must have exactly these keys. A duplicate JSON key, an unknown
key, a missing key, a NaN or Infinity constant, or any other `version` refuses. A v1 or v2
bundle refuses under this hook.

```json
{
  "version": 3,
  "names": ["<operator full name>"],
  "emails": ["<operator email>"],
  "snapshots": [
    {"tips": ["<full commit oid>"],
     "repository": "NousResearch/hermes-agent",
     "source": "git@github.com:NousResearch/hermes-agent.git",
     "ref": "refs/heads/main",
     "verified_at": "2026-01-31T12:00:00Z",
     "note": "<TARS's verification note>"}
  ],
  "tag_records": [
    {"tag": "<full tag-object oid>",
     "repository": "NousResearch/hermes-agent",
     "source": "https://github.com/NousResearch/hermes-agent.git",
     "ref": "refs/tags/<name>",
     "verified_at": "2026-01-31T12:00:00Z",
     "note": "<TARS's review note>"}
  ],
  "pick_records": [
    {"candidate": "<full oid of the fork's cherry-pick commit>",
     "upstream_commit": "<full oid of the upstream commit it was picked from>",
     "repository": "NousResearch/hermes-agent",
     "source": "https://github.com/NousResearch/hermes-agent.git",
     "ref": "refs/pull/<n>/head",
     "verified_at": "2026-01-31T12:00:00Z",
     "note": "<what TARS compared: diffs and message; clean or which change>"}
  ]
}
```

Field rules:

- `names` and `emails` each hold 1 to 32 distinct entries. A name must be a full name of at
  least two words, and an email must be shaped like an email. A record list may be empty; the
  deny-list may not. The gate cannot tell which values are real: keeping the operator's real
  full name and real email in the lists is part of TARS's review.
- `repository` must be exactly `NousResearch/hermes-agent`.
- `source` must be an authenticated URL of that repository on github.com:
  `https://github.com/...`, `ssh://git@github.com/...` or `git@github.com:...`, with an optional
  `.git`. A local path, a remote name such as `upstream`, `file://`, `http://`, `git://`, another
  host, or another owner's fork all refuse.
- `ref`:
  - snapshots use `refs/heads/...` or `refs/tags/...`;
  - tag records use `refs/tags/<the tag's own name>`;
  - pick records use `refs/heads/...`, `refs/tags/...` or `refs/pull/<n>/head`.

  A tracking ref (`refs/remotes/...`) is never evidence.
- `verified_at` is a UTC time in the form `YYYY-MM-DDTHH:MM:SSZ`.
- `note` is non-empty printable text of at most 2000 characters.
- Every object id is a full, lowercase id in the repository's own object format (40 hex digits
  for SHA-1, 64 for SHA-256). Every pinned tip, pick candidate and picked upstream commit must
  be a commit present in this clone, and every recorded tag must be a tag object present in this
  clone. Otherwise the push refuses.

Duplicate and conflicting records refuse:

- **Snapshots:** a tip may appear only once across all snapshots. One tip cannot be pinned
  under two refs; keep one record per tip. One fetch (the same `ref` and `verified_at`)
  recorded with different tips also refuses.
- **Tag records:** a tag object may be recorded only once, and one tag ref may name only one
  tag object.
- **Pick records:** a candidate may be recorded only once, and the candidate and the upstream
  commit must differ. The same upstream commit may be picked into several candidates, one
  record each.

There are no wildcards and no identity allowlists. An `author`, `allowed_authors` or any other
extra field or key refuses.

## Bootstrap (TARS)

Use the fork clone that pushes. Mind the modes: on this fleet the natural `umask 002` plus a
setgid parent gives a `2775` directory and a `664` file, and the gate refuses both.

```sh
common=$(git rev-parse --path-format=absolute --git-common-dir)
mkdir -m 700 "$common/identity-policy"
# write policy.json there with an editor that does not leave backups beside it
chmod 700 "$common/identity-policy"           # keeps setgid on a directory (2700); that passes
chmod 600 "$common/identity-policy/policy.json"
```

If the gate prints `group- or world-writable (mode 2775)` or `(mode 664)`, run the two `chmod`
lines again. Every pinned tip and every recorded object must be fetched into this clone before
the next push, because a tip or record that is absent locally refuses every push from the
clone. Unshallow the clone first: a shallow clone refuses.

## Reviewing records (TARS)

The gate never writes approval. It prints object ids only, and TARS edits the bundle after
independent verification. Simply naming a mutable local `upstream` ref is not approval.

- **Snapshot.** Fetch the canonical `NousResearch/hermes-agent` over an authenticated
  transport (the `source` URL) from the named `ref`, without going through a local remote's
  configuration. Confirm that the fetched ref's commit equals the oid being pinned, then record
  it. Only commits reachable from an exact pinned tip are trusted.
- **Tag record.** Fetch the tag from the canonical source and confirm the tag object id. Read
  the tagger and message with `git cat-file -p <tag>`, then record the tag object id under its
  own `refs/tags/<name>`. A tag object refused for its mergetag is the tag embedded in that
  commit's `mergetag` header, not the commit.
- **Pick record.** Fetch the upstream commit from canonical evidence: a branch or tag that
  contains it, or the pull request head `refs/pull/<n>/head`. Then compare the exact candidate
  with the exact upstream commit, both the diff (`git diff <upstream>^ <upstream>` against
  `git diff <candidate>^ <candidate>`) and the message. Write in `note` what was compared and
  whether the pick is clean or modified. A modified or conflict-resolved pick needs the same
  explicit review; it is never equivalent automatically.

  The gate then still checks that the candidate is committed by sim, and that its raw author
  name and email equal the upstream commit's.

  A `(cherry picked from commit ...)` trailer is a claim. The refusal prints the oid the claim
  names, as a lead for the review, but the trailer never approves anything. The same goes for
  a matching author or patch-id.

A changed object is a new object id, so amending, re-dating or rebasing a candidate, or picking
a changed upstream commit, leaves the new commit without a record, and it refuses until TARS
reviews it. A record whose upstream commit has another author refuses as an uncertain record.

## Correction and revocation

- **Revocation.** Delete the snapshot, tag record or pick record. Approval ends on the next
  push; there is no cache.
- **Correction.** A demonstrably wrong deny-list entry may be corrected by TARS with a private,
  dated review note kept outside this repository. Neither a correction nor any configuration may
  remove the operator's real full name or real email.
- **Rotation.** To replace a record, delete it and add the new one in the same edit; a stale
  and a new record for the same candidate or tip refuse as duplicates. Re-run a push (or the
  hook with empty input, below) to check the bundle.

To check the bundle without pushing, run the hook with empty input; it still validates the
helper, the patterns and the whole bundle:

```sh
.git/hooks/pre-push origin </dev/null
```

## Limits

The gate fails closed when any limit is hit:

| Limit | Value |
|---|---|
| Commits audited per push (deduplicated over all refs) | 250,000 |
| Wall time for the whole identity gate | 180 s (outer kill at 240 s) |
| Bundle size | 256 KiB |
| Snapshot, tag and pick records | 256 per list |
| Tips per snapshot | 32 |
| Deny-list names and emails | 32 each |
| Annotated tag chain depth | 16 |
| pre-push input | 16 MiB |

`git -c identityguard.maxCommits=N` and `identityguard.timeoutSeconds=S` can only lower the
first two limits; a malformed value refuses.

Measured cost on the fleet host: a new ref holding the full fork base ancestry (64,236 commits)
takes about 9 s in total. Enumerating the commits takes about 0.6 s; the rest is reading and
checking the raw objects.

## Trust assumptions

The bundle is trusted operating state, not a cryptographic proof of who edited it. The gate
trusts that only TARS edits it, after the reviews above, and that the pushing user's account is
not compromised. The owner and mode checks catch accidental sharing; they do not authenticate
an editor. Records approve exact object ids only, so a record can never approve a commit it did
not name.

## Recovery: no bypass

There is no override for this gate. `--no-verify` or a replaced hook skips every local hook;
that is outside the threat model (ordinary-operation accidents) and is not a recovery path.

| Refusal | Recovery |
|---|---|
| Fork commit with a non-sim identity or a deny-list hit | Rewrite the unpublished commits as sim with a clean message, or re-create the unpublished tags, then push again. |
| Upstream commit not approved | TARS verifies it against the canonical source and pins a tip that contains it. |
| Upstream tag not approved | TARS verifies the tag and records it. |
| Cherry-pick not approved | TARS reviews the candidate against its upstream commit and adds a pick record. |
| Verified upstream object hits the deny-list or is malformed | Stop. TARS and the operator investigate. Published upstream history is not rewritten here. |
| Bundle missing, malformed, wrong mode or owner, or a record absent locally | Fix the bundle (see Bootstrap), or fetch the object. |
| Shallow clone, missing objects, or a limit hit | Unshallow or fetch, or push a smaller range. |

### Known refusals in today's fork history

These need a decision before the hook is deployed. None of them is authorized here.

- **Five sim commits in the base ancestry have messages that match the public block pattern:**
  `752c3e7d1414`, `7f94f98d574d`, `ef1654a105f7`, `93e375b7dda8`, `c1d4c3f1a83e`. Messages are
  scanned in every class, so any push that creates a new ref whose ancestry contains them
  refuses. An update whose advertised old OID already contains them is unaffected.
- **89 fork-only commits are not sim.** The fork base has 380 commits beyond the merge-base
  with upstream `main` (`21ae0e1238fd`), and 89 of them have a non-sim author or committer.
  - 14 are authored by others and committed by sim. Each can pass with a pick record once TARS
    has reviewed it against its upstream commit.
  - The other 75 have a non-sim committer, so a pick record cannot help them. Each needs either
    a snapshot of a canonical upstream ref that really contains it, or a history rewrite.
    A rewrite is a separate, coordinated operation that only TARS can authorize, with the
    operator's approval; this hook never asks for one.

  To list them:

  ```sh
  git log --format='%H %an <%ae> | %cn <%ce>' 21ae0e1238fd..<base> \
    | grep -v '^[0-9a-f]* sim <sim@hermes-agent.nousresearch.com> | sim <sim@hermes-agent.nousresearch.com>$'
  ```

## Concurrent pushes

The gate audits what git advertised. If another writer moves the destination ref after that
advertisement, the receiver still rejects the update, because git sends the advertised old
OID with it. `remote rejected (failed to update ref)`, `cannot lock ref ... is at X but
expected Y`, and nothing moves, even with `--force`. A test exercises this with a real race.
The gate cannot make a remote transaction atomic, and it cannot stop a separate push from
another clone or another configured client.

## Deployment

Landing the topic branch and deploying the hook is one later step. It needs TARS, with Adam's
independent authorization. In that step TARS:

1. bootstraps and reviews the bundle in the deploying clone;
2. resolves the known refusals above;
3. copies `pre-push`, `identity_guard.py` and `patterns/` from one commit together, keeping
   the copies equal to the tracked files;
4. records the deployment in the wrapper's operations log.

`pre-push` pins the helper's SHA-256 and the pattern files' md5s, so a partial copy refuses
every push. Until then the deployed hook stays as it is.
