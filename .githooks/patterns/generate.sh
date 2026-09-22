#!/usr/bin/env bash
# SOLE writer of patterns/block.ere and patterns/warn.ere (beside this script).
#
# The hook verifies both files against published md5s and REFUSES on any mismatch, so
# changing a pattern means: run this script, copy the md5 it prints into pre-push, and
# commit both together. That coupling is deliberate — a weakened pattern file is a
# silent bypass, and the patterns are tracked, so a commit inside a pushed range could
# otherwise edit them unnoticed.
#
# Design note: BLOCK carries only shapes that identify THIS host (its concrete mount
# root and home root). Generic spellings (/home/<name>, /Users/<name>, RFC1918 literals,
# ~/.hermes/profiles/<name>) are WARNed, never blocked: upstream's own docs and tests are
# full of them, so blocking them refuses ordinary pushes. The operator handle is WARNed
# too — it is this PUBLIC fork's own name and already in every clone's .git/config.
set -eu
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
for f in "$here/block.ere" "$here/warn.ere"; do
  printf "%s %sB md5 %s\n" "$(basename "$f")" "$(wc -c < "$f")" "$(md5sum < "$f" | cut -d" " -f1)"
done
