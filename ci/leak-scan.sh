#!/usr/bin/env bash
# Refuse to let internal infrastructure reach the public mirror.
#
# This repository is push-mirrored to a public host on EVERY push, on EVERY
# branch -- a Forgejo push mirror cannot be limited to one ref. So a machine
# name, a home directory or an internal domain committed on a throwaway branch
# is published the moment it lands, and deleting the branch afterwards does not
# unpublish it. This runs in CI ahead of the suite, and from tools/ci/pre-push
# before anything leaves a clone.
#
#   ci/leak-scan.sh <dir>            scan a working tree
#   ci/leak-scan.sh --range A..B     scan content AND commit messages
#
# WHY THE PATTERNS ARE NOT IN THIS FILE. A list of every machine name on the
# fleet is itself the most concentrated piece of internal infrastructure there
# is -- committing the detector would publish exactly what it exists to keep
# private. (Written inline first; the scanner's first act was to flag its own
# source, which is the correct answer.) The patterns and their test vectors
# live outside the repository:
#
#   $NEREUS_LEAK_PATTERNS    path to the file, or
#   $LEAK_PATTERNS           the file's CONTENT, for CI (an Actions secret), or
#   ~/.dotfiles/nereus/leak-patterns.conf   the default
#
# WHY IT SELF-TESTS. An earlier scan of this repository reported a clean tree
# because it used `\b` inside `grep -E`, where POSIX ERE has no word-boundary
# escape, so the pattern matched nothing at all. A scanner that cannot find a
# string it is standing on returns a CLEAN verdict -- worse than no scanner. So
# the loaded patterns are run against vectors that must match and vectors that
# must not, and any disagreement aborts.
#
# IT FAILS CLOSED. No patterns, unreadable patterns, or a failed self-test all
# exit non-zero. "I could not check" must never read as "there is nothing here".
set -uo pipefail

die() { echo "leak-scan: $*" >&2; exit 2; }

# ---- load patterns from outside the repository ----------------------------
conf=""
if [ -n "${LEAK_PATTERNS:-}" ]; then
  conf=$(mktemp); printf '%s\n' "$LEAK_PATTERNS" > "$conf"; trap 'rm -f "$conf"' EXIT
elif [ -n "${NEREUS_LEAK_PATTERNS:-}" ]; then
  conf="$NEREUS_LEAK_PATTERNS"
elif [ -f "$HOME/.dotfiles/nereus/leak-patterns.conf" ]; then
  conf="$HOME/.dotfiles/nereus/leak-patterns.conf"
fi
[ -n "$conf" ] && [ -r "$conf" ] || die "no pattern file. Set NEREUS_LEAK_PATTERNS or LEAK_PATTERNS.
Refusing to report a tree clean without having checked it."

HOSTS=""; PATHS=""; ALLOW=""; POS=(); NEG=()
while IFS= read -r line; do
  case "$line" in
    ''|\#*)      continue ;;
    HOSTS=*)     HOSTS="${line#HOSTS=}" ;;
    PATHS=*)     PATHS="${line#PATHS=}" ;;
    ALLOW=*)     ALLOW="${line#ALLOW=}" ;;
    POSITIVE=*)  POS+=("${line#POSITIVE=}") ;;
    NEGATIVE=*)  NEG+=("${line#NEGATIVE=}") ;;
  esac
done < "$conf"
[ -n "$HOSTS" ] || die "pattern file defines no HOSTS"
[ -n "$PATHS" ] || die "pattern file defines no PATHS"
[ "${#POS[@]}" -ge 1 ] || die "pattern file carries no POSITIVE vectors; the self-test would be vacuous"
[ "${#NEG[@]}" -ge 1 ] || die "pattern file carries no NEGATIVE vectors; the self-test would be vacuous"

drop_allowed() { if [ -n "$ALLOW" ]; then grep -viE "$ALLOW" || true; else cat; fi; }

# Both patterns must see the SAME input. Piping into `grep A || grep B` does
# not: the first grep drains stdin and the second reads an empty stream, so
# half the patterns silently never run.
scan_text() {
  { printf '%s\n' "$1" | grep -inE  "$PATHS"
    printf '%s\n' "$1" | grep -inwE "$HOSTS"; } 2>/dev/null | drop_allowed
}
# Defined in terms of scan_text, not alongside it. Written as a separate pair of
# greps first, and the two drifted immediately: the self-test did not apply the
# allowlist the real scan did, so a NEGATIVE vector covering the allowlist
# failed. A self-test that does not exercise the production path tests nothing.
matches() { [ -n "$(scan_text "$1")" ]; }

fail=0
for s in "${POS[@]}"; do matches "$s" || { echo "SELF-TEST: no match on known-positive: $s" >&2; fail=1; }; done
for s in "${NEG[@]}"; do matches "$s" && { echo "SELF-TEST: fired on known-negative: $s" >&2; fail=1; }; done
[ "$fail" -eq 0 ] || die "patterns are unreliable; refusing to certify anything."
echo "leak-scan: self-test ok (${#POS[@]} positives matched, ${#NEG[@]} negatives ignored)"

# ---- scan -----------------------------------------------------------------
hits=0
if [ "${1:-}" = "--range" ]; then
  range="${2:?usage: ci/leak-scan.sh --range A..B}"
  echo "leak-scan: scanning commits in $range"
  for c in $(git rev-list "$range"); do
    m=$(scan_text "$(git log -1 --format='%s%n%b' "$c")")
    [ -n "$m" ] && { echo "COMMIT MESSAGE $(git log -1 --format='%h %s' "$c")"; printf '%s\n' "$m" | sed 's/^/    /'; hits=1; }
    t=$( { git grep -inE "$PATHS" "$c" -- 2>/dev/null; git grep -inwE "$HOSTS" "$c" -- 2>/dev/null; } | drop_allowed )
    [ -n "$t" ] && { echo "COMMIT CONTENT $(git log -1 --format='%h %s' "$c")"; printf '%s\n' "$t" | sed 's/^/    /'; hits=1; }
  done
else
  cd "${1:-.}" || die "no such directory: ${1:-.}"
  echo "leak-scan: scanning tracked files in $(pwd)"
  # Tracked files only: untracked scratch (run logs, figures) is never pushed.
  out=$( { git ls-files -z | xargs -0 grep -inE  "$PATHS" 2>/dev/null
           git ls-files -z | xargs -0 grep -inwE "$HOSTS" 2>/dev/null; } | drop_allowed )
  [ -n "$out" ] && { printf '%s\n' "$out" | sed 's/^/    /'; hits=1; }
fi

if [ "$hits" -ne 0 ]; then
  cat >&2 <<'MSG'

leak-scan: FAILED -- internal infrastructure would be published.

The mirror publishes every branch on every push. Move the value into an
Actions variable or a file outside the repository, or reword the comment to
name the architecture rather than the machine.
MSG
  exit 1
fi
echo "leak-scan: clean"
