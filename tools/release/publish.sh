#!/usr/bin/env bash
# Phase 2 of a release. Everything here is IRREVERSIBLE.
#
#   tools/release/publish.sh <nereus-version> <astronereus-version> --confirm
#
# Order is load-bearing:
#   1. tag on Forgejo            (never on GitHub: a GitHub-only tag is pruned
#                                 by the next mirror sync, which demotes the
#                                 release to a draft and 404s every bundle URL)
#   2. wait for the mirror
#   3. create the release EMPTY, then upload assets ONE AT A TIME
#      (`gh release create` with assets deletes the whole release if any upload
#       fails; v0.4.2 hit a transient 404 and the release vanished)
#   4. verify anonymously -- unauthenticated, which is how a stranger fetches
#   5. only then PyPI, because a wheel hard-codes those URLs and a published
#      filename can never be reused
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
. "$here/lib.sh"

JL_VER="${1:?usage: publish.sh <nereus-version> <astronereus-version> --confirm}"
PY_VER="${2:?}"; CONFIRM="${3:-}"
[ "$CONFIRM" = "--confirm" ] || die "refusing to publish without --confirm.
Everything after this point is irreversible: a tag, a public release, a PyPI
filename that can never be reused."
load_config

JL_ROOT=$(cd "$here/../.." && pwd)
PY_ROOT="$CFG_NEREUS_PY"
OUT="$CFG_STAGE/v$JL_VER"
[ -f "$OUT/manifest.json" ] || die "nothing staged at $OUT -- run prepare.sh first"

say "publishing Nereus $JL_VER / astronereus $PY_VER"

# The staged bundles must match the commit about to be tagged. Compared by
# TREE, not commit: a build checkout made with a fresh `git init` hashes the
# same source to an unrelated commit id.
staged_tree=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["tree"])' "$OUT/manifest.json")
now_tree=$(tree_of "$JL_ROOT")
[ "$staged_tree" = "$now_tree" ] || die "staged bundles were built from tree $staged_tree
but HEAD is tree $now_tree. Re-run prepare.sh; do not publish bundles that do
not correspond to the source being tagged."
info "tree matches: $staged_tree"

say "1/5  tag on Forgejo"
if git -C "$JL_ROOT" rev-parse "v$JL_VER" >/dev/null 2>&1; then
  info "tag v$JL_VER already exists locally"
else
  git -C "$JL_ROOT" tag -a "v$JL_VER" -m "Nereus $JL_VER"
fi
git -C "$JL_ROOT" push origin "v$JL_VER" || die "could not push the tag"

say "2/5  waiting for the mirror to carry the tag to GitHub"
for i in $(seq 1 60); do
  if git ls-remote --tags "https://github.com/$CFG_GH_REPO" "refs/tags/v$JL_VER" 2>/dev/null | grep -q .; then
    info "tag visible on GitHub after ~$((i*10))s"; break
  fi
  [ "$i" -eq 60 ] && die "tag never appeared on GitHub. Do NOT create it there by
hand -- the next mirror sync prunes it and the release becomes a draft."
  sleep 10
done

say "3/5  release, created empty then filled one asset at a time"
if gh release view "v$JL_VER" --repo "$CFG_GH_REPO" >/dev/null 2>&1; then
  info "release v$JL_VER already exists; uploading into it"
else
  notes="$OUT/release-notes.md"
  [ -f "$notes" ] || printf 'Nereus %s\n\nRuntime bundles for astronereus %s.\n' "$JL_VER" "$PY_VER" > "$notes"
  # --notes-from-tag is rejected together with --repo; use a file.
  gh release create "v$JL_VER" --repo "$CFG_GH_REPO" --verify-tag \
     --title "Nereus $JL_VER" --notes-file "$notes" || die "could not create the release"
fi
for f in "$OUT"/nereus-runtime-*.tar.zst "$OUT/SHA256SUMS"; do
  n=$(basename "$f")
  for attempt in 1 2 3; do
    if gh release upload "v$JL_VER" "$f" --repo "$CFG_GH_REPO" --clobber; then
      info "uploaded $n"; break
    fi
    # A pipeline's status is the LAST command's, so gh is checked directly.
    warn "upload of $n failed (attempt $attempt/3)"
    [ "$attempt" -eq 3 ] && die "could not upload $n; the release is intact, retry"
    sleep 15
  done
done

say "4/5  anonymous verification -- the check that actually matters"
base="https://github.com/$CFG_GH_REPO/releases/download/v$JL_VER"
for n in $(cd "$OUT" && ls nereus-runtime-*.tar.zst) SHA256SUMS; do
  code=$(env -u GITHUB_TOKEN -u GH_TOKEN curl -sIL -o /dev/null -w '%{http_code}' "$base/$n")
  [ "$code" = "200" ] || die "$n is not anonymously downloadable (HTTP $code).
Every published wheel resolves these URLs; do not upload to PyPI."
  info "200  $n"
done

say "5/5  PyPI"
: "${TWINE_PASSWORD:?TWINE_PASSWORD is not set -- run under sops exec-env}"
docker run --rm --name "nereus-release-upload-$$" \
  --label cl.jvines.owner=nereus-release \
  -e TWINE_USERNAME=__token__ -e TWINE_PASSWORD \
  -v "$OUT/dist":/dist:ro python:3.12-slim \
  bash -c 'pip install -q twine && twine upload /dist/*' \
  || die "twine upload failed. The GitHub release is live and correct; fix the
upload and retry -- do NOT rebuild, the filenames are already decided."

say "tagging nereus-py"
git -C "$PY_ROOT" tag -a "v$PY_VER" -m "astronereus $PY_VER" 2>/dev/null || true
git -C "$PY_ROOT" push origin "v$PY_VER" || warn "could not push the astronereus tag"

say "PUBLISHED  Nereus $JL_VER / astronereus $PY_VER"
info "verify a clean install:  pip install astronereus==$PY_VER"
