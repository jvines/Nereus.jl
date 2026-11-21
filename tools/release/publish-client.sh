#!/usr/bin/env bash
# A release of astronereus alone, against runtime bundles already published.
#
#   tools/release/publish-client.sh <astronereus-version> --check
#   tools/release/publish-client.sh <astronereus-version> --confirm
#
# A client fix (the daemon, the Python API) changes nothing in Nereus, and the
# wheel only POINTS at runtime bundles: the ones RUNTIME_VERSION names are on a
# public release already. So this builds nothing but the wheel and publishes
# nothing but the wheel and its tag -- no Nereus version, no bundles, no suite.
#
# --check runs every gate, the unit tests and the build, then stops. --confirm
# does the same and then uploads and tags; a PyPI filename can never be reused,
# so nothing irreversible runs until everything else has passed.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
. "$here/lib.sh"

PY_VER="${1:?usage: publish-client.sh <astronereus-version> --check|--confirm}"
MODE="${2:-}"
case "$MODE" in
  --check|--confirm) ;;
  *) die "say --check (stop before anything is published) or --confirm (publish).
A PyPI filename can never be reused." ;;
esac

load_config
OUT="$CFG_STAGE/astronereus-$PY_VER"
# A fresh clone of main (see fresh_clone), beside $OUT rather than in it:
# $OUT is emptied for the dists below, and the clone is tagged at the end.
PY_ROOT="$OUT.src"
say "astronereus $PY_VER against published runtime bundles ($MODE)"

# ---- gates ----------------------------------------------------------------
say "fresh clone of nereus-py main"
fresh_clone "$(py_url)" "$PY_ROOT"

[ "$(py_version "$PY_ROOT")" = "$PY_VER" ] \
  || die "pyproject.toml says $(py_version "$PY_ROOT"), you asked for $PY_VER. Bump it first."
[ "$(py_initver "$PY_ROOT")" = "$PY_VER" ] \
  || die "__init__.py says $(py_initver "$PY_ROOT") but pyproject says $PY_VER -- they must agree."
if git -C "$PY_ROOT" ls-remote --exit-code --tags origin "refs/tags/v$PY_VER" >/dev/null 2>&1; then
  die "tag v$PY_VER already exists on nereus-py."
fi
code=$(curl -s -o /dev/null -w '%{http_code}' "https://pypi.org/pypi/astronereus/$PY_VER/json")
[ "$code" = 404 ] || die "astronereus $PY_VER is already on PyPI (HTTP $code), and a filename can never be reused."
info "v$PY_VER is free on nereus-py and on PyPI"
"$PY_ROOT/ci/leak-scan.sh" "$PY_ROOT" || die "nereus-py would publish internal infrastructure"

# ---- the runtime it points at ---------------------------------------------
# Read from the public release, anonymously, which is how a user's install
# resolves it: every bundle URL answers, every checksum matches the release's
# SHA256SUMS, and that release's Nereus speaks this client's contract.
RT=$(py_runtime "$PY_ROOT")
say "runtime: Nereus $RT"
# In the same image as the tests: _runtime.py needs Python >= 3.10, which the
# host's python3 need not be.
docker run -i --rm --name "nereus-client-pointer-$$" \
  --label cl.jvines.owner=nereus-release \
  -v "$PY_ROOT/src/astronereus/_runtime.py":/_runtime.py:ro python:3.12-slim \
  python - /_runtime.py "$CFG_GH_REPO" "$(py_api "$PY_ROOT")" <<'PY' \
  || die "the runtime pointer does not resolve to a published, matching release"
import re, runpy, sys, urllib.request
m = runpy.run_path(sys.argv[1])
repo, api = sys.argv[2], sys.argv[3]
rt, rel = m["RUNTIME_VERSION"], m["_REL"]
want = f"https://github.com/{repo}/releases/download/v{rt}"
if rel != want:
    sys.exit(f"_REL is {rel}, but RUNTIME_VERSION {rt} lives at {want}")
fetch = lambda url: urllib.request.urlopen(url, timeout=60).read().decode()
sums = {}
for line in fetch(f"{rel}/SHA256SUMS").splitlines():
    h, name = line.split()
    sums[name.lstrip("*")] = h
for plat, (url, sha) in sorted(m["BUNDLES"].items()):
    name = url.rsplit("/", 1)[1]
    if not url.startswith(rel + "/") or sums.get(name) != sha:
        sys.exit(f"{plat}: {url} with sha256 {sha} is not in v{rt}'s SHA256SUMS")
    status = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60).status
    if status != 200:
        sys.exit(f"{plat}: {url} answers HTTP {status}")
    print(f"    200  {name}  sha256 matches")
src = fetch(f"https://raw.githubusercontent.com/{repo}/v{rt}/src/Nereus.jl")
got = re.search(r"^const PY_API_VERSION\s*=\s*(\d+)", src, re.M)
if not got or got.group(1) != api:
    sys.exit(f"contract mismatch: Nereus {rt} has PY_API_VERSION "
             f"{got.group(1) if got else '?'}, this client {api}")
print(f"    contract ok: PY_API_VERSION={api} on both sides")
PY

# ---- tests and dists ------------------------------------------------------
say "unit tests"
docker run --rm --name "nereus-client-test-$$" \
  --label cl.jvines.owner=nereus-release \
  -v "$PY_ROOT":/src:ro python:3.12-slim \
  bash -c 'cp -r /src /w && cd /w && pip install -q -e ".[test]" && python -m pytest tests/ -q --ignore=tests/test_smoke.py' \
  || die "the unit tests failed"

say "astronereus dists"
rm -rf "$PY_ROOT/dist"
docker run --rm --name "nereus-client-build-$$" \
  --label cl.jvines.owner=nereus-release \
  -v "$PY_ROOT":/src -w /src python:3.12-slim \
  bash -c 'pip install -q build twine && python -m build --outdir dist && twine check dist/*' \
  || die "astronereus build or twine check failed"
rm -rf "$OUT" && mkdir -p "$OUT/dist" && cp "$PY_ROOT"/dist/* "$OUT/dist/"
rm -rf "$PY_ROOT/dist"
wheel=$(ls "$OUT"/dist/astronereus-"$PY_VER"-*.whl 2>/dev/null | head -1)
[ -n "$wheel" ] || die "no astronereus-$PY_VER wheel was built"
"$here/wheel-matches-tree.py" "$PY_ROOT/src/astronereus/_runtime.py" "$wheel" \
  || die "the wheel's runtime pointer is not the committed one"
info "$(cd "$OUT/dist" && ls | tr '\n' ' ')"

if [ "$MODE" = --check ]; then
  say "CHECKED -- nothing has been published"
  info "publish with: tools/release/publish-client.sh $PY_VER --confirm"
  exit 0
fi

# ---- publish --------------------------------------------------------------
say "PyPI"
: "${TWINE_PASSWORD:?TWINE_PASSWORD is not set}"
docker run --rm --name "nereus-client-upload-$$" \
  --label cl.jvines.owner=nereus-release \
  -e TWINE_USERNAME=__token__ -e TWINE_PASSWORD \
  -v "$OUT/dist":/dist:ro python:3.12-slim \
  bash -c 'pip install -q twine && twine upload /dist/*' \
  || die "twine upload failed; nothing else was changed, retry"

say "tagging nereus-py"
git -C "$PY_ROOT" tag -a "v$PY_VER" -m "astronereus $PY_VER" 2>/dev/null || true
git -C "$PY_ROOT" push origin "v$PY_VER" || warn "could not push the astronereus tag"

for i in $(seq 1 30); do
  code=$(curl -s -o /dev/null -w '%{http_code}' "https://pypi.org/pypi/astronereus/$PY_VER/json")
  [ "$code" = 200 ] && break
  sleep 10
done
[ "$code" = 200 ] || warn "PyPI has not listed $PY_VER after 5 min; check it by hand"
say "PUBLISHED  astronereus $PY_VER (runtime Nereus $RT)"
