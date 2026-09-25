#!/usr/bin/env bash
# Phase 1 of a release. Builds and verifies EVERYTHING; publishes NOTHING.
#
#   tools/release/prepare.sh <nereus-version> <astronereus-version>
#
# Nothing here leaves the fleet: no tag is pushed, no GitHub release is made,
# no wheel is uploaded. That is the point. A PyPI filename can never be reused
# and a GitHub release that half-uploads takes the whole release with it, so
# everything that CAN be checked is checked before anything irreversible runs.
#
# On success it leaves, under <stage>/v<version>/:
#   nereus-runtime-<julia>-<platform>.tar.zst   x3, each smoke-tested
#   SHA256SUMS
#   BUILD_INFO.<platform>.txt                   commit, tree, cpu target, api
#   manifest.json                               what publish.sh consumes
#   dist/                                       wheel + sdist, twine-checked
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
. "$here/lib.sh"

JL_VER="${1:?usage: prepare.sh <nereus-version> <astronereus-version>}"
PY_VER="${2:?usage: prepare.sh <nereus-version> <astronereus-version>}"
load_config

JL_ROOT=$(cd "$here/../.." && pwd)
PY_ROOT="$CFG_NEREUS_PY"
OUT="$CFG_STAGE/v$JL_VER"

say "release prepare: Nereus $JL_VER / astronereus $PY_VER"
info "julia      $CFG_JULIA"
info "nereus.jl  $JL_ROOT"
info "nereus-py  $PY_ROOT"
info "staging    $OUT"

# ---- gates ----------------------------------------------------------------
say "gates"
[ -z "$(git -C "$JL_ROOT" status --porcelain --untracked-files=no)" ] \
  || die "Nereus.jl tree is dirty. A bundle built from it would match no commit."
[ -z "$(git -C "$PY_ROOT" status --porcelain --untracked-files=no)" ] \
  || die "nereus-py tree is dirty."

have=$(jl_version "$JL_ROOT")
[ "$have" = "$JL_VER" ] || die "Project.toml says $have, you asked for $JL_VER. Bump it first."
havep=$(py_version "$PY_ROOT"); havei=$(py_initver "$PY_ROOT")
[ "$havep" = "$PY_VER" ] || die "pyproject.toml says $havep, you asked for $PY_VER."
[ "$havei" = "$PY_VER" ] || die "__init__.py says $havei but pyproject says $havep -- they must agree."
info "versions ok: Nereus $JL_VER, astronereus $PY_VER"

check_contract "$JL_ROOT" "$PY_ROOT"

if git -C "$JL_ROOT" rev-parse "v$JL_VER" >/dev/null 2>&1; then
  die "tag v$JL_VER already exists. A released version is never rebuilt --
its bundles are already on a GitHub release that published wheels resolve."
fi

say "leak scan (both repositories, tree and history)"
"$JL_ROOT/ci/leak-scan.sh" "$JL_ROOT" || die "Nereus.jl would publish internal infrastructure"
"$PY_ROOT/ci/leak-scan.sh" "$PY_ROOT" || die "nereus-py would publish internal infrastructure"

TREE=$(tree_of "$JL_ROOT")
info "source tree $TREE"

# ---- test suite -----------------------------------------------------------
# build_bundle.sh gates on ONE fit_rv smoke test, which passes whatever else is
# broken: v0.4.0 shipped with test_builder.jl failing since the priors refactor
# because nothing ran it. SKIP_TESTS is for re-running a prepare whose suite
# already passed on this exact tree, not for skipping it on a new one.
if [ -n "${SKIP_TESTS:-}" ]; then
  warn "SKIP_TESTS set -- the suite is NOT part of this verification"
else
  say "test suite (5 shards, in the CI image)"
  ( cd "$JL_ROOT" && docker build -q -t "nereus-ci:julia-$CFG_JULIA" ci/ >/dev/null ) || die "CI image build failed"
  docker run --rm --name "nereus-release-suite-$$" \
    --label cl.jvines.owner=nereus-release \
    --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -e JULIA_DEPOT_PATH=/depot -v "$CFG_DEPOT":/depot \
    -v "$JL_ROOT":/repo -w /repo --cpus 12 --memory 24g \
    "nereus-ci:julia-$CFG_JULIA" ci/run_tests.sh 5 3 6 \
    || die "the suite failed; nothing is built from a red tree"
fi

mkdir -p "$OUT"

# ---- bundles --------------------------------------------------------------
# Three platforms, three machines. pkgimages are native code: portable across
# CPUs of one architecture, never across architectures, so each is built where
# it will run. The two local ones run concurrently; the x86_64 build is driven
# over ssh on whichever candidate the coordinator can give us.
say "building runtime bundles"

build_local_mac() {
  ( cd "$PY_ROOT" && OUT_DIR="$OUT" NEREUS_JL="$JL_ROOT" NEREUS_PY="$PY_ROOT" \
      WORK="$CFG_STAGE/work-macos" tools/build_bundle.sh ) \
    >"$OUT/build-macos-arm64.log" 2>&1
}
build_local_docker_arm() {
  ( cd "$PY_ROOT" && OUT_DIR="$OUT" NEREUS_JL="$JL_ROOT" NEREUS_PY="$PY_ROOT" \
      tools/build_bundle_docker.sh linux/arm64 ) \
    >"$OUT/build-linux-aarch64.log" 2>&1
}

build_local_mac &            mac_pid=$!
build_local_docker_arm &     arm_pid=$!

say "picking an x86_64 builder"
X86=$(pick_x86_host) || die "no x86_64 candidate available: tried $CFG_X86"
info "building linux-x86_64 on the first available candidate"
"$here/build-remote.sh" "$X86" "$OUT" "$JL_VER" >"$OUT/build-linux-x86_64.log" 2>&1 &
x86_pid=$!

fail=0
wait "$mac_pid" || { warn "macos-arm64 build failed (see build-macos-arm64.log)"; fail=1; }
wait "$arm_pid" || { warn "linux-aarch64 build failed (see build-linux-aarch64.log)"; fail=1; }
wait "$x86_pid" || { warn "linux-x86_64 build failed (see build-linux-x86_64.log)"; fail=1; }
[ "$fail" -eq 0 ] || die "at least one bundle failed to build"

# ---- verify the three agree ------------------------------------------------
say "verifying the bundles"
n=$(ls "$OUT"/nereus-runtime-*.tar.zst 2>/dev/null | wc -l | tr -d ' ')
[ "$n" -eq 3 ] || die "expected 3 bundles, found $n"
for f in "$OUT"/nereus-runtime-*.tar.zst; do
  info "$(basename "$f")  $(du -h "$f" | cut -f1)"
done
( cd "$OUT" && shasum -a 256 nereus-runtime-*.tar.zst > SHA256SUMS )
info "SHA256SUMS written"

# ---- astronereus dists -----------------------------------------------------
# The wheel hard-codes the release URLs and their checksums, so it can only be
# built once those checksums exist -- but it must NOT be uploaded before the
# release is live, because a published filename can never be replaced.
say "astronereus dists"
"$here/patch-runtime.sh" "$PY_ROOT" "$JL_VER" "$OUT/SHA256SUMS" "$CFG_GH_REPO"
rm -rf "$PY_ROOT/dist"
docker run --rm --name "nereus-release-build-$$" \
  --label cl.jvines.owner=nereus-release \
  -v "$PY_ROOT":/src -w /src python:3.12-slim \
  bash -c 'pip install -q build twine && python -m build --outdir dist && twine check dist/*' \
  || die "astronereus build or twine check failed"
mkdir -p "$OUT/dist" && cp "$PY_ROOT"/dist/* "$OUT/dist/"

# ---- manifest --------------------------------------------------------------
python3 - "$OUT" "$JL_VER" "$PY_VER" "$TREE" "$CFG_GH_REPO" "$CFG_JULIA" <<'PY'
import hashlib, json, os, sys
out, jlv, pyv, tree, repo, julia = sys.argv[1:7]
bundles = {}
for name in sorted(os.listdir(out)):
    if name.endswith(".tar.zst"):
        h = hashlib.sha256(open(os.path.join(out, name), "rb").read()).hexdigest()
        plat = name.replace(f"nereus-runtime-{julia}-", "").replace(".tar.zst", "")
        bundles[plat] = {"file": name, "sha256": h}
json.dump({"nereus": jlv, "astronereus": pyv, "tree": tree,
           "github_repo": repo, "julia": julia, "bundles": bundles,
           "dists": sorted(os.listdir(os.path.join(out, "dist")))},
          open(os.path.join(out, "manifest.json"), "w"), indent=2)
print("    manifest.json written:", len(bundles), "bundles")
PY

say "PREPARED -- nothing has been published"
info "staged at $OUT"
info "publish with: tools/release/publish.sh $JL_VER $PY_VER --confirm"
