#!/usr/bin/env bash
# Build the linux-x86_64 bundle on a remote fleet node and bring it back.
#
#   tools/release/build-remote.sh <host> <out-dir> <version>
#
# An elastic node is held by `fleet with`, which brackets exactly ONE local
# process with a claim and a release. That shape matters: a bare `fleet claim`
# followed by an interactive hop releases the claim when the claiming shell
# exits, which once dropped a node out from under a running build.
#
# QUOTING. An ssh "exec" request carries exactly one string; extra argv words
# are joined with spaces and REPARSED by the remote shell, so local quoting
# does not survive. Values are therefore passed as a single %q-escaped prefix
# and the script body arrives on stdin under a QUOTED heredoc delimiter, so
# nothing is expanded locally. Never add `-n`: it rebinds stdin to /dev/null
# and the heredoc silently never arrives.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
. "$here/lib.sh"
load_config

HOST="${1:?usage: build-remote.sh <host> <out-dir> <version> [--claimed]}"
OUT="${2:?}"; VER="${3:?}"; CLAIMED="${4:-}"
SSHOPT="-o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4"
JL_ROOT=$(cd "$here/../.." && pwd)
PY_ROOT="${NEREUS_PY_ROOT:?set by prepare.sh: the nereus-py clone to build against}"

# Put one repository on the builder as exactly this commit. Only `.git` is
# sent, then checked out there: the builder gets the same commit id and tree,
# clean, and none of the untracked run outputs a working tree accumulates
# (5.7 GB of them in one dev checkout). Nothing is assumed to be there already
# -- the builder is whichever candidate the coordinator hands over, and until
# this existed the build died on a source directory no step ever created.
# Tar over ssh, not rsync: macOS ships openrsync, which hands `-e "ssh <opts>"`
# to ssh as ONE argument, so every -o option breaks the connection. And tar
# with COPYFILE_DISABLE and --no-xattrs: macOS tar otherwise adds an AppleDouble
# `._` twin of every file (2274 of them for this .git), which GNU tar on the
# builder unpacks into objects/pack, where git reads them as corrupt pack
# indexes.
ship_repo() {
  local src="$1" name="$2" base="${3:-nereus-build-src}" want got
  [ -d "$src/.git" ] || die "$src is not a standalone clone (its .git is not a directory)"
  want=$(tree_of "$src")
  # shellcheck disable=SC2086
  COPYFILE_DISABLE=1 tar --no-xattrs -C "$src" -cf - .git | ssh $SSHOPT "$HOST" \
    "rm -rf $base/$name && mkdir -p $base/$name &&
     tar -xf - -C $base/$name &&
     git -C $base/$name reset -q --hard &&
     git -C $base/$name clean -fdxq" \
    || die "could not ship $name to the builder"
  # shellcheck disable=SC2086
  got=$(ssh $SSHOPT "$HOST" "git -C $base/$name rev-parse 'HEAD^{tree}'") \
    || die "could not read $name back on the builder"
  [ "$got" = "$want" ] || die "$name on the builder is tree $got, expected $want"
  info "$name shipped: tree $want"
}

run_build() {
  ship_repo "$JL_ROOT" Nereus.jl
  ship_repo "$PY_ROOT" nereus-py
  # WORK is pinned under $HOME on purpose: mktemp -d lands in /tmp, which is
  # 16 GB on some of these nodes, and the build writes ~4 GB plus an unpacked
  # Julia. $HOME has hundreds of GB on all of them.
  local prefix
  prefix=$(printf 'NEREUS_VER=%q JULIA_VER=%q bash -s' "$VER" "$CFG_JULIA")
  # shellcheck disable=SC2086
  ssh $SSHOPT "$HOST" "$prefix" <<'REMOTE'
set -euo pipefail
export WORK="$HOME/.cache/nereus-build"
export OUT_DIR="$HOME/nereus-bundles/$NEREUS_VER"
# OUT_DIR too: a bundle an earlier build left under this version must never
# be the one copied back.
rm -rf "$WORK" "$OUT_DIR"; mkdir -p "$WORK" "$OUT_DIR"
echo "remote: building on $(uname -s)-$(uname -m), $(nproc) cpus"
OUT_DIR="$OUT_DIR" WORK="$WORK" NEREUS_JL="$HOME/nereus-build-src/Nereus.jl" \
  NEREUS_PY="$HOME/nereus-build-src/nereus-py" \
  bash "$HOME/nereus-build-src/nereus-py/tools/build_bundle.sh"
ls -la "$OUT_DIR"
REMOTE
}

# Copied back while the claim is still held: once `fleet with` returns, the
# node is free to be suspended mid-transfer.
fetch_bundle() {
  local f="nereus-runtime-${CFG_JULIA}-linux-x86_64.tar.zst"
  say "copying the bundle back"
  # shellcheck disable=SC2086
  scp -q $SSHOPT "${HOST}:nereus-bundles/${VER}/$f" "$OUT/" \
    || die "could not copy the bundle back from the builder"
  info "retrieved $(du -h "$OUT/$f" | cut -f1)"
}

# The full suite, in the CI image, sharded to this machine: 3 threads and about
# 5 GB per shard, leaving room for the bundle build running beside it. Its own
# copy of the source (nereus-suite-src), since the suite writes test outputs
# into its tree. The depot persists on the builder between releases, so only
# the first release on a machine precompiles from cold. The verdict goes to
# $OUT/suite.rc, which prepare.sh reads before building anything further.
run_suite() {
  local rc=0 prefix
  prefix=$(printf 'JULIA_VER=%q bash -s' "$CFG_JULIA")
  # A subshell, so a `die` in ship_repo ends it and not the caller.
  (
    ship_repo "$JL_ROOT" Nereus.jl nereus-suite-src &&
    # shellcheck disable=SC2086
    ssh $SSHOPT "$HOST" "$prefix" <<'REMOTE'
set -euo pipefail
cd "$HOME/nereus-suite-src/Nereus.jl"
depot="$HOME/.cache/nereus-suite-depot"; mkdir -p "$depot"
cpus=$(nproc); memg=$(awk '/^MemTotal/{print int($2/1048576)}' /proc/meminfo)
n=$(( cpus / 3 )); m=$(( (memg - 12) / 5 ))
[ "$m" -lt "$n" ] && n=$m
[ "$n" -lt 2 ] && n=2
echo "remote suite: $n shards on $(uname -s)-$(uname -m), $cpus cpus, $memg GB"
docker build -q -t "nereus-ci:julia-$JULIA_VER" ci/ >/dev/null
run() {
  docker run --rm --label cl.jvines.owner=nereus-release \
    --user "$(id -u):$(id -g)" -e HOME=/tmp -e JULIA_DEPOT_PATH=/depot \
    -v "$depot":/depot -v "$PWD":/repo -w /repo "nereus-ci:julia-$JULIA_VER" "$@"
}
run julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
run ci/run_tests.sh "$n" 3 3
REMOTE
  ) >"$OUT/suite-linux-x86_64.log" 2>&1 || rc=$?
  echo "$rc" >"$OUT/suite.rc"
  return 0
}

# Bundle build, and the suite beside it when prepare.sh asks (REMOTE_SUITE).
# Both finish before the claim is given back; the suite's verdict is in
# $OUT/suite.rc, the build's is this function's status.
build_and_test() {
  local spid=""
  if [ -n "${REMOTE_SUITE:-}" ]; then run_suite & spid=$!; fi
  local brc=0
  run_build || brc=$?
  [ -n "$spid" ] && wait "$spid"
  [ "$brc" -eq 0 ] || die "remote build failed"
  fetch_bundle
}

say "linux-x86_64 on the chosen candidate"
if [ "$CLAIMED" = "--claimed" ]; then
  # Already inside a fleet claim; build (and test), fetch, return.
  build_and_test
  exit 0
fi
if printf '%s\n' $CFG_NOCLAIM | grep -qx "$HOST"; then
  info "always-on node: no claim needed"
  build_and_test
else
  info "claiming through the coordinator for the duration of the build"
  # `fleet with` runs the command LOCALLY between claim and release, so the
  # ssh below is our own already-verified invocation rather than fleet's.
  # TTL is sized well over the build and the suite beside it: renewal by the wrapper is not
  # something to rely on, so the TTL is a hard outer bound with real margin.
  # Re-enters THIS script with --claimed, which skips straight to the build:
  # one script, one code path, and the claim brackets exactly this process.
  # 90 minutes: the suite beside the build runs ~15 minutes on the largest
  # candidate but over an hour on the smallest.
  "$CFG_FLEET" with "$HOST" --ttl 90m --reason "nereus $VER bundle build and suite" -- \
    env REMOTE_SUITE="${REMOTE_SUITE:-}" NEREUS_PY_ROOT="$PY_ROOT" \
    bash "$here/build-remote.sh" "$HOST" "$OUT" "$VER" --claimed \
    || die "remote build failed (claim released)"
fi
