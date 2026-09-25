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
PY_ROOT="$CFG_NEREUS_PY"

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
  local src="$1" name="$2" want got
  [ -d "$src/.git" ] || die "$src is not a standalone clone (its .git is not a directory)"
  want=$(tree_of "$src")
  # shellcheck disable=SC2086
  COPYFILE_DISABLE=1 tar --no-xattrs -C "$src" -cf - .git | ssh $SSHOPT "$HOST" \
    "rm -rf nereus-build-src/$name && mkdir -p nereus-build-src/$name &&
     tar -xf - -C nereus-build-src/$name &&
     git -C nereus-build-src/$name reset -q --hard &&
     git -C nereus-build-src/$name clean -fdxq" \
    || die "could not ship $name to the builder"
  # shellcheck disable=SC2086
  got=$(ssh $SSHOPT "$HOST" "git -C nereus-build-src/$name rev-parse 'HEAD^{tree}'") \
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

say "linux-x86_64 on the chosen candidate"
if [ "$CLAIMED" = "--claimed" ]; then
  # Already inside a fleet claim; build, fetch, return.
  run_build || die "remote build failed"
  fetch_bundle
  exit 0
fi
if printf '%s\n' $CFG_NOCLAIM | grep -qx "$HOST"; then
  info "always-on node: no claim needed"
  run_build || die "remote build failed"
  fetch_bundle
else
  info "claiming through the coordinator for the duration of the build"
  # `fleet with` runs the command LOCALLY between claim and release, so the
  # ssh below is our own already-verified invocation rather than fleet's.
  # TTL is sized well over the ~10-20 min build: renewal by the wrapper is not
  # something to rely on, so the TTL is a hard outer bound with real margin.
  # Re-enters THIS script with --claimed, which skips straight to the build:
  # one script, one code path, and the claim brackets exactly this process.
  "$CFG_FLEET" with "$HOST" --ttl 45m --reason "nereus $VER bundle build" -- \
    bash "$here/build-remote.sh" "$HOST" "$OUT" "$VER" --claimed \
    || die "remote build failed (claim released)"
fi
