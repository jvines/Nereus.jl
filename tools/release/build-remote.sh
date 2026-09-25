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

run_build() {
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
rm -rf "$WORK"; mkdir -p "$WORK" "$OUT_DIR"
cd "$HOME/nereus-build-src"
git fetch -q --depth 1 origin "$NEREUS_REF" 2>/dev/null || true
echo "remote: building on $(uname -s)-$(uname -m), $(nproc) cpus"
OUT_DIR="$OUT_DIR" WORK="$WORK" NEREUS_JL="$HOME/nereus-build-src/Nereus.jl" \
  NEREUS_PY="$HOME/nereus-build-src/nereus-py" \
  bash "$HOME/nereus-build-src/nereus-py/tools/build_bundle.sh"
ls -la "$OUT_DIR"
REMOTE
}

say "linux-x86_64 on the chosen candidate"
if [ "$CLAIMED" = "--claimed" ]; then
  # Already inside a fleet claim; just build and return.
  run_build || die "remote build failed"
  exit 0
fi
if printf '%s\n' $CFG_NOCLAIM | grep -qx "$HOST"; then
  info "always-on node: no claim needed"
  run_build || die "remote build failed"
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

say "copying the bundle back"
rsync -az --timeout=180 -e "ssh $SSHOPT" \
  "${HOST}:nereus-bundles/${VER}/nereus-runtime-${CFG_JULIA}-linux-x86_64.tar.zst" "$OUT/" \
  || die "could not copy the bundle back from the builder"
info "retrieved $(du -h "$OUT/nereus-runtime-${CFG_JULIA}-linux-x86_64.tar.zst" | cut -f1)"
