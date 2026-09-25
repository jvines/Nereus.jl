# Shared helpers for the release pipeline. Sourced, not executed.
#
# NO HOST NAMES OR PATHS HERE. This repository is mirrored publicly on every
# push, on every branch. Everything machine-specific comes from the
# RELEASE_CONFIG Actions variable (or $NEREUS_RELEASE_CONFIG pointing at a
# file) and is validated on load, so a missing value fails loudly here rather
# than halfway through a twenty-minute build.
set -uo pipefail

say()  { printf '\033[1m==>\033[0m %s\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '\033[33m !! %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31mrelease: %s\033[0m\n' "$*" >&2; exit 1; }

# ---- configuration --------------------------------------------------------
load_config() {
  local raw=""
  if [ -n "${RELEASE_CONFIG:-}" ]; then raw="$RELEASE_CONFIG"
  elif [ -n "${NEREUS_RELEASE_CONFIG:-}" ] && [ -r "$NEREUS_RELEASE_CONFIG" ]; then raw=$(cat "$NEREUS_RELEASE_CONFIG")
  elif [ -r "$HOME/.dotfiles/nereus/release-config.json" ]; then raw=$(cat "$HOME/.dotfiles/nereus/release-config.json")
  else die "no release config. Set RELEASE_CONFIG or NEREUS_RELEASE_CONFIG."
  fi
  # One python call, so a malformed config is one error rather than six.
  eval "$(printf '%s' "$raw" | python3 -c '
import json, sys, shlex
c = json.load(sys.stdin)
need = ["fleet", "nereus_jl", "nereus_py", "stage", "x86_candidates", "julia_version", "forgejo_base", "github_repo"]
missing = [k for k in need if k not in c]
if missing:
    sys.exit("config is missing: " + ", ".join(missing))
print("CFG_FLEET=" + shlex.quote(c["fleet"]))
print("CFG_NEREUS_JL=" + shlex.quote(c["nereus_jl"]))
print("CFG_NEREUS_PY=" + shlex.quote(c["nereus_py"]))
print("CFG_STAGE=" + shlex.quote(c["stage"]))
# The suite reuses an existing CI depot when one is configured. A cold
# depot precompiles ~636 packages, which is most of an hour before a
# single test runs -- and the depot is the one input a release does not
# need to rebuild from scratch to be trustworthy.
print("CFG_DEPOT=" + shlex.quote(c.get("depot") or (c["stage"].rstrip("/") + "/depot")))
print("CFG_JULIA=" + shlex.quote(c["julia_version"]))
print("CFG_FORGEJO=" + shlex.quote(c["forgejo_base"].rstrip("/")))
print("CFG_GH_REPO=" + shlex.quote(c["github_repo"]))
print("CFG_X86=" + shlex.quote(" ".join(c["x86_candidates"])))
print("CFG_NOCLAIM=" + shlex.quote(" ".join(c.get("x86_no_claim", []))))
')" || die "could not parse the release config"
  [ -x "$CFG_FLEET" ] || warn "fleet not executable at the configured path; elastic candidates will be skipped"
  [ -d "$CFG_NEREUS_PY" ] || die "nereus_py path does not exist: $CFG_NEREUS_PY"
}

# ---- version and contract gates -------------------------------------------
jl_version()  { awk -F'"' '/^version *=/{print $2; exit}' "$1/Project.toml"; }
jl_api()      { awk -F'= *' '/^const PY_API_VERSION/{gsub(/[^0-9]/,"",$2); print $2; exit}' "$1/src/Nereus.jl"; }
py_api()      { awk -F'= *' '/^PY_API_VERSION *=/{gsub(/[^0-9]/,"",$2); print $2; exit}' "$1/src/astronereus/_api.py"; }
py_version()  { awk -F'"' '/^version *=/{print $2; exit}' "$1/pyproject.toml"; }
py_initver()  { awk -F'"' '/^__version__ *=/{print $2; exit}' "$1/src/astronereus/__init__.py"; }
py_runtime()  { awk -F'"' '/^RUNTIME_VERSION *=/{print $2; exit}' "$1/src/astronereus/_runtime.py"; }

check_contract() {
  local jl="$1" py="$2" a b
  a=$(jl_api "$jl"); b=$(py_api "$py")
  [ -n "$a" ] && [ -n "$b" ] || die "could not read PY_API_VERSION from both sides (got '$a' / '$b')"
  [ "$a" = "$b" ] || die "contract mismatch: Nereus.PY_API_VERSION=$a but astronereus PY_API_VERSION=$b.
Bump BOTH, or build from a commit that matches the client. This is the check
that 0.3.0 skipped, and it shipped against a runtime that died mid-fit."
  info "contract ok: PY_API_VERSION=$a on both sides"
}

# A released bundle is identified by the TREE it was built from, not the
# commit: a build checkout made with a fresh `git init` hashes the same source
# to a different commit id with no shared ancestry, which has already cost a
# day of reconstructing which source a bundle came from.
tree_of() { git -C "$1" rev-parse "${2:-HEAD}^{tree}"; }

# ---- picking an x86_64 builder --------------------------------------------
# Tries each candidate in order. An elastic node is claimed through the
# coordinator -- never woken directly, and never slept by us. A node listed in
# x86_no_claim is always-on and needs no claim. Prints the chosen host.
pick_x86_host() {
  local h
  for h in $CFG_X86; do
    if printf '%s\n' $CFG_NOCLAIM | grep -qx "$h"; then
      if ssh -o BatchMode=yes -o ConnectTimeout=8 -n "$h" true 2>/dev/null; then
        printf '%s' "$h"; return 0
      fi
      warn "$h unreachable; trying the next candidate"
      continue
    fi
    [ -x "$CFG_FLEET" ] || continue
    if "$CFG_FLEET" status "$h" >/dev/null 2>&1; then
      printf '%s' "$h"; return 0
    fi
    warn "$h not available from the coordinator; trying the next candidate"
  done
  return 1
}
