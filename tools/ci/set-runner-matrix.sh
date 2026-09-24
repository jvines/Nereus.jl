#!/usr/bin/env bash
# Write the RUNNER_MATRIX Actions variable from values kept OUTSIDE this repository.
#
# ci.yml carries no runner label, no path and no resource budget: this
# repository is mirrored publicly on every push, so any of those committed here
# would be published. They live in the matrix file instead, which the scan in
# ci/leak-scan.sh is there to keep out of git.
#
#   tools/ci/set-runner-matrix.sh <owner>/<repo> [matrix.json]
#
# Default matrix file: ~/.dotfiles/nereus/ci-matrix.json
# Forgejo base URL:    $FORGEJO_BASE   (internal; deliberately has no default)
# Forgejo token:       $FORGEJO_TOKEN  (sops)
set -euo pipefail

repo="${1:?usage: set-runner-matrix.sh <owner>/<repo> [matrix.json]}"
file="${2:-$HOME/.dotfiles/nereus/ci-matrix.json}"
# No default: the Forgejo host is internal, and this file is published.
: "${FORGEJO_BASE:?FORGEJO_BASE is not set (the Forgejo base URL)}"
base="$FORGEJO_BASE"
: "${FORGEJO_TOKEN:?FORGEJO_TOKEN is not set -- run under sops exec-env}"

[ -r "$file" ] || { echo "no matrix file at $file" >&2; exit 1; }
python3 -c 'import json,sys; d=json.load(open(sys.argv[1]));
assert "include" in d and d["include"], "matrix needs a non-empty include list"
[k for e in d["include"] for k in ("arch","runner","ci_root","cpus","memory") if k in e or (_ for _ in ()).throw(SystemExit(f"entry missing {k}: {e}"))]' "$file"

value=$(python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1])), separators=(",",":")))' "$file")

code=$(curl -s -o /dev/null -w '%{http_code}' -X PUT \
  -H "Authorization: token $FORGEJO_TOKEN" -H 'content-type: application/json' \
  -d "$(python3 -c 'import json,sys; print(json.dumps({"value": sys.argv[1]}))' "$value")" \
  "$base/api/v1/repos/$repo/actions/variables/RUNNER_MATRIX")

case "$code" in
  201|204) echo "RUNNER_MATRIX set on $repo ($code)" ;;
  404) echo "404 -- variable does not exist yet; creating" >&2
       curl -fsS -X POST -H "Authorization: token $FORGEJO_TOKEN" -H 'content-type: application/json' \
         -d "$(python3 -c 'import json,sys; print(json.dumps({"name":"RUNNER_MATRIX","value": sys.argv[1]}))' "$value")" \
         "$base/api/v1/repos/$repo/actions/variables" >/dev/null && echo "RUNNER_MATRIX created on $repo" ;;
  *) echo "failed to set RUNNER_MATRIX: HTTP $code" >&2; exit 1 ;;
esac
