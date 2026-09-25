#!/usr/bin/env bash
# Post the prepare result back to the pull request that triggered it.
#
#   tools/release/pr-comment.sh <manifest-or-missing> <pr-number>
#
# A release prepare that only writes to a job log is a release nobody reads.
# The checksums belong on the PR, next to the diff that produced them.
# Kept as a script rather than inline YAML: a heredoc terminator has to sit at
# column 0, which silently ends a YAML block scalar and makes the workflow
# unparseable.
set -uo pipefail
MANIFEST="${1:?}"; PR="${2:?}"
: "${FORGEJO_TOKEN:?}" "${GITHUB_SERVER_URL:?}" "${GITHUB_REPOSITORY:?}"

python3 - "$MANIFEST" "$PR" <<'PY'
import json, os, sys, urllib.request

manifest, pr = sys.argv[1], sys.argv[2]
if os.path.exists(manifest):
    d = json.load(open(manifest))
    lines = [f"### Release prepared — Nereus {d['nereus']} / astronereus {d['astronereus']}", "",
             f"Built from source tree `{d['tree']}`. **Nothing has been published.**", "",
             "| platform | sha256 |", "|---|---|"]
    lines += [f"| `{p}` | `{b['sha256']}` |" for p, b in sorted(d["bundles"].items())]
    lines += ["", "Dists: " + ", ".join(f"`{x}`" for x in d["dists"]), "",
              "Publish by dispatching the Release workflow with "
              f"`{d['nereus']}` / `{d['astronereus']}` and confirm `PUBLISH`."]
    body = "\n".join(lines)
else:
    body = ("### Release prepare FAILED\n\nNothing was staged, so there is "
            "nothing to publish. See the job log.")

url = (f"{os.environ['GITHUB_SERVER_URL']}/api/v1/repos/"
       f"{os.environ['GITHUB_REPOSITORY']}/issues/{pr}/comments")
req = urllib.request.Request(url, data=json.dumps({"body": body}).encode(), method="POST",
                             headers={"Authorization": "token " + os.environ["FORGEJO_TOKEN"],
                                      "content-type": "application/json"})
try:
    print("    comment posted ->", urllib.request.urlopen(req).status)
except Exception as e:
    print("    could not comment:", e, file=sys.stderr)
PY
