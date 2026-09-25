#!/usr/bin/env bash
# Point astronereus at a new release's bundles.
#
#   tools/release/patch-runtime.sh <nereus-py> <version> <SHA256SUMS> <gh-repo>
#
# Rewrites _REL, RUNTIME_VERSION and the three BUNDLES checksums. Done as a
# script rather than by hand because the checksums are the one thing a wheel
# can never be corrected about: the filename is burned on upload.
set -euo pipefail
PY_ROOT="${1:?}"; VER="${2:?}"; SUMS="${3:?}"; REPO="${4:?}"
python3 - "$PY_ROOT/src/astronereus/_runtime.py" "$VER" "$SUMS" "$REPO" <<'PY'
import re, sys
path, ver, sums, repo = sys.argv[1:5]
want = {}
for line in open(sums):
    h, name = line.split()
    name = name.lstrip("*")
    for plat in ("macos-arm64", "linux-x86_64", "linux-aarch64"):
        if name.endswith(plat + ".tar.zst"):
            want[plat] = h
missing = [p for p in ("macos-arm64", "linux-x86_64", "linux-aarch64") if p not in want]
if missing:
    sys.exit("SHA256SUMS is missing: " + ", ".join(missing))

s = open(path, encoding="utf8").read()
s, n = re.subn(r'_REL = "https://github\.com/[^"]+"',
               f'_REL = "https://github.com/{repo}/releases/download/v{ver}"', s)
assert n == 1, f"_REL not found ({n} matches)"
s, n = re.subn(r'RUNTIME_VERSION = "[^"]+"', f'RUNTIME_VERSION = "{ver}"', s)
assert n == 1, f"RUNTIME_VERSION not found ({n} matches)"
for plat, h in want.items():
    s, n = re.subn(r'("' + re.escape(plat) + r'": \(\s*\n\s*f"\{_REL\}/[^"]+",\s*\n\s*")[0-9a-f]{64}(")',
                   lambda m: m.group(1) + h + m.group(2), s)
    assert n == 1, f"{plat} checksum line not found ({n} matches)"
open(path, "w", encoding="utf8").write(s)
print(f"    _runtime.py -> v{ver}, 3 checksums")
PY
