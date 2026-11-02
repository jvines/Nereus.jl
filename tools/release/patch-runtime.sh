#!/usr/bin/env bash
# Point astronereus at a new release's bundles.
#
#   tools/release/patch-runtime.sh <nereus-py> <version> <SHA256SUMS> <gh-repo>
#
# Rewrites _REL, RUNTIME_VERSION, JULIA_VERSION and the three BUNDLES entries
# (file name and checksum). Done as a script rather than by hand because the
# bundle names and checksums are the one thing a wheel can never be corrected
# about: the filename is burned on upload. The Julia version is read off the
# bundle names in SHA256SUMS (nereus-runtime-<julia>-<platform>.tar.zst), so the
# wheel names exactly the bundles that were built.
set -euo pipefail
PY_ROOT="${1:?}"; VER="${2:?}"; SUMS="${3:?}"; REPO="${4:?}"
python3 - "$PY_ROOT/src/astronereus/_runtime.py" "$VER" "$SUMS" "$REPO" <<'PY'
import re, sys
path, ver, sums, repo = sys.argv[1:5]
want, julia = {}, set()
for line in open(sums):
    h, name = line.split()
    name = name.lstrip("*")
    for plat in ("macos-arm64", "linux-x86_64", "linux-aarch64"):
        m = re.fullmatch(r"nereus-runtime-(\d+\.\d+\.\d+)-" + re.escape(plat) + r"\.tar\.zst", name)
        if m:
            want[plat] = (name, h)
            julia.add(m.group(1))
missing = [p for p in ("macos-arm64", "linux-x86_64", "linux-aarch64") if p not in want]
if missing:
    sys.exit("SHA256SUMS is missing: " + ", ".join(missing))
if len(julia) != 1:
    sys.exit("the bundles in SHA256SUMS disagree on the Julia version: " + ", ".join(sorted(julia)))
jver = julia.pop()

s = open(path, encoding="utf8").read()
s, n = re.subn(r'_REL = "https://github\.com/[^"]+"',
               f'_REL = "https://github.com/{repo}/releases/download/v{ver}"', s)
assert n == 1, f"_REL not found ({n} matches)"
s, n = re.subn(r'RUNTIME_VERSION = "[^"]+"', f'RUNTIME_VERSION = "{ver}"', s)
assert n == 1, f"RUNTIME_VERSION not found ({n} matches)"
s, n = re.subn(r'^JULIA_VERSION = "[^"]+"', f'JULIA_VERSION = "{jver}"', s, flags=re.M)
assert n == 1, f"JULIA_VERSION not found ({n} matches)"
for plat, (name, h) in want.items():
    s, n = re.subn(r'("' + re.escape(plat) + r'": \(\s*\n\s*f"\{_REL\}/)[^"]+(",\s*\n\s*")[0-9a-f]{64}(")',
                   lambda m: m.group(1) + name + m.group(2) + h + m.group(3), s)
    assert n == 1, f"{plat} bundle entry not found ({n} matches)"
open(path, "w", encoding="utf8").write(s)
print(f"    _runtime.py -> v{ver}, julia {jver}, 3 bundles")
PY
