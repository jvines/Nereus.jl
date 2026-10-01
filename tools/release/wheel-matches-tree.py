#!/usr/bin/env python3
"""Refuse to record a runtime pointer that was not the one published.

    wheel-matches-tree.py <_runtime.py> <built.whl>

prepare.sh rewrites _runtime.py in place so the wheel is built against the new
release, and publish.sh commits that file afterwards so the repository matches
what shipped. If the two ever disagree -- a stray edit between build and
upload, a stale wheel, a checkout over the working tree -- committing would
record a pointer nobody can install, which is worse than not committing at all.
Exit non-zero and let publish.sh stop.
"""
import sys
import zipfile


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    tree_path, wheel_path = sys.argv[1], sys.argv[2]
    with open(tree_path, encoding="utf8") as fh:
        tree = fh.read()
    with zipfile.ZipFile(wheel_path) as z:
        names = [n for n in z.namelist() if n.endswith("astronereus/_runtime.py")]
        if not names:
            print(f"no _runtime.py inside {wheel_path}", file=sys.stderr)
            return 2
        shipped = z.read(names[0]).decode("utf8")
    if shipped != tree:
        print(f"_runtime.py in the working tree differs from the one inside\n"
              f"{wheel_path}, which is what was uploaded. Refusing to commit a\n"
              f"pointer that was never published.", file=sys.stderr)
        return 1
    print("    _runtime.py matches the uploaded wheel")
    return 0


if __name__ == "__main__":
    sys.exit(main())
