#!/usr/bin/env bash
# Self-test for ci/run_tests.sh. A stand-in `julia` takes the place of the
# suite, so this needs no Julia and finishes in seconds; it checks what the
# script itself is responsible for -- exit status, every shard's log printed,
# failing shards first, shard 1's extra threads, the summary line.
#
#   ci/test_run_tests.sh              under the bash on PATH
#   /bin/bash ci/test_run_tests.sh    under macOS's bash 3.2
#
# Run it under each bash the shards are run with: the script is exercised by
# the bash running this test. Bash before 4.4 is the one that matters -- it
# treats "${arr[@]}" of an empty array as unbound under set -u, which aborted
# run_tests.sh whenever no shard failed (or none passed), and took the logs
# with it.
set -uo pipefail

here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

mkdir "$work/bin"
cat > "$work/bin/julia" <<'EOF'
#!/bin/sh
# Stand-in for julia: report the shard and the arguments, then pass unless
# the shard's number is listed in $STUB_FAIL.
echo "stub ran shard $NEREUS_TEST_SHARD with: $*"
case " ${STUB_FAIL:-} " in
    *" ${NEREUS_TEST_SHARD%/*} "*) echo "stub failed shard $NEREUS_TEST_SHARD"; exit 1 ;;
esac
exit 0
EOF
chmod +x "$work/bin/julia"

nbad=0
fail() { echo "FAIL [$name]: $*"; nbad=$((nbad + 1)); }

# run <name> <fail list> <run_tests.sh args...>: leaves $out and $rc.
run() {
    name=$1 stub_fail=$2
    shift 2
    out=$(cd "$work" && PATH="$work/bin:$PATH" STUB_FAIL="$stub_fail" \
        "$BASH" "$here/run_tests.sh" "$@" 2>&1)
    rc=$?
}
expect_line() { printf '%s\n' "$out" | grep -qxF -- "$1" || fail "no line '$1'"; }
# The shard headers in the order they were printed, as "1 3 2".
order() { printf '%s\n' "$out" | sed -n 's|^=* shard \([0-9]*\)/[0-9]* =*$|\1|p' | tr '\n' ' ' | sed 's/ $//'; }

run "all pass" "" 3 2 5
[ "$rc" -eq 0 ] || fail "exit $rc, expected 0"
[ "$(order)" = "1 2 3" ] || fail "shards printed as '$(order)'"
expect_line "stub ran shard 1/3 with: --project=. --threads=5 test/runtests.jl"
expect_line "stub ran shard 2/3 with: --project=. --threads=2 test/runtests.jl"
expect_line "stub ran shard 3/3 with: --project=. --threads=2 test/runtests.jl"
expect_line "passed: 1 2 3   failed: none"

run "one fails" "2" 3
[ "$rc" -ne 0 ] || fail "exit 0 with a failed shard"
[ "$(order)" = "2 1 3" ] || fail "shards printed as '$(order)', failing shard not first"
expect_line "stub failed shard 2/3"
expect_line "stub ran shard 1/3 with: --project=. --threads=3 test/runtests.jl"
expect_line "passed: 1 3   failed: 2"

run "all fail" "1 2" 2
[ "$rc" -ne 0 ] || fail "exit 0 with every shard failed"
[ "$(order)" = "1 2" ] || fail "shards printed as '$(order)'"
expect_line "stub failed shard 1/2"
expect_line "stub failed shard 2/2"
expect_line "passed: none   failed: 1 2"

run "one at a time" "" 3 1 1 1
[ "$rc" -eq 0 ] || fail "exit $rc, expected 0"
[ "$(order)" = "1 2 3" ] || fail "shards printed as '$(order)'"
expect_line "passed: 1 2 3   failed: none"

if [ "$nbad" -eq 0 ]; then
    echo "run_tests.sh self-test: all checks passed under bash $BASH_VERSION"
else
    echo "run_tests.sh self-test: $nbad check(s) failed under bash $BASH_VERSION"
    exit 1
fi
