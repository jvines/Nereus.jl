#!/usr/bin/env bash
# Run the test suite as N parallel shards, one Julia process each
# (NEREUS_TEST_SHARD=i/N; the split is test/shards.jl). Used by
# .forgejo/workflows/ci.yml inside the ci/Dockerfile image, from the repo root.
#
#   ci/run_tests.sh [n_shards] [threads_per_shard] [threads_shard_1]
#
# Shard 1 holds the heaviest unit (test/shards.jl deals longest-first, so it
# gets the trans-dim default-settings fit alone), and that fit parallelises
# over walkers and temperatures: give it more threads than the rest.
#
# Each shard writes its own log, printed whole once all have finished --
# interleaved output from several processes is unreadable -- failing shards
# first. Exits non-zero if any shard failed.
set -uo pipefail

n=${1:-5}
threads=${2:-3}
threads1=${3:-$threads}
logs=$(mktemp -d)
trap 'rm -rf "$logs"' EXIT

pids=()
for i in $(seq 1 "$n"); do
    t=$threads; [ "$i" -eq 1 ] && t=$threads1
    NEREUS_TEST_SHARD="$i/$n" julia --project=. --threads="$t" test/runtests.jl \
        > "$logs/$i.log" 2>&1 &
    pids+=("$!")
done

passed=() failed=()
for i in $(seq 1 "$n"); do
    if wait "${pids[$((i - 1))]}"; then passed+=("$i"); else failed+=("$i"); fi
done

for i in "${failed[@]}" "${passed[@]}"; do
    echo
    echo "================================ shard $i/$n ================================"
    cat "$logs/$i.log"
done

echo
echo "passed: ${passed[*]:-none}   failed: ${failed[*]:-none}"
[ ${#failed[@]} -eq 0 ]
