#!/usr/bin/env bash
# Run the test suite as N shards, one Julia process each (NEREUS_TEST_SHARD=i/N;
# the split is test/shards.jl). Used by .forgejo/workflows/ci.yml inside the
# ci/Dockerfile image, from the repo root.
#
#   ci/run_tests.sh [n_shards] [threads_per_shard] [threads_shard_1] [max_parallel]
#
# Shard 1 holds the heaviest unit (test/shards.jl deals longest-first, so it
# gets the trans-dim default-settings fit alone), and that fit parallelises
# over walkers and temperatures: give it more threads than the rest.
#
# max_parallel (default: all of them) caps how many shards run at once. It is
# what a host short of memory turns down: five shards need more than an 11 GB
# VM holds, and the kernel killed all five. Shards and threads stay the same
# whatever it is set to, so a capped run computes exactly what an uncapped one
# does, only later. Shards start in order, heaviest first.
#
# Each shard writes its own log, printed whole once all have finished --
# interleaved output from several processes is unreadable -- failing shards
# first. Exits non-zero if any shard failed.
set -uo pipefail

n=${1:-5}
threads=${2:-3}
threads1=${3:-$threads}
par=${4:-$n}
logs=$(mktemp -d)
trap 'rm -rf "$logs"' EXIT

for i in $(seq 1 "$n"); do
    # A poll, not `wait -n`: that needs bash 4.3, and macOS still ships 3.2 for
    # anyone running the shards locally.
    while [ "$(jobs -rp | wc -l)" -ge "$par" ]; do sleep 1; done
    t=$threads; [ "$i" -eq 1 ] && t=$threads1
    # The exit status goes to a file, so it survives however the shard is reaped.
    ( NEREUS_TEST_SHARD="$i/$n" julia --project=. --threads="$t" test/runtests.jl \
        > "$logs/$i.log" 2>&1; echo $? > "$logs/$i.rc" ) &
done
wait

passed=() failed=()
for i in $(seq 1 "$n"); do
    if [ "$(cat "$logs/$i.rc" 2>/dev/null)" = 0 ]; then passed+=("$i"); else failed+=("$i"); fi
done

for i in "${failed[@]}" "${passed[@]}"; do
    echo
    echo "================================ shard $i/$n ================================"
    cat "$logs/$i.log"
done

echo
# The container's own high-water mark (cgroup v2), to size max_parallel from
# measurement rather than a guess.
peak=/sys/fs/cgroup/memory.peak
[ -r "$peak" ] && echo "memory peak: $(( $(cat "$peak") / 1048576 )) MiB with up to $par of $n shards at once"
echo "passed: ${passed[*]:-none}   failed: ${failed[*]:-none}"
[ ${#failed[@]} -eq 0 ]
