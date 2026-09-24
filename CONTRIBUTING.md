# Working on Nereus.jl

## Branches

- `develop` is where work lands. Branch off it, open a pull request into it.
- `main` is what releases are cut from. `develop` → `main` is the maintainer's
  call alone; nothing else merges into `main`.
- Branch names carry the Plane card: `feature/<CARD>-<slug>` or
  `fix/<CARD>-<slug>`. Work in a separate worktree, not in a checkout another
  session is using -- several sessions share this repository.

## CI

Every pull request into `develop` or `main` runs the full test suite on
Forgejo Actions (`.forgejo/workflows/ci.yml`) on two self-hosted runners at
once, one linux/arm64 (~6 min) and one linux/amd64 (~15 min), inside the
`ci/Dockerfile` image (Julia 1.11.9, the version `Manifest.toml` was resolved
with), as five parallel shards (`ci/run_tests.sh`). Both must pass: the
samplers are chaotic, so the same seed gives different chains on the two
architectures, and a fixed-seed test can pass on one and fail on the other.
The first run after the Manifest changes also precompiles the dependency stack.

Run the whole suite locally in one process, or as the CI shards:

    julia --project=. --threads=8 test/runtests.jl
    ci/run_tests.sh 5 3 6

`test/shards.jl` lists every test file as a unit with its measured cost, and
decides which shard runs it. **A new test file goes in that list** (a file
missing from it is an error). A file must bring its own `using` lines: in a
shard it may run in a process where no earlier file loaded anything. Sharded
runs print `unit time:` per file; keep the `seconds` roughly in line with them.

A new push to the pull request cancels the run in flight. To re-run a branch by
hand, dispatch the workflow from the Actions tab.

## Tests that pin recorded values

Some tests compare likelihoods against recorded values, to 1e-13 relative
(`PIN_RTOL` in test/astrometry/test_iad_multi_instrument.jl) -- not with
`===`: the same code differs by an ulp between macOS/aarch64 and the
Linux/x86_64 CI runner, and reordered arithmetic moves the last bits too. When
a change legitimately moves a value past that -- a constant, a convention --
re-pin it with a comment saying what moved it and how that was checked (for
example: putting the old constant back reproduces the old value). A pin that
moves without such a reason is a bug. Comparisons between two code paths in
the SAME process can and should stay exact.
