# ci/run_tests.sh is also how the shards are run locally, and macOS still ships
# bash 3.2. Under `set -u` that bash calls an empty array unbound, so a run in
# which every shard passed (no failed shard) died on `"${failed[@]}"`, exiting
# 1 with none of the shard logs printed. The script is run here against a stub
# `julia`, once with every shard passing and once with every shard failing,
# under each bash it may meet.
using Test

@testset "ci/run_tests.sh reports passing and failing runs" begin
    root = dirname(@__DIR__)
    script = joinpath(root, "ci", "run_tests.sh")
    shells = unique(filter(b -> b !== nothing && isfile(b),
                           [Sys.isunix() ? "/bin/bash" : nothing, Sys.which("bash")]))
    isempty(shells) && @test_skip "no bash"
    for bash in shells, (code, passes, verdict) in (
            (0, true, "passed: 1 2   failed: none"),
            (1, false, "passed: none   failed: 1 2"))
        mktempdir() do dir
            stub = joinpath(dir, "julia")
            write(stub, "#!/bin/sh\necho \"stub shard \$NEREUS_TEST_SHARD\"\nexit $code\n")
            chmod(stub, 0o755)
            cmd = addenv(Cmd(`$bash $script 2 1 1 2`; dir = root),
                         "PATH" => dir * ":" * get(ENV, "PATH", ""))
            out = IOBuffer()
            p = run(pipeline(ignorestatus(cmd); stdout = out, stderr = out))
            log = String(take!(out))
            @test success(p) == passes
            @test occursin(verdict, log)
            @test occursin("stub shard 1/2", log) && occursin("stub shard 2/2", log)
        end
    end
end
