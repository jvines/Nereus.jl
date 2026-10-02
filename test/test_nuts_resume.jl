# sample_nuts checkpoints and resume (src/checkpoint.jl).
#
# The contract: a run stopped at any iteration and continued with `resume =
# true` is bit-identical to one that ran straight through, so a run that has not
# mixed can simply be extended. With n_warmup = 200 Stan's schedule is: init
# buffer 1-75 (step size only), metric windows 76-100 and 101-150 (Welford
# accumulators, the metric updated and both adaptors reset at 100 and 150), term
# buffer 151-200, and the step size finalised at 200. The stops below cross all
# of it -- 40, 90, 100, 150, 160, 200 -- plus the first and a later sampling
# iteration (201, 230) and a finished run extended past its own end. Chains stop
# at different iterations in one run, as a kill leaves them; the test hook
# `_NUTS_STOP_AFTER` is what stops them, since no `n_samples` ends a run inside
# the warm-up. The last testset kills a real run, in a child process.
using Test
using Nereus
using Random
using Logging
using Serialization

include(joinpath(@__DIR__, "fixtures", "nuts_resume_target.jl"))

# A run killed while one of its chains is still queued for a thread. Chain tasks
# never yield, so with more chains than threads a chain can wait until another
# has finished; every checkpoint written meanwhile must still hold it, or a kill
# in that time leaves none. One default thread and two chains make that wait
# certain: the child runs them one after the other, writes a checkpoint after
# every iteration, and SIGKILLs itself as soon as the first is on disk. Started
# here so that its start-up and compilation overlap the tests below; the last
# testset checks what it left.
const _NR_QUEUED_N = 20_000         # samples per chain: the kill comes long before
const _NR_QUEUED = let dir = mktempdir()
    path = joinpath(dir, "nuts_state.jls")
    script = joinpath(dir, "queued.jl")
    write(script, """
        using Nereus, Random, Logging
        include($(repr(joinpath(@__DIR__, "fixtures", "nuts_resume_target.jl"))))
        # From the interactive thread: the chains hold the one default thread.
        Threads.@spawn :interactive begin
            while !isfile($(repr(path)))
                sleep(0.001)
            end
            ccall(:kill, Cint, (Cint, Cint), getpid(), 9)
        end
        with_logger(ConsoleLogger(stderr, Logging.Error)) do
            sample_nuts(_nr_target(); $(_NR_KW)..., n_chains = 2, warm_start = false,
                        n_samples = $(_NR_QUEUED_N), rng = MersenneTwister(3),
                        checkpoint = $(repr(path)), checkpoint_interval = 0.0)
        end
        """)
    log = joinpath(dir, "child.log")
    cmd = `$(Base.julia_cmd()) --startup-file=no --threads=1,1
           --project=$(dirname(@__DIR__)) $script`
    (; path, log, proc = run(pipeline(cmd; stdout = devnull, stderr = log); wait = false))
end

# `rng` is the generator passed in, a fresh MersenneTwister(3) for every run
# unless given. A resumed run gets a fresh one too, which must come out as the
# uninterrupted run left its own.
function _nr_run(n_samples; stop = Int[], rng = MersenneTwister(3), rv = _NR_RV, kw...)
    tg = _nr_target(rv)
    Nereus._NUTS_STOP_AFTER[] = stop
    try
        # Errors only: the small warm-start pre-search warns about its evidence.
        ch = with_logger(ConsoleLogger(stderr, Logging.Error)) do
            sample_nuts(tg; _NR_KW..., n_samples, rng, kw...)
        end
        return (; ch, rng, windows = Nereus.circular_windows(tg.params))
    finally
        Nereus._NUTS_STOP_AFTER[] = Int[]
    end
end

const _NR_INFO = (:n_divergent, :step_size, :mean_tree_depth, :max_tree_depth, :mean_accept)
_nr_same(a, b) = a.ch.value.data == b.ch.value.data &&
                 all(f -> isequal(getproperty(a.ch.info, f), getproperty(b.ch.info, f)), _NR_INFO) &&
                 a.rng == b.rng && a.windows == b.windows

# The error a call throws, or `nothing`.
_nr_err(f) = try f(); nothing catch e; e end

@testset "sample_nuts resume" begin
    full = _nr_run(150)
    # The warm start moved a circular window, so a resume has one to put back.
    @test full.windows != Nereus.circular_windows(_nr_target().params)

    path = joinpath(mktempdir(), "nuts_state.jls")
    # Killed in the init buffer, a metric window, at a window split and at the
    # last warm-up iteration; resumed, and killed again in the term buffer, at
    # the second split, at the first sampling iteration and later in sampling.
    @test_throws Nereus._NutsStopped _nr_run(150; checkpoint = path, stop = [40, 90, 100, 200])
    @test isfile(path) && !isfile(path * ".tmp")
    @test_throws Nereus._NutsStopped _nr_run(150; checkpoint = path, resume = true,
                                             stop = [160, 150, 201, 230])
    mid = _nr_run(100; checkpoint = path, resume = true)   # a finished run ...
    ext = _nr_run(150; checkpoint = path, resume = true)   # ... extended

    # Identical to the uninterrupted run, bit for bit: draws, every diagnostic
    # in chains.info, the caller's rng afterwards and the windows.
    @test ext.ch.value.data == full.ch.value.data
    for f in _NR_INFO
        @test isequal(getproperty(ext.ch.info, f), getproperty(full.ch.info, f))
    end
    @test isequal(nuts_diagnostics(ext.ch), nuts_diagnostics(full.ch))
    @test ext.rng == full.rng
    @test ext.windows == full.windows
    # The finished 100-sample run kept the first 100 draws of each chain.
    @test mid.ch.value.data == full.ch.value.data[1:100, :, :]

    # Resuming at the checkpoint's own iteration runs nothing and returns the same.
    @test _nr_same(_nr_run(150; checkpoint = path, resume = true), full)

    # Refusals: no checkpoint, a different run, a total shorter than the saved.
    @test_throws ArgumentError _nr_run(150; resume = true)
    @test_throws ArgumentError _nr_run(150; checkpoint = path * ".none", resume = true)
    err = _nr_err(() -> _nr_run(200; checkpoint = path, resume = true, target_accept = 0.9))
    @test err isa ArgumentError && occursin("target_accept", err.msg)
    err = _nr_err(() -> _nr_run(200; checkpoint = path, resume = true, n_warmup = 300))
    @test err isa ArgumentError && occursin("n_warmup", err.msg)
    err = _nr_err(() -> _nr_run(200; checkpoint = path, resume = true, rv = _NR_RV .+ 1.0))
    @test err isa ArgumentError && occursin("data", err.msg)
    @test_throws ArgumentError _nr_run(100; checkpoint = path, resume = true)
    # A generator that did not start where the run's did is another run: another
    # seed, or another kind of generator.
    for rng in (MersenneTwister(4), Xoshiro(3))
        err = _nr_err(() -> _nr_run(200; checkpoint = path, resume = true, rng))
        @test err isa ArgumentError && occursin("rng", err.msg)
    end
end

@testset "sample_nuts resume, one chain on the caller's rng" begin
    # A single chain runs on `rng` itself, which the resume restores in place.
    full = _nr_run(60; n_chains = 1)
    path = joinpath(mktempdir(), "nuts_state.jls")
    @test_throws Nereus._NutsStopped _nr_run(60; n_chains = 1, checkpoint = path, stop = [120])
    @test _nr_same(_nr_run(60; n_chains = 1, checkpoint = path, resume = true), full)
end

@testset "sample_nuts seed: honoured, and part of the checkpoint's fingerprint" begin
    # run_job passes the job's `seed`. sample_nuts used to swallow it, so its
    # NUTS runs were unseeded. Short runs, without the warm start: the RNG is
    # what is under test.
    kw = (rng = nothing, n_chains = 2, warm_start = false)
    path = joinpath(mktempdir(), "nuts_state.jls")
    a = _nr_run(30; kw..., seed = 7, checkpoint = path)
    @test _nr_same(_nr_run(30; kw..., seed = 7), a)
    @test _nr_run(30; kw..., seed = 8).ch.value.data != a.ch.value.data
    err = _nr_err(() -> _nr_run(30; seed = 7))           # with the helper's `rng`
    @test err isa ArgumentError && occursin("not both", err.msg)

    # A resume with another seed is refused; the same seed continues the run.
    err = _nr_err(() -> _nr_run(60; kw..., seed = 8, checkpoint = path, resume = true))
    @test err isa ArgumentError && occursin("seed", err.msg)
    @test _nr_same(_nr_run(60; kw..., seed = 7, checkpoint = path, resume = true),
                   _nr_run(60; kw..., seed = 7))

    # An unseeded run (the default task-local stream) can still be continued.
    upath = joinpath(mktempdir(), "nuts_state.jls")
    u = _nr_run(30; kw..., checkpoint = upath)
    @test _nr_run(40; kw..., checkpoint = upath, resume = true).ch.value.data[1:30, :, :] ==
          u.ch.value.data

    # Through run_job's dispatch: the job's seed reaches the sampler.
    function job(seed)
        tg = _nr_target()
        kwj = Dict{String,Any}("n_samples" => 30, "n_warmup" => 200, "n_chains" => 2,
                               "warm_start" => false, "progress" => false)
        cfg = Dict{String,Any}("sampler" => Dict{String,Any}("name" => "nuts", "kwargs" => kwj),
                               "output_dir" => mktempdir())
        res = with_logger(ConsoleLogger(stderr, Logging.Error)) do
            Nereus._dispatch_sampler(cfg, tg, tg.data, seed)
        end
        return res.chains.value.data
    end
    @test job(7) == job(7)
    @test job(7) != job(8)
end

@testset "sample_nuts checkpoint while a chain waits for a thread" begin
    q = _NR_QUEUED
    wait(q.proc)
    killed = q.proc.termsignal == 9
    killed || @info "queued-chain child process" exitcode = q.proc.exitcode log = read(q.log, String)
    @test killed
    @test isfile(q.path)
    chains = open(deserialize, q.path).state.chains
    started = [haskey(s, :z) for s in chains]
    steps = [s.step for s in chains]
    # The chain that ran had saved mid-run; the one queued behind it is in the
    # checkpoint as its start point. Before the fix the first checkpoint came
    # only once the queued chain had started, with the other one finished.
    @test count(started) == 1
    @test 0 < maximum(steps) < _NR_KW.n_warmup + _NR_QUEUED_N
    @test steps[.!started] == [0]
    # It continues bit for bit, the queued chain starting as it would have.
    n = max(10, maximum(steps) - _NR_KW.n_warmup)
    kw = (n_chains = 2, warm_start = false)
    @test _nr_same(_nr_run(n; kw..., checkpoint = q.path, resume = true), _nr_run(n; kw...))
end
