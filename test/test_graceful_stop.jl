# Ending a PT ensemble run early without losing its draws: an interrupt, a stop
# file, a wall-clock limit, the convergence gate and `Stopping`
# (src/graceful_stop.jl). The step hook delivers each of them at an exact step
# instead of racing the sampler with a timer.
using Test, Nereus, Random, Statistics, MCMCChains, JSON3
include(joinpath(@__DIR__, "fixtures", "easy_rv_target.jl"))

Base.exit_on_sigint(false)       # a SIGINT is an InterruptException, as in the daemon

const _GS_BURNIN = 60
const _GS_STEPS = 400

function _gs_target()
    Random.seed!(7)
    n = 60; t = sort!(400 .* rand(n))
    rv = 40.0 .* sin.(2π .* t ./ 4.23) .+ 1.5 .* randn(n)
    build_target(
        planets = (b = (P = LogUniformPrior(4.0, 4.5), K = LogUniformPrior(10.0, 90.0),
                        sesinw = UniformPrior(-1.0, 1.0), secosw = UniformPrior(-1.0, 1.0),
                        Mo = UniformPrior(0.0, 2pi)),),
        rv = (SIM = (data = (t = t, rv = rv, rv_err = fill(1.5, n)),
                     sigma = LogUniformPrior(0.5, 10.0)),))
end

_gs_run(target; kw...) = sample_pt_emcee(target, target.data; n_temps = 4,
    n_walkers = 20, n_steps = _GS_STEPS, n_burnin = _GS_BURNIN, seed = 42,
    show_progress = false, kw...)

function _gs_td()
    Random.seed!(11)
    t_rv = sort!(200 .* rand(40))
    data = Data(; t_rv = t_rv, rv = 30.0 .* sin.(2π .* t_rv ./ 7.3) .+ randn(40),
                rv_err = ones(length(t_rv)))
    params = Params(max_kplanet = 2, planet_modes = [RV_ONLY, RV_ONLY],
                    instruments = InstrumentConfig(rv = ["HARPS"]), data = data, M_s = 1.0)
    return NereusTarget(params, data; unconstrained = false), data
end

function _gs_td_run(; kw...)
    tg, data = _gs_td()
    return sample_transdim_pt_emcee(tg, data; td = TransDimConfig(max_kplanet = 2),
        n_temps = 4, n_walkers = 16, n_steps = 300, n_burnin = 60, seed = 7,
        show_progress = false, kw...)
end

function _gs_with_hook(f, hook)
    Nereus._STEP_HOOK[] = hook
    try
        return f()
    finally
        Nereus._STEP_HOOK[] = nothing
    end
end

_gs_interrupt(at) = (step, sc) -> step == at && ccall(:kill, Cint, (Cint, Cint), ccall(:getpid, Cint, ()), Base.SIGINT)

# The chains hold exactly the rows of the steps the result says were kept.
function _gs_consistent(r, n_walkers_min = 1)
    s = r.stop
    d = r.chains.value.data
    nrow = size(d, 1) * size(d, 3)
    @test nrow == s.n_draws
    @test all(isfinite, d)
    @test s.n_draws > 0
    @test nrow % n_walkers_min == 0
end

function _gs_capture(f)
    res = nothing
    out = mktemp() do path, io
        redirect_stdout(io) do
            res = f()
        end
        flush(io); read(path, String)
    end
    return res, out
end

@testset "graceful stop" begin
    target = _gs_target()

    @testset "no stop requested: unchanged" begin
        a = _gs_run(target)
        b = _gs_run(target; stop_file = joinpath(mktempdir(), "STOP"))   # never appears
        @test a.stop.stopped == "steps" && a.stop.step == _GS_STEPS
        @test b.stop.stopped == "steps"
        @test a.chains.value.data == b.chains.value.data       # bit for bit
        @test a.log_evidence == b.log_evidence
        @test a.stop.n_draws == 20 * (_GS_STEPS - _GS_BURNIN)
    end

    @testset "an interrupt after burn-in keeps the draws so far" begin
        at = _GS_BURNIN + 40
        r = _gs_with_hook(() -> _gs_run(target), _gs_interrupt(at))
        @test r.stop.stopped == "interrupted"
        # The signal is delivered asynchronously: on a loaded machine it can land
        # a step or two after the hook sends it. What must hold whatever the
        # step is that the step in flight is dropped (the draw counts below).
        @test at - 1 <= r.stop.step <= at + 5
        @test r.stop.n_draws == 20 * (r.stop.step - _GS_BURNIN)
        _gs_consistent(r, 20)
        @test size(r.chains.value.data, 1) == r.stop.step - _GS_BURNIN
        @test isfinite(r.log_evidence)
        @test size(r.ladder.betas, 1) == r.stop.step
    end

    @testset "an interrupt during burn-in aborts" begin
        @test_throws InterruptException _gs_with_hook(() -> _gs_run(target),
                                                      _gs_interrupt(_GS_BURNIN ÷ 2))
        # ... and the next run is unaffected.
        @test _gs_run(target).stop.stopped == "steps"
    end

    @testset "the announcement is one line on stdout" begin
        at = _GS_BURNIN + 25
        r, out = _gs_capture(() -> _gs_with_hook(() -> _gs_run(target), _gs_interrupt(at)))
        m = match(r"stopping early: interrupted at step (\d+)/(\d+) \((\d+) draws kept\)", out)
        @test m !== nothing
        @test parse(Int, m[1]) == r.stop.step && parse(Int, m[2]) == _GS_STEPS
        @test parse(Int, m[3]) == r.stop.n_draws
    end

    @testset "max_seconds" begin
        # The clock jumps past the limit at a step: the run ends there.
        at = _GS_BURNIN + 10
        hook = (step, sc) -> step == at && (sc.t0 = 0.0)
        r = _gs_with_hook(() -> _gs_run(target; max_seconds = 3600.0), hook)
        @test r.stop.stopped == "max_seconds" && r.stop.step == at
        _gs_consistent(r, 20)
        @test r.stop.n_draws == 20 * 10
        # Before the end of burn-in there is nothing to keep.
        early = (step, sc) -> step == 5 && (sc.t0 = 0.0)
        err = try _gs_with_hook(() -> _gs_run(target; max_seconds = 3600.0), early)
              catch e; e end
        @test err isa ErrorException
        @test occursin("burn-in", err.msg) && occursin("max_seconds", err.msg)
        @test_throws ArgumentError _gs_run(target; max_seconds = -1.0)
    end

    @testset "max_evals" begin
        r = _gs_run(target; max_evals = 4 * 20 * 120)
        @test r.stop.stopped == "max_evals"
        @test _GS_BURNIN < r.stop.step < _GS_STEPS
        _gs_consistent(r, 20)
    end

    @testset "stop file" begin
        sf = joinpath(mktempdir(), "STOP")
        at = _GS_BURNIN + 20
        # Created from another task while the run is going; the diagnostic
        # cadence (diag_every) forces the look, so the 5 s poll does not wait.
        hook = (step, sc) -> step == at &&
            fetch(Threads.@spawn write(sf, "stop"))
        r, out = _gs_capture(() -> _gs_with_hook(
            () -> _gs_run(target; stop_file = sf, diag_every = 10), hook))
        @test r.stop.stopped == "requested" && r.stop.step == at
        @test r.stop.stop_file == sf
        _gs_consistent(r, 20)
        @test isfile(sf)                                   # never deleted by the sampler
        @test occursin("stopping early: stop requested at step $at/$_GS_STEPS " *
                       "($(r.stop.n_draws) draws kept)", out)
        # During burn-in: an error, no result.
        rm(sf)
        early = (step, sc) -> step == 8 && (write(sf, "stop"); sc.next_poll = 0.0)
        err = try _gs_with_hook(() -> _gs_run(target; stop_file = sf, diag_every = 10), early)
              catch e; e end
        @test err isa ErrorException
        @test occursin("burn-in", err.msg) && occursin("stop", err.msg)
        # A file that is already there before the run starts counts the same.
        @test_throws ErrorException _gs_run(target; stop_file = sf)
    end

    @testset "polling touches the file system rarely" begin
        sc = Nereus.StopControl(; stop_file = joinpath(mktempdir(), "STOP"))
        @test Nereus._stop_check!(sc, 0) === nothing        # polls, nothing there
        write(sc.stop_file, "x")
        @test Nereus._stop_check!(sc, 0) === nothing        # inside the 5 s window
        @test Nereus._stop_check!(sc, 0; force = true) == "requested"
        sc.next_poll = 0.0
        @test Nereus._stop_check!(sc, 0) == "requested"
    end

    @testset "transdim: interrupt, stop file and the convergence gate" begin
        at = 60 + 30
        r = _gs_with_hook(() -> _gs_td_run(), _gs_interrupt(at))
        @test r.stop.stopped == "interrupted" && at - 1 <= r.stop.step <= at + 5   # async signal
        d = r.chains.value.data
        # (the ensemble has max(n_walkers, 2 n_dim + 2) walkers)
        @test size(d, 1) == r.stop.n_draws
        @test r.stop.n_draws % (r.stop.step - 60) == 0 && r.stop.n_draws ÷ (r.stop.step - 60) >= 16
        @test all(isfinite, d)
        @test_throws InterruptException _gs_with_hook(() -> _gs_td_run(), _gs_interrupt(30))

        sf = joinpath(mktempdir(), "STOP")
        hook = (step, sc) -> step == 100 && write(sf, "stop")
        r = _gs_with_hook(() -> _gs_td_run(stop_file = sf, diag_every = 10), hook)
        @test r.stop.stopped == "requested" && r.stop.step == 100

        # The gate: an easy single-planet problem clears a loose threshold long
        # before the 1500 steps it was given.
        g = _gs_td_run(n_steps = 1500, convergence_stop = true, rhat_threshold = 3.0,
                       tail_ess_threshold = 10, diag_every = 40, n_converged_checks = 2)
        @test g.stop.stopped == "converged"
        @test 60 < g.stop.step < 1500
        @test size(g.chains.value.data, 1) == g.stop.n_draws
        @test g.stop.n_draws % (g.stop.step - 60) == 0
        # Off by default: the same run goes the whole way.
        @test _gs_td_run(n_steps = 200).stop.stopped == "steps"
    end

    @testset "the transdim gating set" begin
        tg, data = _gs_td()
        cols = vcat(Symbol.(tg.params.layout.unfrozen_names), [:n_planets],
                    [:planet_active_1, :planet_active_2])
        n_w, n_s = 16, 60
        rng = MersenneTwister(3)
        # Slot 1 active throughout, slot 2 never; every walker draws the same law.
        X = randn(rng, n_w * n_s, length(cols))
        X[:, findfirst(==(:n_planets), cols)] .= 1.0
        X[:, findfirst(==(:planet_active_1), cols)] .= 1.0
        X[:, findfirst(==(:planet_active_2), cols)] .= 0.0
        gate = Symbol[:n_planets; [c for c in cols if occursin(r"_k\d+$", String(c))]]
        d = Nereus._td_convergence(X, size(X, 1), n_w, cols, gate, tg.params)
        @test d !== nothing
        meanr, maxr, meanb, mte = d
        @test maxr < 1.1
        # One walker stuck at an offset is a convergence failure the gate sees.
        Y = copy(X)
        for (j, c) in enumerate(cols)
            occursin(r"_k1$", String(c)) || continue
            Y[1:2:end, j] .+= 50.0
        end
        @test Nereus._td_convergence(Y, size(Y, 1), n_w, cols, gate, tg.params)[2] > 1.5
    end

    @testset "Stopping" begin
        rng = MersenneTwister(5)
        t = sort!(200 .* rand(rng, 40))
        rv = 15 .* sin.(2π .* t ./ 12.3) .+ randn(rng, 40)
        data = Dict("SIM" => (t = t, rv = rv, rv_err = fill(1.0, 40)))
        pri = Dict("P_k1" => UniformPrior(12.2, 12.4), "K_k1" => UniformPrior(10.0, 20.0))
        eng(; kw...) = Dict("engine" => "pt_emcee", "options" => Dict{String,Any}(
            "n_temps" => 3, "n_walkers" => 24, "n_steps" => 1500, "n_burnin" => 100,
            "show_progress" => false, [string(k) => v for (k, v) in kw]...))

        @testset "rhat_max and ess_min stop a converging fit early" begin
            out = mktempdir()
            r = fit_rv(data; planets = 1, priors = pri, science = false, output_dir = out,
                       engine = eng(), plots = String[],
                       stopping = Stopping(rhat_max = 3.0, ess_min = 10.0, check_every = 40))
            @test r.summary["stopped"] == "converged"
            @test r.summary["stopped_at_step"] >= 100 + 40
            @test 100 < r.summary["stopped_at_step"] < 1500
            @test r.summary["n_steps"] == 1500
            @test r.summary["n_draws_kept"] == 24 * (r.summary["stopped_at_step"] - 100)
            disk = JSON3.read(read(joinpath(out, "summary.json"), String); allow_inf = true)
            @test disk["stopped"] == "converged"
            @test disk["stop_file"] == joinpath(out, "STOP")     # the default, recorded
        end

        @testset "max_seconds ends a run gracefully" begin
            hook = (step, sc) -> step == 140 && (sc.t0 = 0.0)
            r = _gs_with_hook(() -> fit_rv(data; planets = 1, priors = pri, science = false,
                    engine = eng(), plots = String[],
                    stopping = Stopping(max_seconds = 3600.0)), hook)
            @test r.summary["stopped"] == "max_seconds" && r.summary["stopped_at_step"] == 140
        end

        @testset "a stop file given to fit_rv" begin
            sf = joinpath(mktempdir(), "me")
            hook = (step, sc) -> step == 130 && write(sf, "stop")
            r = _gs_with_hook(() -> fit_rv(data; planets = 1, priors = pri, science = false,
                    engine = eng(diag_every = 10), plots = String[], stop_file = sf), hook)
            @test r.summary["stopped"] == "requested" && r.summary["stop_file"] == sf
            @test r.summary["stopped_at_step"] == 130
        end

        @testset "what an engine cannot honour is an error" begin
            e = try fit_rv(data; planets = 1, priors = pri, science = false,
                           engine = eng(), stopping = Stopping(logz_tol = 0.1))
                catch err; err end
            @test e isa ArgumentError
            @test occursin("logz_tol", e.msg) && occursin("nested", e.msg)
            e = try fit_rv(data; planets = 1, priors = pri, science = false,
                           engine = Dict("engine" => "nested"),
                           stopping = Stopping(max_seconds = 5.0))
                catch err; err end
            @test e isa ArgumentError
            @test occursin("max_seconds", e.msg) && occursin("pt_emcee", e.msg)
            # a conflict with the engine's own option is an error too
            e = try Nereus._stopping_options("pt_emcee", Stopping(rhat_max = 1.05),
                                             Dict("rhat_threshold" => 1.01))
                catch err; err end
            @test e isa ArgumentError && occursin("conflicts", e.msg)
            e = try Nereus._stopping_options("pt_emcee", Stopping(ess_min = 100.0),
                                             Dict("convergence_stop" => false))
                catch err; err end
            @test e isa ArgumentError && occursin("convergence_stop", e.msg)
            # agreeing values are fine
            o = Nereus._stopping_options("pt_emcee", Stopping(rhat_max = 1.05),
                                         Dict("rhat_threshold" => 1.05))
            @test o[:rhat_threshold] == 1.05 && o[:convergence_stop] == true
        end

        @testset "the mapping" begin
            m(name, st) = Nereus._stopping_options(name, st, Dict())
            @test m("nested", Stopping(logz_tol = 0.3))[:dlogz] == 0.3
            @test m("nested_dynamic", Stopping(logz_tol = 0.3))[:dlogz_batch] == 0.3
            o = m("transdim_pt_emcee", Stopping(max_seconds = 9.0, max_evals = 1000,
                                                 check_every = 25, ess_min = 300.0))
            @test o[:max_seconds] == 9.0 && o[:max_evals] == 1000 && o[:diag_every] == 25
            @test o[:tail_ess_threshold] == 300.0 && o[:convergence_stop]
            @test isempty(m("pt_emcee", nothing)) && isempty(m("pt_emcee", Stopping()))
        end
    end

    @testset "run_job: stop file and summary.json" begin
        cfg = _easy_rv_cfg()
        cfg["sampler"]["kwargs"] = Dict("show_progress" => false, "n_temps" => 2,
                                        "n_walkers" => 32, "n_steps" => 600,
                                        "n_burnin" => 100, "diag_every" => 10)
        sf = joinpath(cfg["output_dir"], "STOP")        # the default location
        hook = (step, sc) -> step == 170 && write(sf, "stop")
        res = _gs_with_hook(() -> Nereus.run_job(cfg), hook)
        s = res isa AbstractDict ? res : res.summary
        @test s["status"] == "ok" && s["stopped"] == "requested"
        @test s["stopped_at_step"] == 170 && s["n_steps"] == 600
        @test s["n_draws_kept"] == 32 * 70
        @test s["stop_file"] == sf
        disk = JSON3.read(read(joinpath(cfg["output_dir"], "summary.json"), String);
                          allow_inf = true)
        @test disk["stopped"] == "requested" && disk["stopped_at_step"] == 170
        @test disk["stop_file"] == sf
        @test isfile(joinpath(cfg["output_dir"], "chains.nc"))
        @test isfile(sf)

        # A run that was not stopped says so too.
        cfg2 = _easy_rv_cfg()
        cfg2["sampler"]["kwargs"] = Dict("show_progress" => false, "n_temps" => 2,
                                         "n_walkers" => 32, "n_steps" => 150,
                                         "n_burnin" => 100)
        s2 = (r = Nereus.run_job(cfg2); r isa AbstractDict ? r : r.summary)
        @test s2["stopped"] == "steps" && s2["stopped_at_step"] == 150

        # An explicit stop_file replaces the default.
        cfg3 = _easy_rv_cfg()
        cfg3["sampler"]["kwargs"] = cfg2["sampler"]["kwargs"]
        cfg3["stop_file"] = joinpath(mktempdir(), "elsewhere")
        s3 = (r = Nereus.run_job(cfg3); r isa AbstractDict ? r : r.summary)
        @test s3["stop_file"] == cfg3["stop_file"]
    end
end
