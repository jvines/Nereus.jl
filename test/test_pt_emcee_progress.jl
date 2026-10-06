# What sample_pt_emcee reports while it runs: the progress line and the
# progress log. A long fit is watched through these alone, so they must name
# the parameter that is holding it back, and they must not touch the chain.
using Test
using Nereus
using JSON3
using Printf
using Random
using Statistics

# A toy ensemble, (step, parameter, walker), where parameter 2 is deliberately
# slow: each walker drifts on a random walk from its own starting point, so
# the walkers disagree (high R-hat) and every draw repeats the last (low ESS).
# The other three are independent draws from one normal.
function _slow_cube(; n_step = 400, n_w = 16, rng = MersenneTwister(3))
    cube = randn(rng, n_step, 4, n_w)
    for w in 1:n_w
        x = 3.0 * randn(rng)
        for s in 1:n_step
            x += 0.05 * randn(rng)
            cube[s, 2, w] = x
        end
    end
    return cube
end

@testset "pt_emcee readout: worst parameter by name" begin
    names = ["P_k1", "K_k1", "secosw_k1", "sesinw_k1"]
    cube = _slow_cube()
    d = Nereus._pt_cube_diag(cube)
    @test d.worst_rhat == 2 && d.worst_tail == 2
    @test d.max_rhat == maximum(d.rhat) && d.max_rhat > 1.5
    @test d.min_tail == minimum(d.tail_ess)
    @test all(<(1.05), d.rhat[[1, 3, 4]])
    rs, es = Nereus._pt_diag_strings(d, names)
    @test endswith(rs, "(K_k1)") && endswith(es, "(K_k1)")
    @test startswith(rs, @sprintf("%.3f/%.3f", d.mean_rhat, d.max_rhat))

    # The same draws through the flat buffer the sampler keeps (each step's
    # walkers stacked), and only the science columns: the position returned is
    # within `science_cols`, so a nuisance column in front shifts nothing.
    n_step, _, n_w = size(cube)
    flat = Matrix{Float64}(undef, n_step * n_w, 5)
    for s in 1:n_step, w in 1:n_w
        flat[(s - 1) * n_w + w, 1] = randn()               # a nuisance
        flat[(s - 1) * n_w + w, 2:5] .= cube[s, :, w]
    end
    d2 = Nereus._pt_convergence(flat, n_step * n_w, n_w, [2, 3, 4, 5])
    @test d2.worst_rhat == 2 && d2.rhat ≈ d.rhat
    @test Nereus._pt_convergence(flat, 3 * n_w, n_w, [2, 3, 4, 5]) === nothing
end

@testset "pt_emcee readout: ladder and cold log-posterior" begin
    s = Nereus._pt_swap_summary([40.0, 9.0, 30.0], [100, 100, 100])
    @test s.rates == [0.4, 0.09, 0.3] && s.worst == 2
    @test s.mean ≈ 0.79 / 3 && s.min == 0.09
    @test Nereus._pt_swap_str(s) == "0.263/0.090(2-3)"
    @test Nereus._pt_swap_str(Nereus._pt_swap_summary(Float64[], Int[])) == "—"

    # Walkers move on log(x + s) for a log-uniform prior; the readout is the
    # :lp of the recorded draws, so that Jacobian comes back off.
    state = reshape([log(2.0), log(4.0), 0.5, 0.7], 1, 2, 2)   # (rung, walker, dim)
    logπ = [-1.0 -2.0]; logL = [10.0 20.0]
    med, mx = Nereus._pt_cold_lp(state, logπ, logL,
                                 Union{Nothing,Float64}[0.0, nothing])
    @test mx ≈ -2.0 - log(4.0) + 20.0
    @test med ≈ ((-1.0 - log(2.0) + 10.0) + mx) / 2
    @test Nereus._pt_lp_str(74390.4) == "74390" && Nereus._pt_lp_str(-152.34) == "-152.3"
    @test Nereus._pt_count_str(950) == "950" && Nereus._pt_count_str(1_234_567) == "1.2M"
end

@testset "pt_emcee readout: burn-in window in time order" begin
    # A ring that has wrapped: the diagnostic must see the rows oldest first,
    # since split R-hat compares the first half of each walker with the second.
    cube = _slow_cube(n_step = 40)
    n = size(cube, 1)
    ring = similar(cube)
    for s in 1:n, w in 1:size(cube, 3)
        ring[mod1(s + 13, n), :, w] .= cube[s, :, w]     # row s lands in slot s + 13
    end
    d_ring = Nereus._pt_ring_diag(ring, mod1(14, n), n)
    d_cube = Nereus._pt_cube_diag(cube)
    @test d_ring.rhat == d_cube.rhat && d_ring.tail_ess == d_cube.tail_ess
    # Not yet wrapped: only the filled rows, from the first.
    @test Nereus._pt_ring_diag(ring, 11, 10).rhat ==
          Nereus._pt_cube_diag(ring[1:10, :, :]).rhat
    @test Nereus._pt_ring_diag(ring, 4, 3) === nothing
end

# The RV target of test_pt_emcee_resume.jl: one planet with a circular Mo, so
# the burn-in re-cut (steps 40 and 80) moves the window the readout holds.
Random.seed!(7)
const _PG_N = 60
const _PG_T = sort!(400 .* rand(_PG_N))
const _PG_RV = 40.0 .* sin.(2π .* _PG_T ./ 4.23) .+ 1.5 .* randn(_PG_N)
_pg_target() = build_target(
    planets = (b = (P = LogUniformPrior(4.0, 4.5), K = LogUniformPrior(10.0, 90.0),
                    sesinw = UniformPrior(-1.0, 1.0), secosw = UniformPrior(-1.0, 1.0),
                    Mo = UniformPrior(0.0, 2pi)),),
    rv = (SIM = (data = (t = _PG_T, rv = _PG_RV, rv_err = fill(1.5, _PG_N)),
                 sigma = LogUniformPrior(0.5, 10.0)),))
const _PG_KW = (n_temps = 4, n_walkers = 20, n_burnin = 80, n_steps = 150, seed = 42,
                diag_every = 20, adapt_ladder = true, bridge_headline = false)
const _PG_SCI = ["P_k1", "K_k1", "sesinw_k1", "secosw_k1", "Mo_k1"]

# Runs quietly whatever `show_progress` is; returns the result and what the
# progress bar printed.
function _pg_run(; kw...)
    tg = _pg_target()
    out = tempname()
    r = open(out, "w") do io
        redirect_stdout(io) do
            sample_pt_emcee(tg, tg.data; _PG_KW..., kw...)
        end
    end
    return r, read(out, String)
end
_pg_lines(path) = [JSON3.read(ln) for ln in eachline(path)]

@testset "pt_emcee readout: the chain is the same with or without it" begin
    ref, printed = _pg_run(show_progress = false)
    @test isempty(printed)
    for (sp, log) in ((true, false), (false, true), (true, true))
        r, _ = _pg_run(show_progress = sp,
                       progress_log = log ? joinpath(mktempdir(), "p.jsonl") : nothing)
        @test r.chains.value.data == ref.chains.value.data
        @test r.betas == ref.betas && r.n_evals == ref.n_evals
        @test r.acceptance_within == ref.acceptance_within
        @test r.acceptance_swap == ref.acceptance_swap
        @test isequal(r.log_evidence, ref.log_evidence)
    end
end

@testset "pt_emcee progress log" begin
    path = joinpath(mktempdir(), "sub", "progress.jsonl")    # directory made for it
    r, printed = _pg_run(show_progress = true, progress_log = path)
    L = _pg_lines(path)
    # Checks every 20 steps through burn-in (the last at step 80) and after it
    # (100, 120, 140), then the final line.
    @test [l.step for l in L] == [20, 40, 60, 80, 100, 120, 140, 150]
    @test [l.phase for l in L] == [fill("burn", 4); fill("sample", 4)]
    @test [l.final for l in L] == [fill(false, 7); true]
    keys_ = [:time, :step, :n_steps, :n_burnin, :phase, :final, :elapsed_s, :eta_s,
             :steps_per_s, :n_evals, :acc, :beta1_acc, :swap_acc, :swap_mean,
             :swap_min, :swap_worst_pair, :lp_median, :lp_max, :diag_window,
             :rhat_mean, :rhat_max, :rhat_worst, :bulk_ess_mean, :tail_ess_min,
             :tail_ess_worst, :rhat, :bulk_ess, :tail_ess]
    @test all(collect(keys(l)) == keys_ for l in L)

    for l in L
        # Burn-in figures come from the rolling window, the rest from every
        # kept draw.
        @test l.diag_window == (l.phase == "burn" ? [max(1, l.step - 19), l.step] :
                                                   [81, l.step])
        @test sort!(String.(collect(keys(l.rhat)))) == sort(_PG_SCI)
        rh = Dict(String(k) => v for (k, v) in pairs(l.rhat))
        te = Dict(String(k) => v for (k, v) in pairs(l.tail_ess))
        # The named parameter holds the worst value (ties are common at
        # these lengths: the tail ESS bottoms out at the same floor).
        @test l.rhat_max == maximum(values(rh)) == rh[l.rhat_worst]
        @test l.tail_ess_min == minimum(values(te)) == te[l.tail_ess_worst]
        @test length(l.swap_acc) == 3 && l.swap_min == minimum(l.swap_acc)
        @test l.swap_acc[l.swap_worst_pair[1]] == l.swap_min
        @test l.swap_worst_pair[2] == l.swap_worst_pair[1] + 1
        @test l.lp_median <= l.lp_max
    end
    # The last line agrees with the result.
    @test L[end].n_evals == r.n_evals
    @test L[end].swap_acc ≈ r.acceptance_swap atol = 1e-4
    @test L[end].lp_max ≈ maximum(r.chains.value.data[end, end, :]) atol = 1e-3  # :lp

    # The progress line ends on the figures of the last check (step 140), each
    # naming its worst parameter, with the ladder and the cold log-posterior.
    last_line = split(strip(printed), '\n')[end]
    m = match(r"Rhat=(\d+\.\d{3})/(\d+\.\d{3})\((\w+)\)", last_line)
    @test m !== nothing && m[3] == L[7].rhat_worst
    @test abs(parse(Float64, m[1]) - L[7].rhat_mean) < 6e-4
    m = match(r"ESS=(\d+)/(\d+)\((\w+)\)", last_line)
    @test m !== nothing && m[3] == L[7].tail_ess_worst
    @test abs(parse(Int, m[2]) - L[7].tail_ess_min) <= 1
    @test occursin(r"swap=\d\.\d{3}/\d\.\d{3}\(\d-\d\)", last_line)
    @test occursin(r"lp=-?\d+\.\d/-?\d+\.\d", last_line)
    @test !occursin("(burn)", last_line)

    # Until the kept draws have figures of their own (none here before step
    # 100), the burn-in window's are shown, marked as such.
    path90 = joinpath(mktempdir(), "progress.jsonl")
    _, printed = _pg_run(show_progress = true, progress_log = path90, n_steps = 90)
    L90 = _pg_lines(path90)
    last_line = split(strip(printed), '\n')[end]
    @test L90[4].step == 80 && L90[4].phase == "burn"
    m = match(r"Rhat\(burn\)=(\d+\.\d{3})/(\d+\.\d{3})\((\w+)\)", last_line)
    @test m !== nothing && m[3] == L90[4].rhat_worst
    @test abs(parse(Float64, m[2]) - L90[4].rhat_max) < 6e-4
    m = match(r"ESS\(burn\)=(\d+)/(\d+)\((\w+)\)", last_line)
    @test m !== nothing && m[3] == L90[4].tail_ess_worst

    # A fresh run starts the file over.
    _pg_run(show_progress = false, progress_log = path, n_steps = 60)
    @test [l.step for l in _pg_lines(path)] == [20, 40, 60, 60]
end

@testset "pt_emcee progress log: resume" begin
    dir = mktempdir()
    ck, path = joinpath(dir, "state.jls"), joinpath(dir, "progress.jsonl")
    _pg_run(show_progress = false, progress_log = path, checkpoint = ck, n_steps = 90)
    # A run killed after its checkpoint leaves lines past it, and maybe half
    # of one; resuming drops both, so the steps only increase.
    open(path, "a") do io
        println(io, JSON3.write((; step = 100, final = false)))
        print(io, "{\"step\": 12")
    end
    r, _ = _pg_run(show_progress = false, progress_log = path, checkpoint = ck,
                   resume = true)
    L = _pg_lines(path)
    @test [l.step for l in L] == [20, 40, 60, 80, 90, 100, 120, 140, 150]
    @test [l.final for l in L] == [false, false, false, false, true, false, false, false, true]
    @test L[end].n_evals == r.n_evals
end

@testset "pt_emcee progress log: a failure to write never stops the run" begin
    ref, _ = _pg_run(show_progress = false)
    blocker = tempname(); write(blocker, "a file, not a directory")
    bad = joinpath(blocker, "progress.jsonl")
    local r
    @test_logs (:warn, r"cannot write the progress log") match_mode = :any begin
        r, _ = _pg_run(show_progress = false, progress_log = bad)
    end
    @test r.chains.value.data == ref.chains.value.data
    # Warned once, not at every check.
    logger = Test.TestLogger()
    Base.CoreLogging.with_logger(logger) do
        _pg_run(show_progress = false, progress_log = bad)
    end
    @test count(l -> occursin("progress log", l.message), logger.logs) == 1
end

@testset "run_job keeps the progress log in output_dir" begin
    tg = _pg_target()
    dispatch(kw; out = mktempdir()) = (Nereus._dispatch_sampler(
        Dict(:sampler => Dict(:name => "pt_emcee", :kwargs => kw), :output_dir => out),
        tg, tg.data, 1); out)
    small = Dict(:n_temps => 4, :n_walkers => 20, :n_steps => 60, :n_burnin => 30,
                 :diag_every => 20, :show_progress => false, :bridge_headline => false)
    out = dispatch(small)
    @test isfile(joinpath(out, "progress.jsonl"))
    @test _pg_lines(joinpath(out, "progress.jsonl"))[end].final
    # JSON null turns it off; a path from JSON arrives as a Symbol.
    out = dispatch(merge(small, Dict(:progress_log => nothing)))
    @test !isfile(joinpath(out, "progress.jsonl"))
    other = joinpath(mktempdir(), "elsewhere.jsonl")
    out = dispatch(merge(small, Dict(:progress_log => "$other")))
    @test isfile(other) && !isfile(joinpath(out, "progress.jsonl"))
end
