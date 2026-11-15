# The extended burn-in (`extend_burnin`, src/samplers/burnin_extend.jl).
#
# A burn-in too short for the stranded-walker pruning ends with the β = 1 rung
# split between modes, and every draw of the walkers' later migration lands in
# the posterior. With `extend_burnin = true` burn-in runs on, in pruning passes,
# while the rung is split; off (the default) nothing changes.
#
# The split is set up, not waited for: one planet fitted to two sinusoids, P1 =
# 4.23 d at 40 m/s and P2 = 4.40 d at 10 m/s, with the jitter held under 3 m/s
# so neither residual is absorbed: P2 is a narrow local mode of the one-planet
# posterior, down ~1900 in ln p from P1. A run is stopped after its first step
# and its checkpoint rewritten with every walker in P2 but one β = 1 walker in
# P1 (draws from runs on priors around each); the resumed runs then reach
# n_burnin with that split, the pruning's first pass having only recorded it.
# Two rungs at β = 1 and 0.5, both in P2: no state of P1 can leave the cold
# rung, nor can a hotter rung feed it climbing states, so the split holds still
# -- a rung still climbing is one the pruning leaves alone.
using Test
using Nereus
using Random
using Serialization
using Statistics

Random.seed!(20261115)
const _BE_N = 60
const _BE_T = sort!(400 .* rand(_BE_N))
const _BE_RV = 40.0 .* sin.(2π .* _BE_T ./ 4.23) .+
               10.0 .* sin.(2π .* _BE_T ./ 4.40 .+ 1.0) .+ 1.5 .* randn(_BE_N)
const _BE_P1, _BE_P2 = 4.23, 4.40

_be_target(Plo = 4.0, Phi = 4.6) = build_target(
    planets = (b = (P = LogUniformPrior(Plo, Phi), K = LogUniformPrior(1.0, 90.0),
                    sesinw = UniformPrior(-1.0, 1.0), secosw = UniformPrior(-1.0, 1.0),
                    Mo = UniformPrior(0.0, 2pi)),),
    rv = (SIM = (data = (t = _BE_T, rv = _BE_RV, rv_err = fill(1.5, _BE_N)),
                 sigma = LogUniformPrior(0.5, 3.0)),))

# Stretch moves only: the DE move's mode jumps (γ = 1) carry walkers between
# two modes this close in a 7-parameter model within tens of steps.
const _BE_KW = (n_temps = 2, betas = [1.0, 0.5], n_walkers = 32, n_burnin = 100,
                seed = 7, de_move = 0, show_progress = false, bridge_headline = false)

# (step, parameter, walker); `Array(chains)` would stack the walkers.
_cube(r) = r.chains.value.data
_col(r, p) = vec(r.chains.value.data[:, findfirst(==(p), names(r.chains)), :])

# Posterior draws (x, rows) of the one-planet model with P confined to one mode.
function _be_mode_draws(Plo, Phi)
    tg = _be_target(Plo, Phi)
    r = sample_pt_emcee(tg, tg.data; n_temps = 4, n_walkers = 32, n_steps = 400,
                        n_burnin = 300, seed = 3, show_progress = false,
                        bridge_headline = false)
    nms = Symbol.(tg.params.layout.unfrozen_names)
    X = hcat([vec(Array(r.chains[p])) for p in nms]...)
    iM = findfirst(==(:Mo_k1), nms)
    X[:, iM] .= mod2pi.(X[:, iM])          # into the window of a fresh target
    return X
end

# A checkpoint at step 1 of the wide-prior run, every walker put in P2 but β = 1
# walker 1, in P1: positions on the flat scale the walkers move on, and their
# log π (with the flat scale's Jacobian) and log L as the sampler caches them.
const _BE_X = (_be_mode_draws(4.15, 4.30), _be_mode_draws(4.35, 4.45))

# The positions every injection puts walkers at: one P1 draw for β = 1 walker 1,
# P2 draws for the rest.
function _be_position(t, w, rng)
    X1, X2 = _BE_X
    return (t, w) == (1, 1) ? X1[end, :] : X2[rand(rng, 1:size(X2, 1)), :]
end

function _be_split_checkpoint(path)
    tg = _be_target()
    sample_pt_emcee(tg, tg.data; _BE_KW..., n_steps = 1, checkpoint = path)
    ck = open(deserialize, path)
    st = ck.state
    S, LP, LL = copy(st.state), copy(st.logπ), copy(st.logL)
    L = tg.params.layout
    shift = [Nereus.log_scale_shift(p) for p in L.unfrozen_priors]
    bounded = Nereus.NereusTarget(tg.params, tg.data; unconstrained = false)
    rng = MersenneTwister(5)
    for t in axes(S, 1), w in axes(S, 2)
        x = _be_position(t, w, rng)
        lp, ll = Nereus._logdensity_parts(bounded, x)
        for j in eachindex(x)
            s = shift[j]
            S[t, w, j] = s === nothing ? x[j] : log(x[j] + s)
            s === nothing || (lp += S[t, w, j])
        end
        LP[t, w], LL[t, w] = lp, ll
    end
    @assert all(isfinite, LP) && all(isfinite, LL)
    open(io -> serialize(io, merge(ck, (state = merge(st, (state = S, logπ = LP,
                                                           logL = LL)),))), path, "w")
    return path
end

# A copy of the split checkpoint, resumed to `n_steps`: each resume rewrites it.
function _be_resume(src, n_steps; kw...)
    path = joinpath(mktempdir(), "pt_emcee_state.jls")
    cp(src, path)
    tg = _be_target()
    return sample_pt_emcee(tg, tg.data; _BE_KW..., n_steps, checkpoint = path,
                           resume = true, kw...), path
end

_atP1(r) = abs.(_col(r, :P_k1) .- _BE_P1) .< 0.01

@testset "extended burn-in (pt_emcee)" begin
    split = _be_split_checkpoint(joinpath(mktempdir(), "split.jls"))

    @testset "a split rung at n_burnin: burn-in runs on until it is in one mode" begin
        on, _ = _be_resume(split, 700; extend_burnin = true)
        off, _ = _be_resume(split, 700)
        # Off, the default: recording starts at n_burnin + 1 with the rung split.
        @test off.n_burnin_used == 100
        @test size(_cube(off), 1) == 600
        @test mean(_atP1(off)) < 0.5
        # On: several passes, every recorded walker in P1, the extension's steps
        # taken out of the recorded ones.
        @test on.n_burnin_used > 100
        @test (on.n_burnin_used - 100) % 50 == 0
        @test size(_cube(on), 1) == 700 - on.n_burnin_used
        @test all(_atP1(on))
        @test on.ladder.n_burnin == on.n_burnin_used
        @test on.stop.n_draws == size(_cube(on), 1) * 32
    end

    @testset "resumed during the extension, it carries on bit for bit" begin
        # One cap for both: the default grows with n_steps.
        kw = (extend_burnin = true, max_burnin = 600)
        full, _ = _be_resume(split, 700; kw...)
        @test full.n_burnin_used >= 200
        # Ends inside the extension, recording nothing, and says so.
        _, mid = @test_logs (:warn, r"ended at step 160 inside its extended burn-in") match_mode = :any _be_resume(split, 160; kw...)
        tg = _be_target()
        ext = sample_pt_emcee(tg, tg.data; _BE_KW..., n_steps = 700, checkpoint = mid,
                              resume = true, kw...)
        @test ext.n_burnin_used == full.n_burnin_used
        @test _cube(ext) == _cube(full)
        @test ext.n_evals == full.n_evals
    end

    @testset "still split at the cap: records anyway and says what to do" begin
        # max_burnin = n_burnin: the check runs, nothing extends, it warns.
        r = @test_logs (:warn, r"still split between modes at step 100.*P_k1") match_mode = :any begin
            first(_be_resume(split, 400; extend_burnin = true, max_burnin = 100))
        end
        @test r.n_burnin_used == 100
        @test mean(_atP1(r)) < 0.5
        # One pass, then the cap.
        r2 = @test_logs (:warn, r"still split between modes at step 150") match_mode = :any begin
            first(_be_resume(split, 400; extend_burnin = true, max_burnin = 150))
        end
        @test r2.n_burnin_used == 150
        @test_throws ArgumentError _be_resume(split, 400; extend_burnin = true, max_burnin = 50)
    end

    @testset "the check at β = 1, then the other rungs, is one full pass" begin
        # The state at step 100 of the split run without the option: the cold
        # rung split, the record taken at step 50.
        _, path = _be_resume(split, 100)
        ck = open(deserialize, path).state
        tg = _be_target()
        Nereus._restore_windows!(tg.params, ck.windows, ())
        shift = [Nereus.log_scale_shift(p) for p in tg.params.layout.unfrozen_priors]
        bounded = Nereus.NereusTarget(tg.params, tg.data; unconstrained = false)
        evalf = function (y)
            x = [s === nothing ? y[j] : exp(y[j]) - s for (j, s) in enumerate(shift)]
            lp, ll = Nereus._logdensity_parts(bounded, x)
            isfinite(lp) || return (-Inf, -Inf)
            return (lp + sum((y[j] for j in eachindex(y) if shift[j] !== nothing);
                             init = 0.0), ll)
        end
        circ = Nereus.circular_indices(tg.params)
        mem = Nereus.PruneMemory(size(ck.state)...)
        Nereus._prune_restore!(mem, ck)
        βs = [1.0, 0.5]
        pass! = (s, lp, ll, r, m, rg) -> Nereus._pt_prune_stranded!(s, lp, ll, βs, r;
                                            evalf, circ, memory = m, rungs = rg)
        A = (copy(ck.state), copy(ck.logπ), copy(ck.logL))
        B = map(copy, A)
        rA, rB = MersenneTwister(9), MersenneTwister(9)
        mA, mB = deepcopy(mem), deepcopy(mem)
        full = pass!(A..., rA, mA, 1:2)
        two = Nereus._burnin_split_pass!(pass!, B, rB, mB, 2)
        @test full[1] > 0
        @test two == full
        @test isequal(A, B)
        @test rA == rB
        @test all(isequal(getfield(mA, k), getfield(mB, k)) for k in fieldnames(typeof(mA)))
        # Without `commit` (at the cap) only the count: nothing taken.
        C = (copy(ck.state), copy(ck.logπ), copy(ck.logL))
        rC, mC, C0 = MersenneTwister(9), deepcopy(mem), map(copy, C)
        n1 = Nereus._burnin_split_pass!(pass!, C, rC, mC, 2; commit = false)
        @test n1[1] == full[1]
        @test isequal(C, C0) && rC == MersenneTwister(9)
        @test all(isequal(getfield(mC, k), getfield(mem, k)) for k in fieldnames(typeof(mem)))
    end

    @testset "no split at n_burnin: the run draws exactly what it did" begin
        # One mode in the prior: nothing stranded at n_burnin.
        go(; kw...) = (tg = _be_target(4.15, 4.30);
                       sample_pt_emcee(tg, tg.data; _BE_KW..., n_steps = 300, kw...))
        dflt, off, on = go(), go(extend_burnin = false), go(extend_burnin = true)
        @test on.n_burnin_used == off.n_burnin_used == dflt.n_burnin_used == 100
        @test _cube(on) == _cube(off) == _cube(dflt)
        @test on.n_evals == off.n_evals == dflt.n_evals
        @test on.betas == off.betas
        @test on.acceptance_swap == off.acceptance_swap
        @test isequal(on.log_evidence, off.log_evidence)
        @test on.ladder.betas == off.ladder.betas
    end
end

# ------------------------------------------------------------------
# The siblings: sample_transdim_pt_emcee and sample_ensemble, the same split
# put in the same way.
# ------------------------------------------------------------------
using Nereus: TransDimConfig, reset_transdim_caches!

const _BE_TD_KW = (td = TransDimConfig(max_kplanet = 1), n_temps = 2, betas = [1.0, 0.5],
                   n_walkers = 32, n_burnin = 100, seed = 7, de_move = 0,
                   show_progress = false)

# The trans-dim checkpoint: every walker in the one-planet model, positions in
# x, and log π that of a one-planet walker of the run but for the change of the
# prior between their positions (the model's own share is the same constant).
function _be_td_split_checkpoint(path)
    reset_transdim_caches!()
    tg = _be_target()
    sample_transdim_pt_emcee(tg, tg.data; _BE_TD_KW..., n_steps = 1, checkpoint = path)
    ck = open(deserialize, path)
    st = ck.state
    S, LP, LL, TDS = copy(st.state), copy(st.logπ), copy(st.logL), deepcopy(st.td_states)
    bounded = Nereus.NereusTarget(tg.params, tg.data; unconstrained = false)
    ref = findfirst(tds -> tds.planet_active[1], TDS)
    @assert ref !== nothing
    c = LP[ref] - first(Nereus._logdensity_parts(bounded, S[ref[1], ref[2], :]))
    rng = MersenneTwister(5)
    for t in axes(S, 1), w in axes(S, 2)
        x = _be_position(t, w, rng)
        lp, ll = Nereus._logdensity_parts(bounded, x)
        S[t, w, :] .= x
        LP[t, w], LL[t, w] = lp + c, ll
        TDS[t, w].n_planets_active = 1
        TDS[t, w].planet_active[1] = true
    end
    @assert all(isfinite, LP) && all(isfinite, LL)
    open(io -> serialize(io, merge(ck, (state = merge(st, (state = S, logπ = LP, logL = LL,
                                                           td_states = TDS)),))), path, "w")
    return path
end

function _be_td_resume(src, n_steps; kw...)
    path = joinpath(mktempdir(), "td_state.jls")
    cp(src, path)
    reset_transdim_caches!()
    tg = _be_target()
    return sample_transdim_pt_emcee(tg, tg.data; _BE_TD_KW..., n_steps, checkpoint = path,
                                    resume = true, kw...)
end

const _BE_ENS_KW = (n_walkers = 32, n_burnin = 100, seed = 7, de_move = 0,
                    show_progress = false)

# The ensemble's checkpoint (one rung): positions in its sampling coordinates and
# their log-densities as `logpost` gives them. A run that ends before its burn-in
# has no draws to return and throws, after writing the checkpoint.
function _be_ens_split_checkpoint(path)
    tg = _be_target()
    @test_throws ArgumentError sample_ensemble(tg; _BE_ENS_KW..., n_steps = 2,
                                               checkpoint = path)
    return _be_ens_inject!(path, tg)
end

function _be_ens_inject!(path, tg)
    ck = open(deserialize, path)
    st = ck.state
    x, ll = copy(st.x), copy(st.lastll)
    rng = MersenneTwister(5)
    for w in axes(x, 2)
        xb = _be_position(1, w, rng)
        y = tg.transform === nothing ? xb : Nereus.transform_forward(xb, tg.transform)
        x[:, w] .= y
        ll[w] = Nereus.LogDensityProblems.logdensity(tg, y)
    end
    @assert all(isfinite, ll)
    open(io -> serialize(io, merge(ck, (state = merge(st, (x = x, lastll = ll)),))),
         path, "w")
    return path
end

function _be_ens_resume(src, n_steps; kw...)
    path = joinpath(mktempdir(), "ens_state.jls")
    cp(src, path)
    tg = _be_target()
    return sample_ensemble(tg; _BE_ENS_KW..., n_steps, checkpoint = path, resume = true,
                           kw...)
end

# Two chains, the split put in chain 1's checkpoint alone. Each chain throws at
# step 2 after writing its own, the second perhaps after the first has thrown.
function _be_ens2_resume(n_steps; kw...)
    path = joinpath(mktempdir(), "ens_state.jls")
    p1, p2 = (Nereus._chain_checkpoint_path(path, c) for c in 1:2)
    @test_throws Exception sample_ensemble(_be_target(); _BE_ENS_KW..., n_chains = 2,
                                           n_steps = 2, checkpoint = path)
    @test timedwait(() -> isfile(p1) && isfile(p2), 60.0) === :ok
    _be_ens_inject!(p1, _be_target())
    r = sample_ensemble(_be_target(); _BE_ENS_KW..., n_chains = 2, n_steps,
                        checkpoint = path, resume = true, kw...)
    return r, [first(open(deserialize, p).state.burnin_ext) for p in (p1, p2)]
end

@testset "extended burn-in (transdim_pt_emcee, ensemble)" begin
    @testset "transdim_pt_emcee" begin
        split = _be_td_split_checkpoint(joinpath(mktempdir(), "split.jls"))
        on = _be_td_resume(split, 700; extend_burnin = true)
        off = _be_td_resume(split, 700)
        @test off.n_burnin_used == 100
        @test mean(_atP1(off)) < 0.5
        @test on.n_burnin_used > 100
        @test size(_cube(on), 1) == (700 - on.n_burnin_used) * 32   # walkers in rows
        @test all(_col(on, :n_planets) .== 1)
        @test all(_atP1(on))
        @test on.ladder.n_burnin == on.n_burnin_used
        # At the cap it warns.
        r = @test_logs (:warn, r"td-pt_emcee: the β = 1 rung is still split.*P_k1") match_mode = :any begin
            _be_td_resume(split, 400; extend_burnin = true, max_burnin = 100)
        end
        @test r.n_burnin_used == 100
        # No split at n_burnin: the same draws with the option on as off.
        go(; kw...) = (reset_transdim_caches!(); tg = _be_target(4.15, 4.30);
                       sample_transdim_pt_emcee(tg, tg.data; _BE_TD_KW..., n_steps = 250, kw...))
        a, b = go(), go(extend_burnin = true)
        @test b.n_burnin_used == a.n_burnin_used == 100
        @test _cube(a) == _cube(b)
        @test a.n_evals == b.n_evals
    end

    @testset "ensemble" begin
        split = _be_ens_split_checkpoint(joinpath(mktempdir(), "split.jls"))
        on = _be_ens_resume(split, 700; extend_burnin = true)
        off = _be_ens_resume(split, 700)
        Pon, Poff = vec(Array(on[:P_k1])), vec(Array(off[:P_k1]))
        @test off.info.n_burnin_used == 100
        @test mean(abs.(Poff .- _BE_P1) .< 0.01) < 0.5
        @test on.info.n_burnin_used > 100
        @test length(Pon) == (700 - on.info.n_burnin_used) * 32
        @test all(abs.(Pon .- _BE_P1) .< 0.01)
        # At the cap it warns.
        rc = @test_logs (:warn, r"ensemble: the β = 1 rung is still split.*P_k1") match_mode = :any begin
            _be_ens_resume(split, 400; extend_burnin = true, max_burnin = 100)
        end
        @test rc.info.n_burnin_used == 100
        # The summary reports it, from the chain as run_job and fit_* see it.
        @test Nereus._burnin_to_summary!(Dict{String,Any}(), on)["n_burnin_used"] ==
              on.info.n_burnin_used
        @test Nereus._burnin_to_summary!(Dict{String,Any}(), (chains = on,))["n_burnin_used"] ==
              on.info.n_burnin_used
        # No split: the same draws. One rung has no hotter walkers to feed it, and
        # at n_burnin = 100 the pruning's one pass (step 50) only records: a
        # walker stranded then is still out at 100, and the check moves it. At
        # 200 the passes at 50, 100 and 150 have cleaned the rung.
        go(; kw...) = (tg = _be_target(4.15, 4.30);
                       sample_ensemble(tg; _BE_ENS_KW..., n_burnin = 200, n_steps = 300,
                                       kw...))
        a, b = go(), go(extend_burnin = true)
        @test Array(a) == Array(b)
        @test b.info.n_burnin_used == a.info.n_burnin_used == 200
        # Two chains extend apart; each keeps the last draws, as many as the one
        # that ran longest recorded, every one of chain 1's in P1.
        r2, ends = _be_ens2_resume(700; extend_burnin = true)
        @test ends[1] > 100
        @test r2.info.n_burnin_used == maximum(ends)
        @test size(r2, 3) == 2
        @test size(r2, 1) == (700 - r2.info.n_burnin_used) * 32
        @test all(abs.(r2.value.data[:, findfirst(==(:P_k1), names(r2)), 1] .- _BE_P1) .< 0.01)
    end
end

# run_job passes `extend_burnin` and `max_burnin` through from sampler.kwargs (a
# JSON `null` arriving as `nothing`), and summary.json says where recording began.
@testset "extended burn-in through run_job" begin
    include(joinpath(@__DIR__, "fixtures", "easy_rv_target.jl"))
    cfg = _easy_rv_cfg()
    cfg["sampler"]["kwargs"] = Dict("show_progress" => false, "n_temps" => 4,
        "n_walkers" => 20, "n_steps" => 300, "n_burnin" => 100,
        "extend_burnin" => true, "max_burnin" => nothing)
    res = Nereus.run_job(cfg)
    s = res isa AbstractDict ? res : res.summary
    @test s["n_burnin_used"] >= 100
    @test s["n_draws_kept"] == (300 - s["n_burnin_used"]) * 20
end
