# Informed starts and period jumps for the fixed-k PT samplers
# (src/samplers/period_jump.jl).
#
#   1. Off (the default), the samplers draw exactly what they did.
#   2. The informed proposal's density is the density its draws come from.
#   3. The period jump leaves the posterior invariant: a chain made of the jump
#      and a local random walk, on a target with a strong daily alias, puts the
#      two alias modes in the ratio a direct 3-D quadrature gives, through both
#      the raw-parameter path (`sample_pt`) and the flat-scale ensemble sweep
#      (`pt_emcee`, `pt_whitening`).
#   4. With informed starts, a short pt_emcee run finds the weak second planet
#      that prior starts miss.
#   5. A run with the jump resumes bit for bit.
using Test
using Nereus
using Random
using Statistics
using Nereus: period_jump_plan, informed_init_plan, _period_jump_refresh!,
              _period_jump_propose!, _period_jump_sweep!, _pt_period_jump!,
              _informed_slot_log_q, _informed_draw!, TransDimPTState, PTWorkspace,
              Theta, log_prior, rv_log_likelihood, log_scale_shift

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

# One planet sampled about nightly: P = 1.9 d and its daily alias 1/(1 − 1/1.9)
# = 2.111 d fit almost equally well; the 0.008 d jitter in the epochs breaks the
# tie mildly, so both modes carry mass.
const _PJA_T, _PJA_RV = let rng = MersenneTwister(11)
    t = Float64.(0:39) .+ 0.008 .* randn(rng, 40)
    t, 6.0 .* cos.(2π .* t ./ 1.9 .+ 0.7) .+ 2.0 .* randn(rng, 40)
end
_pja_target() = build_target(;
    planets = (b = (P = LogUniformPrior(1.6, 2.6), K = LogUniformPrior(0.5, 30.0),
                    sesinw = 0.0, secosw = 0.0, Mo = UniformPrior(0.0, 2pi)),),
    rv = (SIM = (data = (t = _PJA_T, rv = _PJA_RV, rv_err = fill(2.0, 40)),
                 sigma = 0.0, gamma = 0.0),))

# Two planets: a strong one and a weak one well away from it.
const _PJ2_T, _PJ2_RV = let rng = MersenneTwister(3)
    t = sort!(300 .* rand(rng, 80))
    t, 30.0 .* sin.(2π .* t ./ 3.1) .+ 9.0 .* sin.(2π .* t ./ 17.3 .+ 1.0) .+
       1.5 .* randn(rng, 80)
end
_pj2_target() = build_target(;
    planets = (b = (P = LogUniformPrior(1.0, 300.0), K = LogUniformPrior(1.0, 60.0),
                    sesinw = UniformPrior(-0.7, 0.7), secosw = UniformPrior(-0.7, 0.7),
                    Mo = UniformPrior(0.0, 2pi)),
               c = (P = LogUniformPrior(1.0, 300.0), K = LogUniformPrior(1.0, 60.0),
                    sesinw = UniformPrior(-0.7, 0.7), secosw = UniformPrior(-0.7, 0.7),
                    Mo = UniformPrior(0.0, 2pi))),
    rv = (SIM = (data = (t = _PJ2_T, rv = _PJ2_RV, rv_err = fill(1.5, 80)),
                 sigma = LogUniformPrior(0.1, 10.0)),))

# ---------------------------------------------------------------------------
# 1. Defaults unchanged
# ---------------------------------------------------------------------------
@testset "period jump: off draws what it did" begin
    kw = (n_temps = 3, n_walkers = 16, n_burnin = 40, n_steps = 80, seed = 5,
          show_progress = false, bridge_headline = false)
    a = (tg = _pja_target(); sample_pt_emcee(tg, tg.data; kw...))
    b = (tg = _pja_target(); sample_pt_emcee(tg, tg.data; kw...,
                                             informed_init = false, period_jump_prob = 0.0))
    @test a.chains.value.data == b.chains.value.data
    @test a.n_evals == b.n_evals
    # On, the run differs (it is a different kernel), and stays reproducible.
    c1 = (tg = _pja_target(); sample_pt_emcee(tg, tg.data; kw..., period_jump_prob = 0.2))
    c2 = (tg = _pja_target(); sample_pt_emcee(tg, tg.data; kw..., period_jump_prob = 0.2))
    @test c1.chains.value.data == c2.chains.value.data
    @test c1.chains.value.data != a.chains.value.data

    kwp = (n_rounds = 5, n_chains = 3, seed = 2, show_report = false)
    pa = sample_pt(_pja_target(); kwp...)
    pb = sample_pt(_pja_target(); kwp..., informed_init = false, period_jump_prob = 0.0)
    @test pa[1].value.data == pb[1].value.data

    @test_throws ArgumentError (tg = _pja_target();
        sample_pt_emcee(tg, tg.data; kw..., period_jump_prob = 1.5))
    @test_throws ArgumentError (tg = _pja_target();
        sample_pt_emcee(tg, tg.data; kw..., informed_init = true,
                        init_strategy = :map_scatter))
end

# ---------------------------------------------------------------------------
# 2. The density is the draws' density
# ---------------------------------------------------------------------------
# For draws u ~ q (a draw outside the prior is a rejection, scored 0), the mean
# of p₀(u)/q(u) is ∫ p₀ over what q reaches. With p₀ a normalised density well
# inside the support (a Gaussian in (log P, log K, Mo) about the 2.11 d peak's
# own centre, a little narrower than its kernels, so the weights stay bounded),
# that is 1: a density that disagreed with the sampler would miss it.
@testset "period jump: q matches its draws" begin
    tg = _pja_target(); p = tg.params; L = p.layout
    plan = period_jump_plan(p, tg.data)
    @test plan.planets == [1] && length(plan.slots[1]) == 3
    th = Theta{Float64}(p)
    iP, iK, iM = (L.name_to_idx[n] for n in ("P_k1", "K_k1", "Mo_k1"))
    th.values[iP] = 1.9; th.values[iK] = 6.0; th.values[iM] = 1.0
    _period_jump_refresh!(plan, th, tg.data)
    S = plan.setups[1]
    @test S !== nothing && length(S.w) >= 2
    blk = L.planet_blocks[1]
    c = argmin(abs.(S.log_x .- log(1 / (1 - 1 / 1.9))))
    @test abs(exp(S.log_x[c]) - 2.111) < 0.02
    μ = (S.log_x[c], log(S.K_ls[c]), S.Mo_ls[c]); σ = (0.03, 0.2, 0.25)
    wrapd(d) = mod(d + π, 2π) - π
    logp0(v) = sum(-0.5 * (d / s)^2 - log(s * sqrt(2π)) for (d, s) in
                   zip((log(v[iP]) - μ[1], log(v[iK]) - μ[2], wrapd(v[iM] - μ[3])), σ)) -
               log(v[iP]) - log(v[iK])
    rng = MersenneTwister(1); n = 400_000; acc = 0.0
    sc = Theta{Float64}(p)
    for _ in 1:n
        sc.values .= th.values
        _informed_draw!(sc.values, sc, blk, S, tg.data, rng, false;
                        slots = plan.slots[1]) === nothing && continue
        lq = _informed_slot_log_q(sc, blk, sc.values, S, tg.data, false;
                                  slots = plan.slots[1], strict_wrap = true)
        acc += exp(logp0(sc.values) - lq)
    end
    @test isapprox(acc / n, 1.0; atol = 0.05)
end

# The informed draw puts the planet in phase with the data under a :Mo anchor
# with a free eccentricity vector: the LS phase is that of M + ω, so Mo is
# placed at Mo_LS − ω (it used to ignore ω, a random phase for a newborn whose
# ω is drawn near e = 0). Draws at the peak period beat the flat model.
@testset "informed draw: phase consistent with ω" begin
    tg = build_target(;
        planets = (b = (P = LogUniformPrior(1.6, 2.6), K = LogUniformPrior(0.5, 30.0),
                        sesinw = UniformPrior(-0.7, 0.7), secosw = UniformPrior(-0.7, 0.7),
                        Mo = UniformPrior(0.0, 2pi)),),
        rv = (SIM = (data = (t = _PJA_T, rv = _PJA_RV, rv_err = fill(2.0, 40)),
                     sigma = 0.0, gamma = 0.0),))
    p = tg.params; L = p.layout; blk = L.planet_blocks[1]
    ip = informed_init_plan(p, tg.data)
    S = ip.setups[1]; c = argmax(S.w)
    iP, iK, iM = (L.name_to_idx[n] for n in ("P_k1", "K_k1", "Mo_k1"))
    ll_at(Mo, ω) = (th = Theta{Float64}(p);
        th.values[iP] = exp(S.log_x[c]); th.values[iK] = S.K_ls[c];
        th.values[blk.e1] = 0.1 * sin(ω); th.values[blk.e2] = 0.1 * cos(ω);
        th.values[iM] = mod(Mo, 2π); (th, rv_log_likelihood(th, tg.data)))
    for ω in (-2.0, 1.2, 2.8)
        th, _ = ll_at(0.0, ω)
        kind, μ, σ, per = Nereus._informed_slot_kernel(th, blk, blk.t, S, c,
                              exp(S.log_x[c]), tg.data; ω = Nereus._informed_omega(th, blk, th.values))
        @test μ ≈ S.Mo_ls[c] - ω
        _, ll_new = ll_at(μ, ω)              # the kernel's centre
        _, ll_old = ll_at(S.Mo_ls[c], ω)     # the centre that ignored ω
        @test ll_new > ll_old + 10
        # and the density is centred there too
        thc, _ = ll_at(μ, ω)
        tho, _ = ll_at(S.Mo_ls[c], ω)
        lq(t) = _informed_slot_log_q(t, blk, t.values, S, tg.data, false; slots = [blk.P, blk.t])
        @test lq(thc) > lq(tho)
    end
end

# ---------------------------------------------------------------------------
# 3. Invariance: the alias modes in the right ratio
# ---------------------------------------------------------------------------
# Reference: the posterior mass near each alias by quadrature over (P, K, Mo).
function _pja_reference(tg)
    p = tg.params; L = p.layout
    iP, iK, iM = (L.name_to_idx[n] for n in ("P_k1", "K_k1", "Mo_k1"))
    th = Theta{Float64}(p)
    function mass(Pc)
        Ps = range(Pc * 0.97, Pc * 1.03; length = 72)
        Ks = range(1.5, 11.0; length = 64)
        Ms = range(0.0, 2π; length = 257)[1:256]
        tot = 0.0; lmax = -Inf
        lv = Array{Float64}(undef, length(Ps), length(Ks), length(Ms))
        for (a, P) in enumerate(Ps), (b, K) in enumerate(Ks), (c, M) in enumerate(Ms)
            th.values[iP] = P; th.values[iK] = K; th.values[iM] = M
            lv[a, b, c] = log_prior(th) + rv_log_likelihood(th, tg.data)
        end
        lmax = maximum(lv)
        return lmax + log(sum(exp.(lv .- lmax)) * step(Ps) * step(Ks) * step(Ms))
    end
    lA, lB = mass(1.9), mass(1/(1 - 1/1.9))
    return 1 / (1 + exp(lB - lA))     # mass fraction of the 1.9 d mode
end

# Mode fraction of a chain, with a batch-means standard error.
function _pja_frac(Ps)
    inA = Float64.(abs.(Ps ./ 1.9 .- 1) .< 0.03)
    inB = Float64.(abs.(Ps ./ (1/(1 - 1/1.9)) .- 1) .< 0.03)
    @test mean(inA .+ inB) > 0.99            # nothing outside the two modes
    nb = 40; bl = length(inA) ÷ nb
    fr = [mean(inA[(i-1)*bl+1:i*bl]) for i in 1:nb]
    return mean(inA), std(fr) / sqrt(nb)
end

@testset "period jump: alias modes in the posterior ratio" begin
    tg = _pja_target(); p = tg.params; L = p.layout; data = tg.data
    fA_ref = _pja_reference(tg)
    @test 0.1 < fA_ref < 0.9                 # both modes carry mass
    plan = period_jump_plan(p, data)
    iP, iK, iM = (L.name_to_idx[n] for n in ("P_k1", "K_k1", "Mo_k1"))
    th0 = Theta{Float64}(p)
    th0.values[iP] = 1.9; th0.values[iK] = 6.0; th0.values[iM] = 1.0
    _period_jump_refresh!(plan, th0, data)     # one fixed proposal
    n_iter = 150_000

    # (a) Raw parameters, `sample_pt`'s path: the jump, then a random walk that
    # stays in its mode (steps far below the 0.2 d gap).
    @testset "raw parameters (sample_pt)" begin
        rng = MersenneTwister(21)
        th = Theta{Float64}(p, copy(th0.values))
        rep = TransDimPTState(th, log_prior(th), rv_log_likelihood(th, data))
        sc = Theta{Float64}(p, copy(th0.values)); prop = Theta{Float64}(p, copy(th0.values))
        ws = PTWorkspace(p, p.config.max_kplanet, length(p.config.noise_models);
                         n_obs = length(data.t_rv), n_phot = 0)
        pr = zeros(Int, 1); ac = zeros(Int, 1)
        Ps = Vector{Float64}(undef, n_iter)
        for it in 1:n_iter
            _pt_period_jump!(rep, sc, plan, 1.0, data, 1.0, rng, nothing, ws, pr, ac, 1)
            prop.values .= rep.theta.values
            prop.values[iP] += 0.003 * randn(rng)
            prop.values[iK] += 0.4 * randn(rng)
            prop.values[iM] = mod(prop.values[iM] + 0.1 * randn(rng), 2π)
            lp = log_prior(prop)
            if isfinite(lp)
                ll = rv_log_likelihood(prop, data)
                if log(rand(rng)) < ll - rep.log_L + lp - rep.log_pi
                    rep.theta.values .= prop.values; rep.log_pi = lp; rep.log_L = ll
                end
            end
            Ps[it] = rep.theta.values[iP]
        end
        fA, se = _pja_frac(Ps)
        @test ac[1] > 100                    # the jump does cross
        @test abs(fA - fA_ref) < max(4se, 0.03)
    end

    # (b) The ensemble sweep on the flat scale (`pt_emcee`), one walker at one
    # rung: log π carries Σ log(x + s) over the log-scale coordinates.
    @testset "flat scale (pt_emcee sweep)" begin
        rng = MersenneTwister(22)
        uf = L.unfrozen_idx; nd = length(uf)
        shift = Union{Nothing,Float64}[log_scale_shift(ps) for ps in L.unfrozen_priors]
        th = Theta{Float64}(p)
        function evaly(y)
            jac = 0.0
            for (j, idx) in enumerate(uf)
                s = shift[j]
                th.values[idx] = s === nothing ? y[j] : exp(y[j]) - s
                s === nothing || (jac += y[j])
            end
            lp = log_prior(th) + jac
            isfinite(lp) || return (-Inf, -Inf)
            return (lp, rv_log_likelihood(th, data))
        end
        y0 = [shift[j] === nothing ? th0.values[idx] : log(th0.values[idx] + shift[j])
              for (j, idx) in enumerate(uf)]
        state = reshape(copy(y0), 1, 1, nd)
        lp0, ll0 = evaly(y0)
        logπ = fill(lp0, 1, 1); logL = fill(ll0, 1, 1)
        rngs = [MersenneTwister(23)]; bufs = [zeros(nd)]; ths = [Theta{Float64}(p)]
        pr = [Threads.Atomic{Int}(0)]; ac = [Threads.Atomic{Int}(0)]
        dP, dK, dM = (findfirst(==(i), uf) for i in (iP, iK, iM))
        Ps = Vector{Float64}(undef, n_iter); yb = zeros(nd)
        ev(buf, t, w, slot) = evaly(buf)
        for it in 1:n_iter
            _period_jump_sweep!(ev, state, logπ, logL, [1.0], plan, data, 1.0, rngs,
                                bufs, ths, uf, shift, pr, ac)
            yb .= @view state[1, 1, :]
            yb[dP] += 0.0015 * randn(rng)            # log-scale steps
            yb[dK] += 0.07 * randn(rng)
            yb[dM] = mod(yb[dM] + 0.1 * randn(rng), 2π)
            lp, ll = evaly(yb)
            if isfinite(lp) && log(rand(rng)) < ll - logL[1] + lp - logπ[1]
                state[1, 1, :] .= yb; logπ[1] = lp; logL[1] = ll
            end
            y = state[1, 1, dP]
            Ps[it] = shift[dP] === nothing ? y : exp(y) - shift[dP]
        end
        fA, se = _pja_frac(Ps)
        @test ac[1][] > 100
        @test abs(fA - fA_ref) < max(4se, 0.03)
    end
end

# ---------------------------------------------------------------------------
# 4. Informed starts find the weak planet
# ---------------------------------------------------------------------------
@testset "informed_init: finds the second planet" begin
    tg = _pj2_target()
    ip = informed_init_plan(tg.params, tg.data)
    @test length(ip.planets) == 2 && all(S -> S !== nothing, ip.setups)
    # Each planet's setup peaks at its own signal: the other's is subtracted.
    tops = sort([exp(S.log_x[argmax(S.w)]) for S in ip.setups])
    @test isapprox(tops[1], 3.1; rtol = 0.02) && isapprox(tops[2], 17.3; rtol = 0.02)

    # Informed starts spread the walkers over the peaks and their aliases, so
    # an ensemble starts multimodal and its stretch / DE moves, which need a
    # partner in the same mode, stall; the period jump is what lets walkers
    # move between those starts. With both, the weak planet is found; prior
    # starts miss it, and the jump alone finds it in fewer walkers.
    kw = (n_temps = 4, n_walkers = 30, n_burnin = 150, n_steps = 250, seed = 4,
          show_progress = false, bridge_headline = false)
    function near(ch, P0)
        P1 = vec(Array(ch[:P_k1])); P2 = vec(Array(ch[:P_k2]))
        return mean((abs.(P1 ./ P0 .- 1) .< 0.02) .| (abs.(P2 ./ P0 .- 1) .< 0.02))
    end
    rp = (tg = _pj2_target(); sample_pt_emcee(tg, tg.data; kw...))
    rj = (tg = _pj2_target(); sample_pt_emcee(tg, tg.data; kw..., period_jump_prob = 0.1))
    ri = (tg = _pj2_target(); sample_pt_emcee(tg, tg.data; kw..., informed_init = true,
                                              period_jump_prob = 0.1))
    @test near(rp.chains, 17.3) < 0.2       # prior starts miss it
    @test near(ri.chains, 17.3) > 0.8       # informed starts + jump find it
    @test near(ri.chains, 3.1) > 0.95
    @test near(ri.chains, 17.3) > near(rj.chains, 17.3) + 0.2
    @test maximum(Array(ri.chains[:lp])) > maximum(Array(rp.chains[:lp])) + 50

    # pt_whitening and pt take the options and run.
    rw = (tg = _pj2_target(); sample_pt_whitening(tg, tg.data; n_temps = 3,
          n_walkers = 30, n_burnin = 100, n_steps = 150, seed = 4, show_progress = false,
          informed_init = true, period_jump_prob = 0.1))
    @test near(rw.chains, 3.1) > 0.9
    rq = sample_pt(_pj2_target(); n_rounds = 7, n_chains = 4, seed = 4,
                   show_report = false, informed_init = true, period_jump_prob = 0.1)
    @test size(rq[1], 1) > 0
    @test_throws ArgumentError sample_pt(_pj2_target(); n_rounds = 3,
        td = TransDimConfig(max_kplanet = 2), period_jump_prob = 0.1)
end

# ---------------------------------------------------------------------------
# 5. Resume
# ---------------------------------------------------------------------------
@testset "period jump: resume is bit-identical" begin
    kw = (n_temps = 3, n_walkers = 16, n_burnin = 60, seed = 9, show_progress = false,
          bridge_headline = false, informed_init = true, period_jump_prob = 0.2)
    run(n; k...) = (tg = _pja_target(); sample_pt_emcee(tg, tg.data; kw..., n_steps = n, k...))
    full = run(140)
    path = joinpath(mktempdir(), "pt_emcee_state.jls")
    run(55; checkpoint = path)              # inside burn-in, before a refresh
    ext = run(140; checkpoint = path, resume = true)
    @test ext.chains.value.data == full.chains.value.data
    @test ext.n_evals == full.n_evals
    # A checkpoint without the jump is refused for a run with it.
    path2 = joinpath(mktempdir(), "pt_emcee_state.jls")
    run(55; checkpoint = path2, period_jump_prob = 0.0)
    @test_throws ArgumentError run(140; checkpoint = path2, resume = true)

    # sample_pt
    kwp = (n_chains = 3, seed = 3, show_report = false, informed_init = true,
           period_jump_prob = 0.2, n_warmup_rounds = 3)
    pfull = sample_pt(_pja_target(); kwp..., n_rounds = 7)
    ppath = joinpath(mktempdir(), "pt_state.jls")
    sample_pt(_pja_target(); kwp..., n_rounds = 5, checkpoint = ppath)
    pext = sample_pt(_pja_target(); kwp..., n_rounds = 7, checkpoint = ppath, resume = true)
    @test pext[1].value.data == pfull[1].value.data
end
