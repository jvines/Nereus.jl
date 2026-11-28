# The noise-swap move exchanged one exclusion-group member for another while the
# parameters every state shares (jitters, offsets, always-on noise terms) kept
# the values tuned to the outgoing member, so a swap landed far below the
# incumbent: on HD 18599 (four-member RV menu) the median cold-chain Δln L of a
# swap was -2800 to -85000 for every pair, ~80 swaps in 10^6 proposals were
# accepted, and the member probabilities depended on the seed. State proposals
# (src/transdim/proposals/state_proposals.jl) propose the incoming member's whole
# state from a normal fitted to its own burn-in draws.
using Test, Nereus, Random, LinearAlgebra, Statistics, Serialization

@testset "noise swap with state proposals" begin
    @testset "fit / draw / density" begin
        rng = MersenneTwister(1)
        n = 4000
        # column 1 positive (log-uniform-like support), column 2 real
        X = hcat(exp.(0.3 .* randn(rng, n) .+ log(2.0)), 5.0 .+ 0.5 .* randn(rng, n))
        lo = [1e-3, -100.0]; hi = [1e3, 100.0]
        p = Nereus.fit_noise_state_proposal(X, [3, 7], lo, hi; inflate = 1.0)
        @test p !== nothing
        @test p.logt == BitVector([true, false])
        @test p.mu ≈ [log(2.0), 5.0] atol = 0.03
        @test sqrt.(diag(p.L * p.L')) ≈ [0.3, 0.5] rtol = 0.05
        vals = zeros(8)
        lq = Nereus.noise_state_draw!(vals, p, rng)
        @test isfinite(lq) && lq ≈ Nereus.noise_state_logq(p, vals)
        # density on the original scale: log-normal in column 1, normal in column 2
        x1, x2 = vals[3], vals[7]
        ref = (-0.5 * ((log(x1) - p.mu[1]) / 0.3)^2 - log(0.3) - 0.5log(2π) - log(x1)) +
              (-0.5 * ((x2 - p.mu[2]) / 0.5)^2 - log(0.5) - 0.5log(2π))
        @test lq ≈ ref atol = 0.15
        vals[7] = 1e4                          # outside the bound
        @test Nereus.noise_state_logq(p, vals) == -Inf
        @test Nereus.fit_noise_state_proposal(X[1:3, :], [3, 7], lo, hi) === nothing
        # draws from a few walkers that never moved: no proposal, not a
        # zero-width one (HD 18599: log q = -1e10 at the walkers' own state)
        @test Nereus.fit_noise_state_proposal(repeat(X[1:5, :], 400), [3, 7], lo, hi) === nothing
        Xs = copy(X); Xs[:, 2] .= 5.0                  # one parameter without spread
        @test Nereus.fit_noise_state_proposal(Xs, [3, 7], lo, hi) === nothing
    end


    @testset "seed: a fixed-member ensemble recovers the conditional posterior" begin
        # toy target over slots 2 (positive) and 4 (real): log-normal × normal,
        # started far from the mode
        logp(v, c) = -0.5 * ((log(v[2]) - log(3.0)) / 0.2)^2 - log(v[2]) -
                     0.5 * ((v[4] + 1.0) / 0.7)^2
        X = [[0.0, 8.0 * exp(0.05 * randn(MersenneTwister(i))), 0.0, 5.0 + 0.05i] for i in 1:16]
        ens = Nereus.NoiseStateEnsemble(X, Nereus.TransDimState(max_planets = 0, n_noise = 0), 0)
        D = Nereus.run_noise_state_ensemble!(ens, logp, [2, 4], [1e-3, -50.0], [1e3, 50.0],
                                             MersenneTwister(4); n_steps = 3000)
        p = Nereus.fit_noise_state_proposal(D, [2, 4], [1e-3, -50.0], [1e3, 50.0]; inflate = 1.0)
        @test p !== nothing
        @test p.mu ≈ [log(3.0), -1.0] atol = 0.1
        @test sqrt.(diag(p.L * p.L')) ≈ [0.2, 0.7] rtol = 0.2
        @test ens.n_done == 3000
        # a continued run picks up where it stopped
        Nereus.run_noise_state_ensemble!(ens, logp, [2, 4], [1e-3, -50.0], [1e3, 50.0],
                                         MersenneTwister(5); n_steps = 10)
        @test ens.n_done == 3010
    end

    # data: an RV signal tracked by two indicators, so decorrelation and the
    # rotation GP are competing explanations of the same scatter
    P_ROT, N, SIG = 9.0, 70, 1.0
    rng = MersenneTwister(20261127)
    t = sort(rand(rng, N)) .* 90.0
    act = 6.0 .* sin.(2π .* t ./ P_ROT)
    rv = act .+ SIG .* randn(rng, N)
    bis = 0.5 .* act .+ 0.3 .* randn(rng, N); fwhm = 0.8 .* act .+ 0.3 .* randn(rng, N)
    data = Data(t_rv = t, rv = rv, rv_err = fill(SIG, N), rv_inst = ones(Int, N),
                indicators = Dict("bis" => bis, "fwhm" => fwhm),
                indicator_errs = Dict("bis" => fill(0.3, N), "fwhm" => fill(0.3, N)))
    nms = NoiseModel[CeleriteRotation(), ActivityDecorrelation(; indicators = ["bis", "fwhm"])]
    rvmax = maximum(abs, rv)
    params = Params(; max_kplanet = 1, planet_modes = [RV_ONLY],
        instruments = InstrumentConfig(rv = ["I1"]), data = data, M_s = 1.0,
        priors = Dict{String,PriorSpec}(
            "P_k1" => UniformPrior(2.0, 40.0),
            "gamma_I1" => UniformPrior(-3rvmax, 3rvmax),
            "sigma_I1" => LogUniformPrior(0.05, 12.0),
            "gp_period" => UniformPrior(7.0, 11.0)),
        noise_models = nms, transdim_noise = true, stability = :none)

    @testset "state slots: own + shared, never another member's or a planet's" begin
        tog = [1, 2]
        s1, _ = Nereus.noise_state_slots(params, 1, tog)
        s2, _ = Nereus.noise_state_slots(params, 2, tog)
        nm(s) = Set(params.layout.names[s])
        @test "sigma_I1" in nm(s1) && "sigma_I1" in nm(s2)
        @test "gamma_I1" in nm(s1)
        @test !any(n -> occursin("_k1", n), nm(s1))
        @test isempty(intersect(setdiff(nm(s1), nm(s2)), Set(Nereus.noise_param_names(nms[2], params.config.instruments))))
        # one shared set per group: what differs between members is their own
        gs = Nereus.noise_state_group_slots(params, tog, tog)
        own(j) = Set(Nereus.noise_param_names(nms[j], params.config.instruments))
        @test setdiff(nm(gs[1]), own(1)) == setdiff(nm(gs[2]), own(2))
        @test issubset(Set(s1), Set(gs[1])) && issubset(Set(s2), Set(gs[2]))
        # RV members trade amplitude with the planets: the K slots are offered
        pk = Nereus.noise_state_planet_slots(params, tog)
        @test [params.layout.names[s] for (_, s) in pk] == ["K_k1"]
        @test first(pk[1]) == 1
    end

    @testset "move: reverse proposal ratio is the negative of the forward one" begin
        tog = [1, 2]
        th = Nereus.Theta{Float64}(params)
        th.td = Nereus.TransDimState(max_planets = 1, n_noise = 2)
        Nereus.activate_noise!(th.td, 1)
        props = Vector{Union{Nothing, Nereus.NoiseStateProposal}}(nothing, 2)
        r2 = MersenneTwister(3)
        for j in tog
            sl, up = Nereus.noise_state_slots(params, j, tog)
            lo = [Nereus.bounds(params.layout.unfrozen_priors[u])[1] for u in up]
            hi = [Nereus.bounds(params.layout.unfrozen_priors[u])[2] for u in up]
            X = hcat([clamp.(fill(0.5 * (max(l, -10) + min(h, 10)), 500) .+ 0.01 .* randn(r2, 500), l + 1e-9, h - 1e-9)
                      for (l, h) in zip(lo, hi)]...)
            props[j] = Nereus.fit_noise_state_proposal(X, sl, lo, hi)
            @test props[j] !== nothing
        end
        for v in eachindex(th.values)            # start inside A's proposal support
            th.values[v] = th.values[v]
        end
        Nereus.noise_state_draw!(th.values, props[1], r2)
        grp = NoiseModel[nms[1], nms[2]]
        cand, lq = Nereus.propose_noise_swap_state(th, MersenneTwister(9), grp, nms, props)
        @test isfinite(lq)
        @test Nereus.is_noise_active(cand.td, 2) && !Nereus.is_noise_active(cand.td, 1)
        back = Nereus.noise_state_logq(props[2], cand.values) - Nereus.noise_state_logq(props[1], th.values)
        @test lq ≈ -back
        # a member without a proposal: the move does not apply
        props[2] = nothing
        _, lq2 = Nereus.propose_noise_swap_state(th, MersenneTwister(9), grp, nms, props)
        @test lq2 == -Inf
    end


    @testset "sampler: a member with no burn-in draws still gets a proposal" begin
        # min_draws beyond any count: no member can be fitted from its draws, so
        # every one must come from the fixed-member seed, and the swap still mixes
        tgt = NereusTarget(params, data)
        td = TransDimConfig(; max_kplanet = 1, planets = false, noise = true,
                              toggleable = nms, noise_exclusion_groups = [nms])
        ck = joinpath(mktempdir(), "ck.jls")
        r = Nereus.sample_transdim_pt_emcee(tgt, data; td = td, n_temps = 4,
            n_walkers = 16, n_steps = 1500, n_burnin = 600, seed = 3,
            noise_state_min_draws = 10^9, checkpoint = ck, show_progress = false)
        props = open(deserialize, ck).state.noise_state.props
        @test all(j -> props[j] isa Nereus.NoiseStateProposal, 1:2)
        @test r.noise_td_accepted[1] > 0
    end

    @testset "sampler: per-planet-configuration proposals, and an exact resume" begin
        # planets on/off: each member gets a proposal per planet configuration
        # seen at β = 1, and a resumed run reproduces the uninterrupted one
        tgt = NereusTarget(params, data)
        td = TransDimConfig(; max_kplanet = 1, planets = true, noise = true,
                              toggleable = nms, noise_exclusion_groups = [nms])
        kw = (; td = td, n_temps = 4, n_walkers = 16, n_burnin = 600, seed = 11,
                show_progress = false, noise_state_min_draws = 50)
        d1 = mktempdir(); d2 = mktempdir()
        full = Nereus.sample_transdim_pt_emcee(tgt, data; kw..., n_steps = 1200,
                                               checkpoint = joinpath(d1, "ck.jls"))
        st = open(deserialize, joinpath(d1, "ck.jls")).state.noise_state
        @test !isempty(st.kprops)
        @test all(((j, k),) -> j in (1, 2) && k != typemax(UInt64), keys(st.kprops))
        # a configuration with the planet on carries its amplitude, the pooled
        # proposals never do
        kslot = params.layout.name_to_idx["K_k1"]
        for ((j, k), p) in st.kprops
            @test (kslot in p.slots) == (k & 0x1 == 0x1)
        end
        @test all(p -> p === nothing || !(kslot in p.slots), st.props)
        Nereus.sample_transdim_pt_emcee(tgt, data; kw..., n_steps = 900,
                                        checkpoint = joinpath(d2, "ck.jls"))
        res = Nereus.sample_transdim_pt_emcee(tgt, data; kw..., n_steps = 1200,
                                              checkpoint = joinpath(d2, "ck.jls"), resume = true)
        @test Array(res.chains) == Array(full.chains)
    end

    @testset "sampler: seeds agree where the stale swap leaves them apart" begin
        # Two GP kernels for one quasi-periodic signal, no planet. The stale move
        # draws the incoming kernel's hyperparameters from the prior while the
        # jitter and offset keep the outgoing kernel's values: measured with
        # `noise_state_swap = false`, 30 of ~13000 cold swaps were accepted and
        # the two seeds gave P(rotation) = 0.70 and 0.40. With state proposals
        # ~4% are accepted and the seeds agree to < 0.01.
        PR, NQ = 9.0, 100
        rq = MersenneTwister(20261127)
        tq = sort(rand(rq, NQ)) .* 120.0
        aq = 3.0 .* sin.(2π .* tq ./ PR .+ 0.3 .* sin.(2π .* tq ./ 50)) .*
             (1 .+ 0.4 .* sin.(2π .* tq ./ 70)) .+ 1.0 .* sin.(4π .* tq ./ PR)
        rvq = aq .+ randn(rq, NQ)
        dq = Data(t_rv = tq, rv = rvq, rv_err = ones(NQ), rv_inst = ones(Int, NQ))
        kq = NoiseModel[CeleriteRotation(), CeleriteSHO()]
        mq = maximum(abs, rvq)
        pq = Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
            instruments = InstrumentConfig(rv = ["I1"]), data = dq, M_s = 1.0,
            priors = Dict{String,PriorSpec}("gamma_I1" => UniformPrior(-3mq, 3mq),
                                            "sigma_I1" => LogUniformPrior(0.05, 12.0)),
            noise_models = kq, transdim_noise = true, stability = :none)
        tq_ = NereusTarget(pq, dq)
        tdq = TransDimConfig(; max_kplanet = 0, planets = false, noise = true,
                               toggleable = kq, noise_exclusion_groups = [kq])
        go(seed, on) = Nereus.sample_transdim_pt_emcee(tq_, dq; td = tdq, n_temps = 4,
            n_walkers = 16, n_steps = 4000, n_burnin = 1500, seed = seed,
            noise_state_swap = on, show_progress = false)
        prot(r) = mean(vec(Array(r.chains[:noise_active_1])) .> 0.5)
        # the defect: with the stale move alone the seeds do not agree
        o1 = go(1, false); o2 = go(2, false)
        @test o1.noise_swaps.state_proposed[1] == 0
        @test abs(prot(o1) - prot(o2)) > 0.1
        # the fix
        a = go(1, true); b = go(2, true)
        for r in (a, b)
            ns = r.noise_swaps
            @test ns.members_with_state_proposal == ns.members == 2
            @test ns.state_accepted[1] >= 200
            @test ns.state_accepted[1] > 10 * max(o1.noise_swaps.stale_accepted[1], 1)
        end
        @test abs(prot(a) - prot(b)) < 0.05
    end
end
