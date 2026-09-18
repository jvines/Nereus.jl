# Annealed (bridged) reversible-jump birth for noise models.
#
# The idea is Jose's and it is old: propose a birth, run a few MCMC steps to let
# the newborn settle, THEN decide. `climb_newborn!` already does exactly that
# and is restricted to burn-in, because done naively it is not reversible — you
# optimise the candidate before judging it, so births are accepted more often
# than the posterior warrants and the chain drifts to higher dimension.
#
# The annealed construction makes it reversible by bridging
#   rho_0 = prior_A * L_A * q(u)   ->   rho_T = prior_B * L_B
# and accumulating sum_t (b_t - b_{t-1}) * [log rho_T - log rho_0] along the way.
#
# THE TEST THAT MATTERS is the reduction: with a SINGLE bridge stage no
# relaxation happens and log W must collapse to the ordinary RJ acceptance
# exponent, to machine precision. That is what catches a mis-accumulated
# weight — the failure mode that produces plausible, wrong occupancies.
using Test
using Nereus
using Nereus: Params, Data, InstrumentConfig, NereusTarget, Theta,
              TransDimState, NoiseModel, PlanetDataSources,
              ActivityDecorrelation, CeleriteRotation, MaternGP,
              propose_noise_birth, propose_noise_birth_annealed,
              rv_log_likelihood, log_prior, _noise_model_slots,
              _noise_block_logprior
using Random, Statistics

@testset "annealed noise birth" begin
    P_ROT, N, SIG = 8.0, 60, 1.5
    rng0 = MersenneTwister(4242)
    t = sort(rand(rng0, N)) .* 70.0
    act = 5.0 .* (sin.(2π .* t ./ P_ROT) .+ 0.4 .* sin.(4π .* t ./ P_ROT .+ 0.5))
    rv = act .+ SIG .* randn(rng0, N)
    bis = act .+ 1.5 .* randn(rng0, N)
    data = Data(t_rv = t, rv = rv, rv_err = fill(SIG, N), rv_inst = ones(Int, N),
                indicators = Dict("bis" => bis),
                indicator_errs = Dict("bis" => fill(1.0, N)))
    rvmax = maximum(abs, rv)

    function build(nm)
        params = Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
            instruments = InstrumentConfig(rv = ["I1"]), data = data,
            priors = Dict{String,PriorSpec}(
                "gamma_I1" => UniformPrior(-3rvmax, 3rvmax),
                "sigma_I1" => LogUniformPrior(0.2, 12.0)),
            noise_models = NoiseModel[nm], transdim_noise = true,
            stability = :none)
        th = Theta{Float64}(params)
        th.td = TransDimState(max_planets = 0, n_noise = 1)
        # A freshly constructed Theta leaves inactive-model slots at junk, so
        # `log_prior` is -Inf there and nothing can bridge FROM it. Seat every
        # unfrozen parameter inside its own prior first. (In the sampler this
        # is handled by spike_slab_log_prior, which conditions on the active
        # set — that difference is why the annealed move takes the caller's
        # prior function rather than assuming one.)
        rr = MersenneTwister(99)
        for (i, slot) in enumerate(params.layout.unfrozen_idx)
            lo, hi = bounds(params.layout.unfrozen_priors[i])
            v = rand(rr, params.layout.unfrozen_priors[i].dist)
            th.values[slot] = clamp(v, lo, hi)
        end
        return params, th
    end
    loglik(x) = rv_log_likelihood(x, data)
    logpri(x) = log_prior(x)

    @testset "one bridge stage == ordinary RJ birth" begin
        for nm in (CeleriteRotation(channel = :rv), MaternGP(),
                   ActivityDecorrelation(indicators = ["bis"]))
            params, th = build(nm)
            tog = NoiseModel[nm]
            checked = 0
            for trial in 1:10
                # identical RNG stream => identical drawn candidate
                a_cand, a_acc, a_ll = propose_noise_birth_annealed(
                    th, MersenneTwister(900 + trial), tog, loglik, logpri;
                    data = data, n_bridge = 1)
                isfinite(a_acc) || continue
                b_cand, b_lqr = propose_noise_birth(
                    th, MersenneTwister(900 + trial), tog; data = data)
                isfinite(b_lqr) || continue
                # ordinary RJ exponent at the same candidate
                ll_new = loglik(b_cand); ll_old = loglik(th)
                slots = _noise_model_slots(b_cand, 1)
                rj = (logpri(b_cand) + ll_new) - (logpri(th) + ll_old) + b_lqr
                @test a_acc ≈ rj atol = 1e-8
                checked += 1
            end
            @test checked > 0
        end
    end

    @testset "bridging does not change the target, only the path" begin
        # More bridge stages must still produce a FINITE, usable exponent and a
        # candidate that respects the model's bounds. (Correctness of the
        # stationary distribution is the reduction test above plus the
        # occupancy gates; here we only guard against NaN/out-of-bounds.)
        nm = CeleriteRotation(channel = :rv)
        params, th = build(nm)
        tog = NoiseModel[nm]
        for nb in (1, 2, 4, 8)
            fin = 0
            for trial in 1:6
                cand, acc, ll = propose_noise_birth_annealed(
                    th, MersenneTwister(50 + trial), tog, loglik, logpri;
                    data = data, n_bridge = nb, n_relax = 2)
                isfinite(acc) || continue
                fin += 1
                @test isfinite(ll)
                for slot in _noise_model_slots(cand, 1)
                    uf = findfirst(==(slot), params.layout.unfrozen_idx)
                    lo, hi = bounds(params.layout.unfrozen_priors[uf])
                    @test lo <= cand.values[slot] <= hi
                end
            end
            @test fin > 0
        end
    end

    @testset "relaxation improves the newborn's fit" begin
        # Not a correctness property — an efficiency one, and the entire point
        # of the move. With bridging the accepted candidate should sit at a
        # better likelihood than the raw draw more often than not.
        nm = CeleriteRotation(channel = :rv)
        params, th = build(nm)
        tog = NoiseModel[nm]
        better = 0; total = 0
        for trial in 1:25
            raw, lqr = propose_noise_birth(th, MersenneTwister(trial), tog;
                                           data = data)
            isfinite(lqr) || continue
            ll_raw = loglik(raw)
            _, acc, ll_br = propose_noise_birth_annealed(
                th, MersenneTwister(trial), tog, loglik, logpri;
                data = data, n_bridge = 8, n_relax = 3)
            (isfinite(acc) && isfinite(ll_raw) && isfinite(ll_br)) || continue
            total += 1
            ll_br > ll_raw && (better += 1)
        end
        @test total > 0
        @test better > total ÷ 2
    end

    @testset "n_bridge must be at least 1" begin
        nm = CeleriteRotation(channel = :rv)
        params, th = build(nm)
        @test_throws ArgumentError propose_noise_birth_annealed(
            th, MersenneTwister(1), NoiseModel[nm], loglik, logpri;
            data = data, n_bridge = 0)
    end
end
