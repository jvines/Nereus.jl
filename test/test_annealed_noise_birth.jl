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
              propose_noise_death, propose_noise_death_annealed, ErrorScale,
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

    @testset "one bridge stage death == ordinary RJ death" begin
        for nm in (CeleriteRotation(channel = :rv), MaternGP(),
                   ActivityDecorrelation(indicators = ["bis"]))
            params, th = build(nm)
            tog = NoiseModel[nm]
            th.td.noise_active[1] = true
            checked = 0
            for trial in 1:10
                a_cand, a_acc, a_ll = propose_noise_death_annealed(
                    th, MersenneTwister(700 + trial), tog, loglik, logpri;
                    data = data, n_bridge = 1)
                b_cand, b_lqr = propose_noise_death(
                    th, MersenneTwister(700 + trial), tog; data = data)
                (isfinite(a_acc) && isfinite(b_lqr)) || continue
                rj = (logpri(b_cand) + loglik(b_cand)) - (logpri(th) + loglik(th)) + b_lqr
                @test a_acc ≈ rj atol = 1e-8
                @test a_ll ≈ loglik(b_cand) atol = 1e-10
                checked += 1
            end
            @test checked > 0
        end
    end

    @testset "annealed birth + annealed death sample P(M | D)" begin
        # One toggleable model with ONE parameter (ErrorScale), everything else
        # frozen, so the exact posterior odds are a 1-D quadrature. The chain
        # uses the sampler's move mix: half birth, half death (each annealed
        # over 6 stages), plus a within-model step on the parameter. The
        # annealed birth paired with the INSTANTANEOUS death is not reversible
        # and over-occupies the model; the time-reversed annealed death is.
        Nn = 40
        r0 = MersenneTwister(31)
        tt = sort(rand(r0, Nn)) .* 50.0
        ee = fill(1.0, Nn)
        yy = 1.2 .* randn(r0, Nn)              # errors underestimated by ~20%
        dd = Data(t_rv = tt, rv = yy, rv_err = ee, rv_inst = ones(Int, Nn))
        es = ErrorScale()
        pp = Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
            instruments = InstrumentConfig(rv = ["I1"]), data = dd,
            priors = Dict{String,PriorSpec}(
                "gamma_I1" => FixedPrior(0.0), "sigma_I1" => FixedPrior(0.0),
                "errscale_I1" => LogUniformPrior(0.3, 3.0)),
            noise_models = NoiseModel[es], transdim_noise = true, stability = :none)
        th = Theta{Float64}(pp)
        th.td = TransDimState(max_planets = 0, n_noise = 1)
        si = pp.layout.name_to_idx["errscale_I1"]
        th.values[si] = 1.0
        ll(x) = rv_log_likelihood(x, dd)
        lp(x) = log_prior(x)
        # exact: Z_off = L(off); Z_on = ∫ p(s) L(s) ds (log-uniform prior)
        L_off = ll(th)
        on = Theta{Float64}(pp); on.td = TransDimState(max_planets = 0, n_noise = 1)
        on.td.noise_active[1] = true; on.values .= th.values
        grid = range(log(0.3), log(3.0), length = 4001)
        vals = map(grid) do u
            on.values[si] = exp(u); ll(on)
        end
        mx = maximum(vals)
        Z_on = log(sum(exp.(vals .- mx)) * step(grid) / (log(3.0) - log(0.3))) + mx
        p_exact = 1 / (1 + exp(L_off - Z_on))
        @test 0.2 < p_exact < 0.8                    # the toy must be sensitive

        function run_chain(paired::Bool; n_iter = 30_000, seed = 5)
            rng = MersenneTwister(seed)
            x = Theta{Float64}(pp); x.td = TransDimState(max_planets = 0, n_noise = 1)
            x.values .= th.values
            cur_ll = ll(x); cur_lp = lp(x)
            n_on = 0
            for it in 1:n_iter
                if rand(rng) < 0.5
                    if rand(rng) < 0.5
                        c, a, l = propose_noise_birth_annealed(x, rng, NoiseModel[es], ll, lp;
                                      data = dd, n_bridge = 6, n_relax = 2,
                                      ll_current = cur_ll, lp_current = cur_lp)
                    elseif paired
                        c, a, l = propose_noise_death_annealed(x, rng, NoiseModel[es], ll, lp;
                                      data = dd, n_bridge = 6, n_relax = 2,
                                      ll_current = cur_ll, lp_current = cur_lp)
                    else
                        c, lq = propose_noise_death(x, rng, NoiseModel[es]; data = dd)
                        l = isfinite(lq) ? ll(c) : NaN
                        a = isfinite(lq) ? (l + lp(c)) - (cur_ll + cur_lp) + lq : -Inf
                    end
                    if isfinite(a) && log(rand(rng)) < a
                        x = c; cur_ll = l; cur_lp = lp(x)
                    end
                elseif x.td.noise_active[1]
                    old = x.values[si]
                    nv = old * exp(0.3 * randn(rng))
                    if 0.3 <= nv <= 3.0
                        x.values[si] = nv
                        l2 = ll(x); p2 = lp(x)
                        # log-scale step: Hastings log(nv/old)
                        if log(rand(rng)) < (l2 + p2) - (cur_ll + cur_lp) + log(nv / old)
                            cur_ll = l2; cur_lp = p2
                        else
                            x.values[si] = old
                        end
                    end
                end
                n_on += x.td.noise_active[1]
            end
            return n_on / n_iter
        end
        p_paired = run_chain(true)
        @test abs(p_paired - p_exact) < 0.03
        # The old pairing is measurably off on the same toy (guards the test's
        # sensitivity, so a future change cannot pass it by accident).
        p_old = run_chain(false)
        @test p_old - p_exact > 0.05
    end

    @testset "n_bridge must be at least 1" begin
        nm = CeleriteRotation(channel = :rv)
        params, th = build(nm)
        @test_throws ArgumentError propose_noise_birth_annealed(
            th, MersenneTwister(1), NoiseModel[nm], loglik, logpri;
            data = data, n_bridge = 0)
    end
end
