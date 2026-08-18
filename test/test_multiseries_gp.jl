# Correctness gates for the O(N·r²) semiseparable multi-series activity GP
# (src/noise/multiseries_gp.jl). See NEREUS_MULTISERIES_GP_SPEC.md §4.
using Nereus, Test, Random, LinearAlgebra
import ForwardDiff
using Nereus: multiseries_loglike, dense_multiseries_loglike, semiseparable_loglike,
               SSQP, SSMatern32, SSSHO, SSMEP, SSES, SSESP,
               k0, mk2_0, derivative_kink, _terms, _base_generators, _kernel_rank,
               activity_gp_joint_logpdf_lowrank, _ss_latent_kernel,
               activity_gp_covariance

# a random non-simultaneous 3-channel config
function _cfg(rng, npc)
    t = Float64[]; y = Float64[]; v = Float64[]; sid = Int[]
    for s in 1:3
        ts = sort(rand(rng, npc) .* 40.0)
        append!(t, ts); append!(y, randn(rng, npc)); append!(v, fill(0.3, npc)); append!(sid, fill(s, npc))
    end
    return t, y, v, sid
end

_admissible() = ("SSQP"       => SSQP(1.0, 1.0 / sqrt(3), 0.2, 0.2 * sqrt(3)),  # λa=νb ⇒ differentiable
                 "SSMatern32" => SSMatern32(1.2, 8.0),
                 "SSSHO_under" => SSSHO(1.0, 9.0, 3.0),
                 "SSSHO_over"  => SSSHO(1.0, 9.0, 0.3),
                 "SSMEP"      => SSMEP(1.1, 12.0, 8.0, 0.5),
                 "SSES"       => SSES(1.3, 9.0),
                 "SSESP"      => SSESP(1.2, 15.0, 8.0, 0.6; nharm = 3))

@testset "multiseries semiseparable GP" begin
    rng = MersenneTwister(20260720)
    t, y, v, sid = _cfg(rng, 40)
    α = [1.0, 0.6, -0.4]; β = [0.3, -0.2, 0.5]; jit = [0.1, 0.2, 0.05]

    @testset "gate 1 — dense-oracle parity ($nm)" for (nm, k) in _admissible()
        ss = multiseries_loglike(t, y, v, sid, α, β, k; jitter = jit)
        de = dense_multiseries_loglike(t, y, v, sid, α, β, k; jitter = jit)
        @test isfinite(ss)
        @test abs(ss - de) / max(abs(de), 1) < 1e-8
    end

    @testset "gate 2 — identities + rejection" begin
        # Identity 1: −k″(0) = Σ dU·dV, constant across rows.
        for (nm, k) in _admissible()
            terms = nm == "SSESP" ? nothing : _terms(k)
            if terms !== nothing
                U0, V0, dU0, dV0, φ = _base_generators(terms, t[1:10], diff(t[1:10]))
                r = _kernel_rank(terms)
                B = [sum(dU0[i, c] * dV0[i, c] for c in 1:r) for i in 1:10]
                @test maximum(abs, B .- mk2_0(k)) < 1e-9 * max(1, abs(mk2_0(k)))
            end
            @test abs(derivative_kink(k)) < 1e-8 * max(1, abs(k0(k)))   # admissible ⇒ kink≈0
            @test mk2_0(k) > 0                                          # Var(Ġ) > 0
        end
        # a bare Exp (kink≠0) rejects a β≠0 derivative coupling, but works with β=0.
        exp_k = SSQP(1.0, 0.0, 0.3, 0.0)
        @test abs(derivative_kink(exp_k)) > 1e-6
        @test_throws ArgumentError multiseries_loglike(t, y, v, sid, α, β, exp_k)
        @test isfinite(multiseries_loglike(t, y, v, sid, α, zeros(3), exp_k))
    end

    @testset "gate 3 — ForwardDiff = finite-diff" begin
        mk = (name, h) -> name == :MEP ? SSMEP(h[1], h[2], h[3], h[4]) :
                          name == :ESP ? SSESP(h[1], h[2], h[3], h[4]; nharm = 3) :
                          name == :SHO ? SSSHO(h[1], h[2], h[3]) : SSMatern32(h[1], h[2])
        for (name, h0) in ((:MEP, [1.1, 12.0, 8.0, 0.5]), (:ESP, [1.2, 15.0, 8.0, 0.6]),
                           (:SHO, [1.0, 9.0, 3.0]), (:M32, [1.2, 8.0]))
            f = p -> multiseries_loglike(t, y, v, sid, p[1:3], p[4:6], mk(name, p[7:end]); jitter = jit)
            p0 = vcat(α, β, h0)
            gad = ForwardDiff.gradient(f, p0)
            gfd = map(eachindex(p0)) do i
                δ = 1e-6 * max(abs(p0[i]), 1); pp = copy(p0); pm = copy(p0); pp[i] += δ; pm[i] -= δ
                (f(pp) - f(pm)) / (2δ)
            end
            @test maximum(abs.(gad .- gfd) ./ (abs.(gfd) .+ 1)) < 1e-5
        end
    end

    @testset "gate 4 — PSD sweep never throws" begin
        rng2 = MersenneTwister(7); nthrow = 0
        for _ in 1:400
            σ = exp(randn(rng2)); ρ = exp(2randn(rng2) + 1); P = exp(2randn(rng2) + 2); η = exp(randn(rng2))
            k = rand(rng2, (SSMEP(σ, ρ, P, η), SSES(σ, ρ), SSESP(σ, ρ, P, η; nharm = 3),
                            SSSHO(σ, P, exp(randn(rng2))), SSMatern32(σ, ρ)))
            ll = try
                multiseries_loglike(t, y, v, sid, randn(rng2, 3), 0.3 .* randn(rng2, 3), k)
            catch
                nthrow += 1; NaN
            end
            @test !isnan(ll) || nthrow > 0     # finite or −Inf, never NaN-from-throw
        end
        @test nthrow == 0
    end

    @testset "gate 5 — linear scaling + speedup vs lowrank" begin
        k = SSMEP(1.0, 20.0, 12.0, 0.5)
        function tmed(N)
            tt, yy, vv, ss = _cfg(MersenneTwister(N), N ÷ 3)
            a = [1.0, 0.6, 0.4]; b = [0.2, -0.1, 0.3]
            multiseries_loglike(tt, yy, vv, ss, a, b, k)
            minimum(@elapsed(multiseries_loglike(tt, yy, vv, ss, a, b, k)) for _ in 1:3)
        end
        t1 = tmed(1500); t4 = tmed(6000)
        @test t4 / t1 < 8                      # 4× N ⇒ ≪ 16× time (sub-quadratic; linear≈4×)
        # speedup vs the Woodbury lowrank on a comparable size
        Ne = 1000; ep = sort(rand(MersenneTwister(1), Ne) .* 50.0)
        yl = randn(MersenneTwister(2), 2Ne); σ2 = fill(0.09, 2Ne)
        activity_gp_joint_logpdf_lowrank(ep, [1.0, 0.6], [0.2, -0.1], 1.0, 12.0, 20.0, 0.5, yl, σ2)
        tlow = minimum(@elapsed(activity_gp_joint_logpdf_lowrank(ep, [1.0, 0.6], [0.2, -0.1], 1.0, 12.0, 20.0, 0.5, yl, σ2)) for _ in 1:3)
        @test tlow / tmed(2000) > 50           # ≫ crossover (measured ~1000×; assert conservatively)
    end

    @testset "gate 6 — recovery: loglike peaks at the true couplings" begin
        # simulate a 3-series MEP system, hold hyperparams at truth, check the
        # marginal loglike is maximised at the injected (α,β) vs perturbations.
        rng3 = MersenneTwister(99)
        tt, _, vv, ss = _cfg(rng3, 60)
        kern = SSMEP(1.0, 18.0, 11.0, 0.5)
        αt = [1.0, 0.7, -0.5]; βt = [0.25, -0.15, 0.3]
        # draw y from the true joint covariance via the dense builder's Cholesky
        Kd = [begin
                  τ = tt[i] - tt[j]; kk, kp, kpp = Nereus._kfuncs(kern, τ)
                  ai = αt[ss[i]]; bi = βt[ss[i]]; aj = αt[ss[j]]; bj = βt[ss[j]]
                  ai * aj * kk - ai * bj * kp + bi * aj * kp - bi * bj * kpp + (i == j ? vv[i] : 0.0)
              end for i in eachindex(tt), j in eachindex(tt)]
        ytrue = cholesky(Symmetric(Kd)).L * randn(rng3, length(tt))
        ll_true = multiseries_loglike(tt, ytrue, vv, ss, αt, βt, kern)
        nbetter = 0
        for _ in 1:30
            αp = αt .+ 0.3 .* randn(rng3, 3); βp = βt .+ 0.15 .* randn(rng3, 3)
            multiseries_loglike(tt, ytrue, vv, ss, αp, βp, kern) > ll_true && (nbetter += 1)
        end
        @test nbetter <= 3                     # truth beats ≥90% of random perturbations
    end

    @testset "gate 7 — ActivityGP backend wiring (end-to-end)" begin
        # Synthesise a joint RV+BIS dataset from the true Rajpaul covariance,
        # then score it through the PUBLIC Nereus.rv_log_likelihood with the
        # ActivityGP latent_kernel routing into the semiseparable backend.
        rng = MersenneTwister(31)
        n = 30
        t_rv = sort!(40.0 .* rand(rng, n))
        amp_t, P_t, λe_t, λp_t = 1.0, 12.0, 50.0, 0.5
        Vc_t, Vr_t, Bc_t, Br_t = 3.0, 0.3, 0.05, 0.005
        σ_rv, σ_bis = 0.5, 0.005
        t_flat = vcat(t_rv, t_rv)
        ch_flat = vcat(fill(:rv, n), fill(:bis, n))
        a_flat = vcat(fill(Vc_t, n), fill(Bc_t, n))
        b_flat = vcat(fill(Vr_t, n), fill(Br_t, n))
        Σ = activity_gp_covariance(t_flat, ch_flat, a_flat, b_flat, amp_t, P_t, λe_t, λp_t)
        Σn = Σ + Diagonal(vcat(fill(σ_rv, n), fill(σ_bis, n)) .^ 2)
        yj = cholesky(Symmetric(Σn)).L * randn(rng, 2n)
        rv_obs, bis_obs = yj[1:n], yj[(n+1):end]

        data = Nereus.Data(; t_rv = t_rv, rv = rv_obs, rv_err = fill(σ_rv, n),
                              rv_inst = ones(Int, n),
                              indicators = Dict("bis" => bis_obs),
                              indicator_errs = Dict("bis" => fill(σ_bis, n)))
        ic = Nereus.InstrumentConfig(rv = ["SIM"])

        function _score(latent)
            agp = Nereus.ActivityGP(channels = [:bis], latent_kernel = latent)
            p = Nereus.Params(; max_kplanet = 0, planet_modes = Nereus.PlanetDataSources[],
                                 instruments = ic, data = data, M_s = 1.0,
                                 noise_models = Nereus.NoiseModel[agp])
            th = Nereus.Theta{Float64}(p)
            truth = Dict("gamma_SIM" => 0.0, "sigma_SIM" => 1e-3,
                         "gp_act_period" => P_t, "gp_act_lambda_e" => λe_t,
                         "gp_act_lambda_p" => λp_t, "Vc" => Vc_t, "Vr" => Vr_t,
                         "Bc" => Bc_t, "Br" => Br_t)
            for nm in p.layout.unfrozen_names
                haskey(truth, nm) && Nereus.set_param!(th, nm, truth[nm])
            end
            ll0 = Nereus.rv_log_likelihood(th, data)
            Nereus.set_param!(th, "gp_act_period", 3.0)   # very wrong
            lloff = Nereus.rv_log_likelihood(th, data)
            return ll0, lloff
        end

        # (A) every semiseparable family routes through the backend, stays
        # finite, and prefers the true period over a wrong one — i.e. the
        # wired kernel is a working activity model end-to-end.
        ll_qp, _ = _score(:qp_dense)
        for latent in (:mep, :esp, :sho, :matern32, :es)
            ll_t, ll_off = _score(latent)
            @test isfinite(ll_t)
            @test ll_off < ll_t              # truth beats a wrong period
            @test ll_t != ll_qp              # took the new path (different kernel)
        end

        # (B) the ActivityGP→semiseparable hyperparameter mapping yields an
        # admissible kernel whose linear-time solver matches its dense oracle
        # to machine precision (closes the mapping↔solver loop on the real map).
        ts, ys, vs, ss = _cfg(MersenneTwister(5), 45)
        αm = [Vc_t, Bc_t, -0.02]; βm = [Vr_t, Br_t, 0.01]
        for latent in (:mep, :esp, :sho, :matern32, :es)
            k = _ss_latent_kernel(latent, amp_t, P_t, λe_t, λp_t)
            @test abs(derivative_kink(k)) < 1e-8 * max(1, abs(k0(k)))   # admissible ⇒ β allowed
            a = multiseries_loglike(ts, ys, vs, ss, αm, βm, k)
            d = dense_multiseries_loglike(ts, ys, vs, ss, αm, βm, k)
            @test isfinite(a) && abs(a - d) / max(abs(d), 1) < 1e-8
        end
    end
end
