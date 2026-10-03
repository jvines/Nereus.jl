# The obliquity model options, one at a time: each is a general option of the
# Nereus target (ObliquityConfig, the a/R★ parametrization, the wrapped λ
# prior, the per-night tomographic noise), and each is pinned here against the
# thing it exists to do. The term-by-term match to the bespoke models is in
# test_obliquity_parity.jl; run_job and the samplers in test_obliquity_runjob.jl.
using Test
using Nereus
using Nereus: Theta, set_param!, get_param, log_prior, rv_log_likelihood,
              tomogram_log_likelihood, transit_log_likelihood, NereusTarget,
              transform_forward, eval_packed_logpdf, PACKED_WRAPPED, build_transform,
              TRANSFORM_IDENTITY, set_circular_window!, is_circular,
              _tomo_temporal_kernel, content_hash, _hash_skip, ParamsConfig,
              MaternGP, CeleriteSHO, NoiseModel, TomoNight, RMNight
using ForwardDiff, LogDensityProblems
using Distributions: logpdf
using Random, Statistics, LinearAlgebra

@isdefined(OS_P) || include(joinpath(@__DIR__, "fixtures", "obliquity_synthetic.jl"))

const OT_DIR = os_write_data(mktempdir())
const OT_RM, OT_TOMO = os_load(OT_DIR)

ot_params(rm, tomo; kw...) = begin
    d, names = obliquity_data(rm; tomo_nights = tomo)
    p = obliquity_params(d, names; P = OS_P, Tc = OS_TC0, b = OS_B, a_Rs = OS_ARS,
                         rr = OS_RR, vsini = OS_VSINI_MS,
                         K = isempty(rm) ? nothing : OS_K,
                         sigma0 = isempty(rm) ? nothing : rm, beta_p_floor = 2000.0,
                         occultation = :point, ld = (OS_U1, OS_U2), kw...)
    return d, names, p
end

"Seat every unfrozen parameter at the centre of its prior, then the geometry."
function ot_seat(p; extra = Dict{String,Float64}())
    th = Theta{Float64}(p)
    for (i, s) in enumerate(p.layout.unfrozen_idx)
        lo, hi = bounds(p.layout.unfrozen_priors[i])
        th.values[s] = isfinite(lo) && isfinite(hi) ? (lo + hi) / 2 :
                       mean(p.layout.unfrozen_priors[i].dist)
    end
    has(k) = haskey(p.layout.name_to_idx, k)
    for (k, v) in ("lambda_k1" => OS_LAM, "v_sin_i_star" => 25_900.0, "b_k1" => 0.75,
                   "a_Rs_k1" => 6.82, "K_k1" => 368.0)
        has(k) && set_param!(th, k, v)
    end
    for nt in OT_TOMO
        for (k, v) in ("tomo_alpha_$(nt.tag)" => 1.0, "tomo_alpha" => 1.0,
                       "tomo_sigma_line_$(nt.tag)" => 7.0,
                       "tomo_ell_v_$(nt.tag)" => 8.0, "tomo_jit_$(nt.tag)" => 2e-3,
                       "matern_sigma_tomo_$(nt.tag)" => 1e-3,
                       "matern_rho_tomo_$(nt.tag)" => 0.05)
            has(k) && set_param!(th, k, v)
        end
    end
    for (k, v) in extra
        set_param!(th, k, v)
    end
    return th
end

@testset "obliquity model options" begin

    @testset "wrapped lambda: a circle, not an interval" begin
        ps = WrappedUniformPrior(-π, π)
        @test is_wrapped(ps) && !is_wrapped(UniformPrior(-π, π))
        @test_throws ArgumentError WrappedUniformPrior(0.0, 3.0)   # not one period
        # density 1/2π everywhere, including outside the chart
        for x in (-π, 0.3, π, 7.5, -20.0)
            @test logpdf(ps, x) ≈ -log(2π)
            @test eval_packed_logpdf(Float64(x), PACKED_WRAPPED, 0.0, 0.0, -π * 1.0, π * 1.0) ≈ -log(2π)
        end
        @test logpdf(ps, Inf) == -Inf
        @test logpdf(UniformPrior(-π, π), 7.5) == -Inf          # the wall it removes
        @test prior_from_dict(prior_to_dict(ps)).dist isa Nereus.WrappedUniform

        d, _, p = ot_params(OT_RM, OT_TOMO)
        i = findfirst(==("lambda_k1"), p.layout.unfrozen_names)
        @test is_wrapped(p.layout.unfrozen_priors[i])
        @test is_circular("lambda_k1", p.layout.unfrozen_priors[i], p.config)
        # no logit wall for the unconstrained samplers
        @test build_transform(p).type_ids[i] == TRANSFORM_IDENTITY
        # moving the chart keeps it wrapped
        set_circular_window!(p, i, 0.0)
        @test is_wrapped(p.layout.unfrozen_priors[i]) && p.layout.unfrozen_priors[i].lo == 0.0
        # the posterior is periodic in lambda: +2π is the same point
        th = ot_seat(p)
        tg = NereusTarget(p, d; unconstrained = false)
        x = [th.values[s] for s in p.layout.unfrozen_idx]
        x2 = copy(x); x2[i] += 2π
        @test tg(x2) ≈ tg(x) rtol = 1e-12
        @test isfinite(tg(x2))
        # bounded lambda is still available
        _, _, pb = ot_params(OT_RM, OT_TOMO; lambda_prior = :bounded)
        ib = findfirst(==("lambda_k1"), pb.layout.unfrozen_names)
        @test !is_wrapped(pb.layout.unfrozen_priors[ib])
    end

    @testset "per-night sigma0: beta_p derived, no free amplitude" begin
        d, names, p = ot_params(OT_RM, TomoNight[])
        @test !("sigma_ccf" in p.layout.names) && !("beta_p" in p.layout.names)
        @test p.config.obliquity.sigma0 == OS_SIG0
        # beta_p from sigma0 and v sin i, floored
        @test arome_beta_p(15_702.0, 25_900.0, 2000.0) ≈ sqrt(15_702.0^2 - (0.5503 * 25_900.0)^2)
        @test arome_beta_p(15_082.0, 28_000.0, 2000.0) == 2000.0
        @test arome_beta_p(15_082.0, 28_000.0, 0.0) == 0.0
        # the floor branch has a finite derivative (no sqrt at 0)
        g = ForwardDiff.derivative(v -> arome_beta_p(15_082.0, v, 0.0), 28_000.0)
        @test isfinite(g) && g == 0.0
        # each night reads its own sigma0: changing one night's changes the likelihood
        th = ot_seat(p)
        ll = rv_log_likelihood(th, d)
        d2, n2, p2 = ot_params(OT_RM, TomoNight[])
        s0 = copy(OS_SIG0); s0["B"] = 20_000.0
        p3 = obliquity_params(d2, n2; P = OS_P, Tc = OS_TC0, b = OS_B, a_Rs = OS_ARS,
                              rr = OS_RR, vsini = OS_VSINI_MS, K = OS_K, sigma0 = s0,
                              beta_p_floor = 2000.0, occultation = :point,
                              ld = (OS_U1, OS_U2))
        @test rv_log_likelihood(ot_seat(p3), d2) != ll
        # without sigma0 the old free pair is still there
        p4 = obliquity_params(d2, n2; P = OS_P, Tc = OS_TC0, b = OS_B, a_Rs = OS_ARS,
                              rr = OS_RR, vsini = OS_VSINI_MS, arome = true)
        @test "sigma_ccf" in p4.layout.names && "beta_p" in p4.layout.names
        # sigma0 must cover every RV instrument
        @test_throws ArgumentError obliquity_params(d2, n2; P = OS_P, Tc = OS_TC0,
            b = OS_B, a_Rs = OS_ARS, rr = OS_RR, vsini = OS_VSINI_MS,
            sigma0 = Dict("A" => 15_000.0))
        @test_throws ArgumentError ObliquityConfig(occultation = :ring)
    end

    @testset "occulted flux: disc overlap or point planet" begin
        d, names, p_pt = ot_params(OT_RM, TomoNight[]; occultation = :point)
        _, _, p_dc = ot_params(OT_RM, TomoNight[]; occultation = :disc)
        a, b = rv_log_likelihood(ot_seat(p_pt), d), rv_log_likelihood(ot_seat(p_dc), d)
        @test isfinite(a) && isfinite(b) && a != b
        # point: rr² I(μ)/⟨I⟩ at the centre, zero off the disc
        f = Nereus.point_occulted_flux(0.3, 0.2, 0.1, 0.32, 0.30)
        μ = sqrt(1 - 0.3^2 - 0.2^2)
        @test f ≈ 0.01 * (1 - 0.32 * (1 - μ) - 0.30 * (1 - μ)^2) / (1 - 0.32 / 3 - 0.30 / 6)
        @test Nereus.point_occulted_flux(1.01, 0.0, 0.1, 0.32, 0.30) == 0.0
    end

    @testset "spectroscopic limb darkening" begin
        d, names, p = ot_params(OT_RM, TomoNight[])
        @test "u1_spec" in p.layout.names && "u2_spec" in p.layout.names
        th = ot_seat(p)
        @test spectroscopic_ld(th) == (OS_U1, OS_U2)
        ll = rv_log_likelihood(th, d)
        # read by the RM anomaly: a different pair changes the likelihood
        d2, n2 = obliquity_data(OT_RM)
        p0 = obliquity_params(d2, n2; P = OS_P, Tc = OS_TC0, b = OS_B, a_Rs = OS_ARS,
                              rr = OS_RR, vsini = OS_VSINI_MS, K = OS_K, sigma0 = OT_RM,
                              beta_p_floor = 2000.0, occultation = :point, ld = (0.0, 0.0))
        @test rv_log_likelihood(ot_seat(p0), d2) != ll
        # unphysical pairs are rejected where they are read
        p5 = obliquity_params(d2, n2; P = OS_P, Tc = OS_TC0, b = OS_B, a_Rs = OS_ARS,
                              rr = OS_RR, vsini = OS_VSINI_MS, K = OS_K, sigma0 = OT_RM,
                              ld = (0.3, 0.3),
                              priors = Dict{String,PriorSpec}("u1_spec" => UniformPrior(0.0, 1.0)))
        th5 = ot_seat(p5); set_param!(th5, "u1_spec", 0.9)       # u1 + u2 > 1
        @test rv_log_likelihood(th5, d2) == -Inf
        # without the option: no slots, the uniform disc as before
        p6 = obliquity_params(d2, n2; P = OS_P, Tc = OS_TC0, b = OS_B, a_Rs = OS_ARS,
                              rr = OS_RR, vsini = OS_VSINI_MS, K = OS_K, sigma0 = OT_RM)
        @test !("u1_spec" in p6.layout.names)
        @test spectroscopic_ld(ot_seat(p6)) == (0.0, 0.0)
    end

    @testset "geometry: fixed or Gaussian, a/R★ sampled directly" begin
        d, names, p = ot_params(OT_RM, OT_TOMO)
        pr = p.config.priors
        @test is_fixed(pr["P_k1"]) && fixed_value(pr["P_k1"]) == OS_P
        @test is_fixed(pr["Tc_k1"]) && fixed_value(pr["Tc_k1"]) == OS_TC0
        @test is_fixed(pr["rr_k1"]) && is_fixed(pr["sesinw_k1"]) && is_fixed(pr["secosw_k1"])
        @test pr["b_k1"].dist.untruncated.μ == OS_B[1]
        @test pr["a_Rs_k1"].dist.untruncated.σ == OS_ARS[2]
        @test p.config.parametrization.sample_a_Rs && !p.config.parametrization.use_rho_s
        # Gaussian P and Tc
        _, _, pg = ot_params(OT_RM, OT_TOMO; P = (OS_P, 4e-7), Tc = (OS_TC0, 1e-4))
        @test !is_fixed(pg.config.priors["P_k1"]) && "Tc_k1" in pg.layout.unfrozen_names
        # a/R★ is what the likelihood reads
        th = ot_seat(p)
        a = tomogram_log_likelihood(th, d)
        set_param!(th, "a_Rs_k1", 7.3)
        @test tomogram_log_likelihood(th, d) != a
        @test Nereus.planet_a_Rs(th, 1, OS_P) == 7.3
        @test_throws ArgumentError ParametrizationConfig(use_rho_s = true, sample_a_Rs = true)
    end

    @testset "a/R★ slot reaches the transit model" begin
        # photometry alongside: the light curve is computed from the sampled a/R★
        t = collect(range(OS_TC0 - 0.15, OS_TC0 + 0.15; length = 200))
        dp = Data(t_phot = t, flux = ones(200), flux_err = fill(1e-3, 200),
                  phot_inst = ones(Int, 200), tomo = OT_TOMO)
        pp = obliquity_params(dp, String[]; P = OS_P, Tc = OS_TC0, b = OS_B,
                              a_Rs = OS_ARS, rr = OS_RR, vsini = OS_VSINI_MS,
                              pm_names = ["LC"])
        th = ot_seat(pp)
        set_param!(th, "q1_LC", 0.3); set_param!(th, "q2_LC", 0.3)
        l1 = transit_log_likelihood(th, dp)
        set_param!(th, "a_Rs_k1", 4.0)
        @test transit_log_likelihood(th, dp) != l1
    end

    @testset "maps alone: no dummy velocity" begin
        d, names, p = ot_params(RMNight[], OT_TOMO)
        @test isempty(d.rv) && isempty(names)
        @test p.config.planet_modes == [PM_DT]
        @test !any(n -> startswith(n, "gamma_") || startswith(n, "sigma_") || n == "K_k1",
                   p.layout.names)
        th = ot_seat(p)
        @test rv_log_likelihood(th, d) == 0.0
        @test isfinite(tomogram_log_likelihood(th, d))
    end

    @testset "shared alpha: one amplitude for every map" begin
        d, _, p = ot_params(RMNight[], OT_TOMO; shared_alpha = true)
        @test "tomo_alpha" in p.layout.names
        @test !any(n -> startswith(n, "tomo_alpha_"), p.layout.names)
        @test length(unique(p.layout.systemic.tomo_alpha)) == 1
        _, _, pn = ot_params(RMNight[], OT_TOMO)
        # equal per-night amplitudes give the same likelihood as the shared one
        @test tomogram_log_likelihood(ot_seat(p), d) ≈ tomogram_log_likelihood(ot_seat(pn), d) rtol = 1e-12
    end

    @testset "per-night Kronecker kernel and white term" begin
        d, _, p = ot_params(RMNight[], OT_TOMO)
        for nt in OT_TOMO
            @test "matern_sigma_tomo_$(nt.tag)" in p.layout.names
            @test "tomo_jit_$(nt.tag)" in p.layout.names
        end
        th = ot_seat(p)
        # each night reads ITS kernel
        K1 = _tomo_temporal_kernel(th, OT_TOMO[1])
        set_param!(th, "matern_sigma_tomo_B", 5e-3)
        @test _tomo_temporal_kernel(th, OT_TOMO[1]) == K1
        @test _tomo_temporal_kernel(th, OT_TOMO[2]) ≈ (5e-3)^2 .* Nereus._kern(OT_TOMO[2].t, 0.05)
        # σ_n is a parameter, read by the likelihood
        a = tomogram_log_likelihood(th, d)
        set_param!(th, "tomo_jit_A", 4e-3)
        @test tomogram_log_likelihood(th, d) != a
        # the :tomo defaults are in map units and hours, not RV units and days
        pr = p.config.priors
        @test bounds(pr["matern_rho_tomo_A"]) == (0.05 / 24, 12.0 / 24)
        @test bounds(pr["matern_sigma_tomo_A"])[2] == 1.0
        @test bounds(pr["tomo_jit_A"]) == (1e-6, 1.0)
        @test bounds(pr["tomo_ell_v_A"]) == (1.0, 60.0)
        # a :tomo model naming a map that does not exist is an error, not white
        @test_throws ArgumentError obliquity_params(d, String[]; P = OS_P, Tc = OS_TC0,
            b = OS_B, a_Rs = OS_ARS, rr = OS_RR, vsini = OS_VSINI_MS,
            noise_models = NoiseModel[MaternGP(channel = :tomo, instruments = ["Z"])])
        # an SHO temporal kernel decodes NATURAL logs, like its priors
        _, _, ps = ot_params(RMNight[], OT_TOMO; tomo_noise = :sho)
        ths = ot_seat(ps, extra = Dict("gp_log_S0_tomo_A" => log(1e-6),
                                       "gp_log_Q_tomo_A" => log(2.0),
                                       "gp_log_omega0_tomo_A" => log(20.0)))
        K = _tomo_temporal_kernel(ths, OT_TOMO[1])
        @test K[1, 1] ≈ 1e-6 * 20.0 * 2.0 rtol = 1e-10        # k(0) = S0 ω0 Q
    end

    @testset "the shadow follows the fitted ephemeris" begin
        d, _, p = ot_params(RMNight[], OT_TOMO; Tc = (OS_TC0, 0.01))
        th = ot_seat(p)
        set_param!(th, "Tc_k1", OS_TC0)
        a = tomogram_log_likelihood(th, d)
        set_param!(th, "Tc_k1", OS_TC0 + 0.02)
        @test tomogram_log_likelihood(th, d) < a
        # an eccentric orbit uses the full sky position, and reduces to the
        # circular geometry at e -> 0
        _, _, pe = ot_params(RMNight[], OT_TOMO; ecc = :free)
        the = ot_seat(pe, extra = Dict("sesinw_k1" => 0.0, "secosw_k1" => 1e-7))
        @test tomogram_log_likelihood(the, d) ≈ tomogram_log_likelihood(ot_seat(p), d) rtol = 1e-6
    end

    @testset "gradients: circular orbit and the Kronecker term" begin
        # the fixed circular orbit used to give NaN derivatives (atan(0, 0))
        dv, nv = obliquity_data(OT_RM)
        pv = obliquity_params(dv, nv; P = OS_P, Tc = OS_TC0, b = 0.75, a_Rs = 6.82,
                              rr = OS_RR, vsini = OS_VSINI_MS, K = OS_K, sigma0 = OT_RM)
        tv = NereusTarget(pv, dv)
        yv = transform_forward([mean(bounds(q)) for q in pv.layout.unfrozen_priors], tv.transform)
        @test all(isfinite, ForwardDiff.gradient(y -> LogDensityProblems.logdensity(tv, y), yv))
        # joint target: AD gradient (analytic Kronecker derivative) = finite differences
        d, _, p = ot_params(OT_RM, OT_TOMO)
        tg = NereusTarget(p, d)
        th = ot_seat(p)
        y = transform_forward([th.values[s] for s in p.layout.unfrozen_idx], tg.transform)
        f(y) = LogDensityProblems.logdensity(tg, y)
        g = ForwardDiff.gradient(f, y)
        h = 1e-6
        for i in eachindex(y)
            e = zeros(length(y)); e[i] = h
            fd = (f(y .+ e) - f(y .- e)) / (2h)
            @test g[i] ≈ fd rtol = 1e-4 atol = 1e-5
        end
        # Enzyme would differentiate only part of this posterior: refused
        @test_throws ArgumentError Nereus.EnzymeGradientConfig(tg)
    end

    @testset "checkpoints written before the options keep resuming" begin
        d, _, p = ot_params(OT_RM, OT_TOMO)
        # default options are left out of the model hash ...
        @test _hash_skip(p.config, :obliquity) == false        # non-default: hashed
        p0 = Params(max_kplanet = 1, planet_modes = [RV_ONLY],
                    instruments = InstrumentConfig(rv = ["I"]),
                    data = Data(t_rv = [1.0, 2.0, 3.0], rv = [1.0, 2.0, 0.0],
                                rv_err = ones(3)), stability = :none)
        @test _hash_skip(p0.config, :obliquity)
        @test _hash_skip(p0.config.parametrization, :sample_a_Rs)
        # ... so the hash of a default model is the one computed before they
        # existed: the same fold over every OTHER field.
        legacy = let x = p0.config, seen = IdDict{Any,Nothing}()
            h = hash(string(nameof(typeof(x))), zero(UInt))
            for f in fieldnames(typeof(x))
                f === :obliquity && continue
                h = Nereus._content_hash(getfield(x, f), hash(f, h), seen)
            end
            h
        end
        @test content_hash(p0.config) == legacy
        @test content_hash(p.config) != content_hash(p0.config)
    end
end
