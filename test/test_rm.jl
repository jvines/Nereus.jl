# Rossiter-McLaughlin (Hirano+ 2011 lite) tests.

using Test
using Nereus
using Random: MersenneTwister
using ForwardDiff

@testset "Rossiter-McLaughlin (Hirano+ 2011)" begin
    # ----- rm_signal math sanity -----------------------------------
    # Mid-transit (x=0, y=b) with λ=0: x_p = 0, no RM
    @test rm_signal(0.0, 0.5, 0.01, 4000.0, 0.0) ≈ 0.0  atol=1e-12
    # Approaching limb (x=-1, y=b) with λ=0: planet on blue half →
    # RM positive (red-shift of unblocked disk mean)
    rm_neg_x = rm_signal(-1.0, 0.5, 0.005, 4000.0, 0.0)
    rm_pos_x = rm_signal( 1.0, 0.5, 0.005, 4000.0, 0.0)
    @test rm_neg_x > 0
    @test rm_pos_x < 0
    @test isapprox(rm_neg_x, -rm_pos_x; atol=1e-12)
    # No transit (Δflux=0) → zero RM
    @test rm_signal(0.5, 0.2, 0.0, 4000.0, 0.3) == 0.0
    # Sign-flip under λ → λ+π
    @test rm_signal(0.7, 0.3, 0.005, 4000.0, 0.0) ≈
          -rm_signal(0.7, 0.3, 0.005, 4000.0, π)  atol=1e-10

    # ----- planet_sky_position circular -----------------------------
    # Circular orbit at mid-transit (t = Tp + (P/4) for ω=π/2 → mid-transit)
    # Use ω=π/2 so transit happens at t=Tp; check x≈0, |y|≈b.
    x, y = planet_sky_position(0.0, 3.5, 0.0, π/2, 0.0, 0.5, 8.0)
    @test abs(x) < 1e-10
    @test abs(abs(y) - 0.5) < 1e-10

    # Quarter-period later (t = P/4): planet moved along orbit by 90°.
    # For ω=π/2, e=0: at t = Tp+P/4 the planet is at the side of the
    # orbit, sky-plane x should be large, y near zero.
    x2, _ = planet_sky_position(3.5/4, 3.5, 0.0, π/2, 0.0, 0.5, 8.0)
    @test abs(x2) > 1.0

    # ----- front/back gate: no RM at superior conjunction ------------
    # x_sky and y_sky are identical at inferior and superior conjunction, so
    # a sky-separation test alone cannot tell a transit from an occultation.
    # z_los disambiguates: > 0 in front of the star, < 0 behind it.
    P_, e_, ω_, Tp_, b_, aRs_ = 3.5, 0.0, π/2, 0.0, 0.5, 8.0
    xin, yin, zin = planet_sky_position(Tp_,          P_, e_, ω_, Tp_, b_, aRs_)
    xoc, yoc, zoc = planet_sky_position(Tp_ + P_ / 2, P_, e_, ω_, Tp_, b_, aRs_)
    @test zin > 0                                  # transit: planet in front
    @test zoc < 0                                  # occultation: planet behind
    # the sky projection really is degenerate between the two
    @test abs(xin - xoc) < 1e-10
    @test abs(abs(yin) - abs(yoc)) < 1e-10
    # so the gate, not the separation test, is what suppresses the occultation
    @test rm_signal_at_time(Tp_, P_, e_, ω_, Tp_, b_, aRs_,
                            0.005, 4000.0, 0.3) != 0.0
    @test rm_signal_at_time(Tp_ + P_ / 2, P_, e_, ω_, Tp_, b_, aRs_,
                            0.005, 4000.0, 0.3) == 0.0
    # eccentric case: ω away from π/2 so the conjunctions are asymmetric
    for ω_ecc in (0.0, π/3, 1.9π)
        _, _, z_tr = planet_sky_position(Tp_, P_, 0.3, ω_ecc, Tp_, b_, aRs_)
        @test rm_signal_at_time(Tp_, P_, 0.3, ω_ecc, Tp_, b_, aRs_,
                                0.005, 4000.0, 0.3) == (z_tr > 0 ?
              rm_signal_at_time(Tp_, P_, 0.3, ω_ecc, Tp_, b_, aRs_,
                                0.005, 4000.0, 0.3) : 0.0)
        @test (z_tr > 0) || rm_signal_at_time(Tp_, P_, 0.3, ω_ecc, Tp_,
                                b_, aRs_, 0.005, 4000.0, 0.3) == 0.0
    end

    # ----- ARoME CCF kernel (Boué+2013 Eq. 15) ----------------------
    let x = -0.4, y = 0.5, f = 0.0134, lam = deg2rad(33.0), βp = 6_700.0
        # No-rotation limit: σ₀ → β_p makes the prefactor exactly 1, and for
        # v_p ≪ the widths the exponential → 1, so Eq. 15 reduces to −f·v_p.
        for vs in (10.0, 100.0)
            @test rm_signal_arome(x, y, f, vs, lam, βp, βp) ≈
                  rm_signal(x, y, f, vs, lam)  rtol=1e-4
        end
        # Δflux ≤ 0 → no signal
        @test rm_signal_arome(x, y, 0.0, 26_200.0, lam, 15_890.0, βp) == 0.0
        # Sign flips with λ → λ+π, exactly as the flux-weighted form does
        @test rm_signal_arome(x, y, f, 26_200.0, 0.0, 15_890.0, βp) ≈
              -rm_signal_arome(x, y, f, 26_200.0, π, 15_890.0, βp)  atol=1e-9
        # The physical content: a Gaussian CCF fit recovers MORE than the
        # flux-weighted mean near line centre and LESS far out in the wings.
        σ0 = 15_890.0; vs = 26_200.0
        near = rm_signal_arome(-0.05, y, f, vs, lam, σ0, βp) /
               rm_signal(-0.05, y, f, vs, lam)
        limb = rm_signal_arome(-0.85, y, f, vs, lam, σ0, βp) /
               rm_signal(-0.85, y, f, vs, lam)
        @test near > 1.5
        @test limb < 1.0
        @test near > limb                      # monotone suppression outward
    end

    # ----- Layout extension: RM source adds slots -------------------
    # Build a Params with a single RVPM_RM planet.
    n = 12
    rng = MersenneTwister(0)
    bjd  = sort(collect(range(0.0, 30.0; length=n))) .+ 2_450_000.0
    rv_data   = randn(rng, n)
    rv_err = fill(1.0, n)
    rv_inst = ones(Int, n)
    t_phot = collect(range(0.0, 30.0; length=200)) .+ 2_450_000.0
    flux   = fill(1.0, length(t_phot))
    fl_err = fill(1e-3, length(t_phot))
    pm_inst = ones(Int, length(t_phot))

    data = Data(; t_rv=bjd, rv=rv_data, rv_err=rv_err, rv_inst=rv_inst,
                  t_phot=t_phot, flux=flux, flux_err=fl_err,
                  phot_inst=pm_inst)
    ic = InstrumentConfig(rv=["HARPS"], pm=["TESS"])

    params = Params(; max_kplanet=1, planet_modes=[RVPM_RM],
                      instruments=ic, data=data,
                      parametrization=ParametrizationConfig(time=:Tp),
                      M_s=1.0, R_s=1.0, stability=:none)

    # λ_k1 and v_sin_i_star should both be in the layout
    @test haskey(params.layout.name_to_idx, "lambda_k1")
    @test haskey(params.layout.name_to_idx, "v_sin_i_star")
    @test params.layout.systemic.v_sin_i_star > 0

    # Default priors must be populated
    @test haskey(params.config.priors, "lambda_k1")
    @test haskey(params.config.priors, "v_sin_i_star")

    # ----- Likelihood evaluates ---------------------------------------
    target = NereusTarget(params, data; unconstrained=false)
    theta  = Theta{Float64}(params)
    # Populate a sensible-orbit theta (uninitialized → P=0, would blow
    # up Kepler). Set values manually for the RV-relevant params.
    idx = params.layout.name_to_idx
    theta.values[idx["P_k1"]]      = 3.5
    theta.values[idx["K_k1"]]      = 50.0
    theta.values[idx["sesinw_k1"]] = 0.0
    theta.values[idx["secosw_k1"]] = 0.0
    theta.values[idx["Tp_k1"]]     = 2_450_000.0
    theta.values[idx["b_k1"]]      = 0.3
    theta.values[idx["rr_k1"]]     = 0.1
    theta.values[idx["gamma_HARPS"]] = 0.0
    theta.values[idx["sigma_HARPS"]] = 1.0
    theta.values[idx["q1_TESS"]]   = 0.3
    theta.values[idx["q2_TESS"]]   = 0.2
    theta.values[idx["lambda_k1"]] = 0.0

    # v_sini=0 → RM contribution = 0 even though n_rm > 0
    theta.values[idx["v_sin_i_star"]] = 0.0
    ll_zero = rv_log_likelihood(theta, data)
    @test isfinite(ll_zero)

    # v_sini=5 km/s → RM contributes
    theta.values[idx["v_sin_i_star"]] = 5000.0
    theta.values[idx["lambda_k1"]]    = 0.3
    ll_with_rm = rv_log_likelihood(theta, data)
    @test isfinite(ll_with_rm)

    # Non-RM planet (RVPM) shouldn't have a lambda slot
    params_no_rm = Params(; max_kplanet=1, planet_modes=[RVPM],
                            instruments=ic, data=data,
                            parametrization=ParametrizationConfig(time=:Tp),
                            M_s=1.0, R_s=1.0, stability=:none)
    @test !haskey(params_no_rm.layout.name_to_idx, "lambda_k1")
    @test !haskey(params_no_rm.layout.name_to_idx, "v_sin_i_star")
end

@testset "Reloaded RM (Cegla+ 2016)" begin
    # ----- rm_reloaded_signal sanity --------------------------------
    # No transit (Δflux=0) → zero
    @test rm_reloaded_signal(0.0, 0.3, 0.1, 0.3, 0.2, 4000.0, 0.0, 0.0) == 0.0

    # Mid-transit (x=0), λ=0: by symmetry ⟨v_los⟩ = 0 → ΔRV = 0
    @test isapprox(rm_reloaded_signal(0.0, 0.3, 0.1, 0.3, 0.2,
                                       4000.0, 0.0, 0.01),
                   0.0; atol=1e-10)

    # Approaching limb (x=-0.7): planet on blue half → ΔRV > 0
    rm_blue = rm_reloaded_signal(-0.7, 0.2, 0.1, 0.3, 0.2,
                                   5000.0, 0.0, 0.01)
    rm_red  = rm_reloaded_signal( 0.7, 0.2, 0.1, 0.3, 0.2,
                                   5000.0, 0.0, 0.01)
    @test rm_blue > 0
    @test rm_red  < 0
    @test isapprox(rm_blue, -rm_red; atol=1e-9)

    # Sign-flip under λ → λ+π
    @test isapprox(rm_reloaded_signal(0.5, 0.1, 0.08, 0.3, 0.2,
                                        5000.0, 0.0, 0.01),
                   -rm_reloaded_signal(0.5, 0.1, 0.08, 0.3, 0.2,
                                         5000.0, π, 0.01);
                   atol=1e-9)

    # ----- Small-rr limit agrees with Hirano within a few percent ----
    # For rr=0.02, Reloaded should agree with Hirano to <2%.
    x, y = -0.6, 0.2
    rr  = 0.02
    u1, u2 = 0.3, 0.2
    v_sini = 6000.0
    λ = 0.4
    Δflux = 0.01  # arbitrary but consistent
    rm_h = rm_signal(x, y, Δflux, v_sini, λ)
    rm_c = rm_reloaded_signal(x, y, rr, u1, u2, v_sini, λ, Δflux)
    @test abs(rm_h - rm_c) / abs(rm_h) < 0.02

    # ----- Layout: RM-R variant uses same slots as RM -----------------
    n = 12
    rng = MersenneTwister(1)
    bjd  = sort(collect(range(0.0, 30.0; length=n))) .+ 2_450_000.0
    rv_data = randn(rng, n)
    rv_err  = fill(1.0, n)
    rv_inst = ones(Int, n)
    t_phot  = collect(range(0.0, 30.0; length=200)) .+ 2_450_000.0
    flux    = fill(1.0, length(t_phot))
    fl_err  = fill(1e-3, length(t_phot))
    pm_inst = ones(Int, length(t_phot))

    data = Data(; t_rv=bjd, rv=rv_data, rv_err=rv_err, rv_inst=rv_inst,
                  t_phot=t_phot, flux=flux, flux_err=fl_err,
                  phot_inst=pm_inst)
    ic = InstrumentConfig(rv=["HARPS"], pm=["TESS"])

    params = Params(; max_kplanet=1, planet_modes=[RVPM_RM_R],
                      instruments=ic, data=data,
                      parametrization=ParametrizationConfig(time=:Tp),
                      M_s=1.0, R_s=1.0, stability=:none)

    # Same slots as RM-A
    @test haskey(params.layout.name_to_idx, "lambda_k1")
    @test haskey(params.layout.name_to_idx, "v_sin_i_star")
    @test params.layout.systemic.v_sin_i_star > 0
    @test has_rm_r(RVPM_RM_R)
    @test has_any_rm(RVPM_RM_R)
    @test !has_rm(RVPM_RM_R)
    @test !has_rm_r(RVPM_RM)
    @test has_any_rm(RVPM_RM)

    # ----- Likelihood eval (state.is_reloaded[1] true) ---------------
    theta = Theta{Float64}(params)
    idx = params.layout.name_to_idx
    theta.values[idx["P_k1"]]      = 3.5
    theta.values[idx["K_k1"]]      = 50.0
    theta.values[idx["sesinw_k1"]] = 0.0
    theta.values[idx["secosw_k1"]] = 0.0
    theta.values[idx["Tp_k1"]]     = 2_450_000.0
    theta.values[idx["b_k1"]]      = 0.3
    theta.values[idx["rr_k1"]]     = 0.1
    theta.values[idx["gamma_HARPS"]] = 0.0
    theta.values[idx["sigma_HARPS"]] = 1.0
    theta.values[idx["q1_TESS"]]   = 0.3
    theta.values[idx["q2_TESS"]]   = 0.2
    theta.values[idx["lambda_k1"]] = 0.0
    theta.values[idx["v_sin_i_star"]] = 5000.0

    ll_r = rv_log_likelihood(theta, data)
    @test isfinite(ll_r)

    # Decode and verify is_reloaded flag
    p_idx = planet_indices(theta)
    Ps = [planet_P(theta, 1)]
    n_rm, st = Nereus._decode_rm_state(theta, p_idx, Ps)
    @test n_rm == 1
    @test st.is_reloaded == [true]

    # ----- RM-A counterpart: is_reloaded[1] false ---------------------
    params_a = Params(; max_kplanet=1, planet_modes=[RVPM_RM],
                        instruments=ic, data=data,
                        parametrization=ParametrizationConfig(time=:Tp),
                        M_s=1.0, R_s=1.0, stability=:none)
    theta_a = Theta{Float64}(params_a)
    idx_a = params_a.layout.name_to_idx
    for k in keys(idx)
        haskey(idx_a, k) || continue
        theta_a.values[idx_a[k]] = theta.values[idx[k]]
    end
    n_rm_a, st_a = Nereus._decode_rm_state(theta_a, planet_indices(theta_a),
                                              [planet_P(theta_a, 1)])
    @test n_rm_a == 1
    @test st_a.is_reloaded == [false]
end

# The photometric likelihood takes sky positions from per-call orbit constants
# (`_sky_orbit`, `_sky_position`, `_sky_separation_signed(o, t)`) instead of
# `planet_sky_position` and `sky_separation`: the same quantities, with the true
# anomaly from one sincos(E) by the half-angle identity, and Kepler's equation
# skipped when e is a fixed zero.
@testset "sky position from per-call orbit constants" begin
    _sky_orbit, _sky_position = Nereus._sky_orbit, Nereus._sky_position
    sep = Nereus._sky_separation_signed
    rng = MersenneTwister(11)
    worst = Dict("circular" => 0.0, "e < 0.95" => 0.0, "0.95 <= e <= 0.999" => 0.0)
    n_flip = 0
    for k in 1:3000
        cls = ("circular", "e < 0.95", "0.95 <= e <= 0.999")[k % 3 + 1]
        e = cls == "circular" ? 0.0 : cls == "e < 0.95" ? 0.95 * rand(rng) :
            0.95 + 0.049 * rand(rng)
        P  = exp(log(0.5) + rand(rng) * log(400))
        ω  = 2π * rand(rng) - π
        aR = 2 + 60 * rand(rng)
        b  = 1.2 * rand(rng)
        Tp = 2_459_000.0 + 1000 * rand(rng)
        o = _sky_orbit(P, e, ω, Tp, b, aR)
        @test o.circ == (e == 0)
        for _ in 1:20
            t = Tp + P * 100 * (rand(rng) - 0.5)        # ±50 orbits from Tp
            d = maximum(abs.(planet_sky_position(t, P, e, ω, Tp, b, aR) .- _sky_position(o, t)))
            z0, s0 = sep(t, P, e, ω, Tp, b, aR)
            z1, s1 = sep(o, t)
            worst[cls] = max(worst[cls], d / aR, abs(z0 - z1) / aR)
            n_flip += abs(s0) > 1e-9 && sign(s0) != sign(s1)
        end
    end
    @info "sky position vs planet_sky_position / sky_separation, max |Δ| / (a/R*)" worst
    # Rounding, ~1e-14. Above e = 0.95 cos f and r are ill-conditioned near
    # apoastron in both forms; they still agree far inside the Kepler solver's
    # own 1e-10 tolerance, which sets the accuracy of either.
    @test worst["circular"] < 1e-13 && worst["e < 0.95"] < 1e-13
    @test worst["0.95 <= e <= 0.999"] < 1e-10
    @test n_flip == 0

    # The true anomaly's e: clamped into [0, 1) as `true_anomaly` does, which
    # leaves every e < 1 alone (it was clamped to 0.9999), unless the caller
    # passes its own (the workspace refresh of an ordinary planet).
    @test _sky_orbit(3.0, 0.99995, 0.3, 0.0, 0.2, 400.0).ef == 0.99995
    @test _sky_orbit(3.0, 1.2, 0.3, 0.0, 0.2, 400.0).ef == prevfloat(1.0)
    @test _sky_orbit(3.0, -0.1, 0.3, 0.0, 0.2, 400.0).ef == 0.0
    @test _sky_orbit(3.0, 0.99995, 0.3, 0.0, 0.2, 400.0, 0.99995).ef == 0.99995
    # Kepler's equation is skipped only for an e that is zero and fixed.
    @test _sky_orbit(3.0, ForwardDiff.Dual(0.0, 0.0), 0.3, 0.0, 0.2, 8.0).circ
    @test !_sky_orbit(3.0, ForwardDiff.Dual(0.0, 1.0), 0.3, 0.0, 0.2, 8.0).circ
    @test !_sky_orbit(3.0, 1e-300, 0.3, 0.0, 0.2, 8.0).circ

    # ForwardDiff through the orbit constants: the derivatives of the old routine.
    ts = collect(range(0.6, 0.9; length = 50))
    for (e, free_e) in ((0.3, true), (0.0, false))
        x0 = [2.83, e, 0.7, 0.05, 0.6, 7.5]                # P, e, ω, Tp, b, a/R*
        sel = free_e ? (1:6) : [1, 3, 4, 5, 6]
        function sum_pos(v, f)
            x = convert(Vector{eltype(v)}, x0); x[sel] = v
            return sum(sum(f(t, x...) .* (1, 2, 3)) for t in ts)
        end
        lib(t, P, e, ω, Tp, b, aR) = planet_sky_position(t, P, e, ω, Tp, b, aR)
        hoisted(t, P, e, ω, Tp, b, aR) = _sky_position(_sky_orbit(P, e, ω, Tp, b, aR), t)
        g_lib = ForwardDiff.gradient(v -> sum_pos(v, lib), x0[sel])
        g_new = ForwardDiff.gradient(v -> sum_pos(v, hoisted), x0[sel])
        @test g_new ≈ g_lib rtol = 1e-10
    end
end
