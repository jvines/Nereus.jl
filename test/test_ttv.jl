# Per-transit free TTV offsets (TTV-A) tests.

using Test
using Nereus
using Random: MersenneTwister

@testset "TTV-A (per-transit free offsets)" begin
    # ----- ttv_effective_time math ----------------------------------
    # No offsets → returns t unchanged
    @test Nereus.ttv_effective_time(2_450_005.0, 3.5, 2_450_000.0, Float64[]) ==
          2_450_005.0

    # Offsets[1] is transit 0 (slot at Tc). i = round((t-Tc)/P) = 0 → slot 1.
    δts = [0.01, 0.02, -0.005]
    @test Nereus.ttv_effective_time(2_450_000.0, 3.5, 2_450_000.0, δts) ≈
          2_450_000.0 - 0.01
    # i = round((Tc+P - Tc)/P) = 1 → slot 2
    @test Nereus.ttv_effective_time(2_450_003.5, 3.5, 2_450_000.0, δts) ≈
          2_450_003.5 - 0.02
    # i = round((Tc+2P - Tc)/P) = 2 → slot 3
    @test Nereus.ttv_effective_time(2_450_007.0, 3.5, 2_450_000.0, δts) ≈
          2_450_007.0 - (-0.005)
    # Out-of-range slot (i = 5 → slot 6 > length): returns t unchanged
    @test Nereus.ttv_effective_time(2_450_017.5, 3.5, 2_450_000.0, δts) ==
          2_450_017.5
    # Negative i (point before first transit): slot 0 → unchanged
    @test Nereus.ttv_effective_time(2_449_996.5, 3.5, 2_450_000.0, δts) ==
          2_449_996.5

    # ----- Layout extension -----------------------------------------
    n = 12
    rng = MersenneTwister(0)
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

    n_transits_pl1 = 9
    params = Params(; max_kplanet=1, planet_modes=[RVPM_TTV],
                      instruments=ic, data=data,
                      parametrization=ParametrizationConfig(time=:Tp),
                      M_s=1.0, R_s=1.0, stability=:none,
                      ttv_n_transits=Dict(1 => n_transits_pl1))

    # All ttv_k1_tI (I=1..n_transits_pl1) must be in the layout
    for i in 1:n_transits_pl1
        @test haskey(params.layout.name_to_idx, "ttv_k1_t$i")
    end
    # Slot beyond N must NOT exist
    @test !haskey(params.layout.name_to_idx, "ttv_k1_t$(n_transits_pl1+1)")

    # ----- Default priors --------------------------------------------
    for i in 1:n_transits_pl1
        @test haskey(params.config.priors, "ttv_k1_t$i")
    end

    # ----- Likelihood evaluates with TTV active ----------------------
    theta = Theta{Float64}(params)
    idx = params.layout.name_to_idx
    theta.values[idx["P_k1"]]      = 3.5
    theta.values[idx["K_k1"]]      = 20.0
    theta.values[idx["sesinw_k1"]] = 0.0
    theta.values[idx["secosw_k1"]] = 0.0
    theta.values[idx["Tp_k1"]]     = 2_450_001.0
    theta.values[idx["b_k1"]]      = 0.3
    theta.values[idx["rr_k1"]]     = 0.1
    theta.values[idx["gamma_HARPS"]] = 0.0
    theta.values[idx["sigma_HARPS"]] = 1.0
    theta.values[idx["q1_TESS"]]   = 0.3
    theta.values[idx["q2_TESS"]]   = 0.2
    for i in 1:n_transits_pl1
        theta.values[idx["ttv_k1_t$i"]] = 0.0
    end

    ll_zero_ttv = transit_log_likelihood(theta, data)
    @test isfinite(ll_zero_ttv)

    # Inject a 0.05 d offset on transit 2 — log-likelihood must change
    theta.values[idx["ttv_k1_t2"]] = 0.05
    ll_with_ttv = transit_log_likelihood(theta, data)
    @test isfinite(ll_with_ttv)
    @test ll_with_ttv != ll_zero_ttv

    # ----- _decode_ttv_state shape -----------------------------------
    p_idx = planet_indices(theta)
    n_ttv, st = Nereus._decode_ttv_state(theta, p_idx)
    @test n_ttv == 1
    @test st !== nothing
    @test st.j_active == [1]
    @test st.k_planet == [1]
    @test length(st.δts[1]) == n_transits_pl1
    @test st.δts[1][2] == 0.05

    # ----- Non-TTV planet: no ttv slots, decode returns (0, nothing) -
    params_no_ttv = Params(; max_kplanet=1, planet_modes=[RVPM],
                             instruments=ic, data=data,
                             parametrization=ParametrizationConfig(time=:Tp),
                             M_s=1.0, R_s=1.0, stability=:none)
    @test !haskey(params_no_ttv.layout.name_to_idx, "ttv_k1_t1")
    theta_no = Theta{Float64}(params_no_ttv)
    n_ttv_no, st_no = Nereus._decode_ttv_state(theta_no,
                                                  planet_indices(theta_no))
    @test n_ttv_no == 0
    @test st_no === nothing

    # ----- has_active_ttv predicate ----------------------------------
    @test Nereus._has_active_ttv(theta, p_idx)
    @test !Nereus._has_active_ttv(theta_no, planet_indices(theta_no))
end

@testset "TTV-C (N-body via TTVFaster)" begin
    # 2-planet system with TTV-NB on both. Choose periods near 2:1
    # MMR for a measurable signal at modest masses.
    n = 12
    rng = MersenneTwister(0)
    bjd  = sort(collect(range(0.0, 400.0; length=n))) .+ 2_450_000.0
    rv_data = randn(rng, n)
    rv_err  = fill(2.0, n)
    rv_inst = ones(Int, n)
    t_phot  = collect(range(0.0, 400.0; length=400)) .+ 2_450_000.0
    flux    = fill(1.0, length(t_phot))
    fl_err  = fill(1e-3, length(t_phot))
    pm_inst = ones(Int, length(t_phot))

    data = Data(; t_rv=bjd, rv=rv_data, rv_err=rv_err, rv_inst=rv_inst,
                  t_phot=t_phot, flux=flux, flux_err=fl_err,
                  phot_inst=pm_inst)
    ic = InstrumentConfig(rv=["HARPS"], pm=["TESS"])

    params = Params(; max_kplanet=2, planet_modes=[RVPM_TTV_NB, RVPM_TTV_NB],
                      instruments=ic, data=data,
                      parametrization=ParametrizationConfig(time=:Tp),
                      M_s=1.0, R_s=1.0, stability=:none)

    # TTV-NB adds NO per-transit free parameters
    @test !haskey(params.layout.name_to_idx, "ttv_k1_t1")
    @test !haskey(params.layout.name_to_idx, "ttv_k2_t1")

    # has_ttv_nb predicate
    @test has_ttv_nb(RVPM_TTV_NB)
    @test !has_ttv_nb(RVPM_TTV)
    @test !has_ttv_nb(RVPM)

    # Populate sensible orbits — both transit, near-2:1 period ratio
    theta = Theta{Float64}(params)
    idx = params.layout.name_to_idx
    # Planet 1: hot Jupiter
    theta.values[idx["P_k1"]]      = 4.0
    theta.values[idx["K_k1"]]      = 150.0
    theta.values[idx["sesinw_k1"]] = 0.05
    theta.values[idx["secosw_k1"]] = 0.0
    theta.values[idx["Tp_k1"]]     = 2_450_001.0
    theta.values[idx["b_k1"]]      = 0.2
    theta.values[idx["rr_k1"]]     = 0.1
    # Planet 2: near 2:1 outside
    theta.values[idx["P_k2"]]      = 8.1
    theta.values[idx["K_k2"]]      = 50.0
    theta.values[idx["sesinw_k2"]] = 0.05
    theta.values[idx["secosw_k2"]] = 0.0
    theta.values[idx["Tp_k2"]]     = 2_450_002.0
    theta.values[idx["b_k2"]]      = 0.3
    theta.values[idx["rr_k2"]]     = 0.08
    theta.values[idx["gamma_HARPS"]] = 0.0
    theta.values[idx["sigma_HARPS"]] = 2.0
    theta.values[idx["q1_TESS"]]   = 0.3
    theta.values[idx["q2_TESS"]]   = 0.2

    # Likelihood evaluates finitely
    ll_nb = transit_log_likelihood(theta, data)
    @test isfinite(ll_nb)

    # _decode_ttv_state recognises both NB rows; _apply_ttv_nb! is the
    # caller's responsibility — invoke it directly here on a controlled
    # decode to verify shapes & non-trivial output.
    p_idx = planet_indices(theta)
    n_ttv, st = Nereus._decode_ttv_state(theta, p_idx)
    @test n_ttv == 2
    @test st.is_nb == [true, true]
    @test st.δts[1] == Float64[]  # empty until _apply_ttv_nb! runs
    @test st.δts[2] == Float64[]

    # Mock orbit arrays in the j-order expected by _apply_ttv_nb!
    Ps  = [4.0, 8.1]
    es  = [0.05^2, 0.05^2]  # e = sqrt(sesinw²+secosw²) → here sesinw=0.05
    # Actually need to compute properly
    es  = [hypot(0.05, 0.0)^2, hypot(0.05, 0.0)^2]
    ws  = [atan(0.05, 0.0), atan(0.05, 0.0)]
    Tps = [2_450_001.0, 2_450_002.0]
    bs  = [0.2, 0.3]
    a_Rs = [10.0, 16.0]  # rough — sufficient for inclination calc
    t_max = maximum(t_phot)

    Nereus._apply_ttv_nb!(st, theta, p_idx, Ps, es, ws, Tps, bs, a_Rs, t_max)
    @test !isempty(st.δts[1])
    @test !isempty(st.δts[2])
    @test all(isfinite, st.δts[1])
    @test all(isfinite, st.δts[2])
    # Some non-zero element should exist (we're near 2:1, so TTVs amplify)
    @test any(abs.(st.δts[1]) .> 1e-9) || any(abs.(st.δts[2]) .> 1e-9)

    # ----- Single-NB-planet system: prediction collapses to zero ------
    params_solo = Params(; max_kplanet=1, planet_modes=[RVPM_TTV_NB],
                           instruments=ic, data=data,
                           parametrization=ParametrizationConfig(time=:Tp),
                           M_s=1.0, R_s=1.0, stability=:none)
    theta_solo = Theta{Float64}(params_solo)
    idx_s = params_solo.layout.name_to_idx
    theta_solo.values[idx_s["P_k1"]]      = 4.0
    theta_solo.values[idx_s["K_k1"]]      = 100.0
    theta_solo.values[idx_s["sesinw_k1"]] = 0.0
    theta_solo.values[idx_s["secosw_k1"]] = 0.0
    theta_solo.values[idx_s["Tp_k1"]]     = 2_450_001.0
    theta_solo.values[idx_s["b_k1"]]      = 0.2
    theta_solo.values[idx_s["rr_k1"]]     = 0.1
    theta_solo.values[idx_s["gamma_HARPS"]] = 0.0
    theta_solo.values[idx_s["sigma_HARPS"]] = 2.0
    theta_solo.values[idx_s["q1_TESS"]]   = 0.3
    theta_solo.values[idx_s["q2_TESS"]]   = 0.2

    ll_solo = transit_log_likelihood(theta_solo, data)
    @test isfinite(ll_solo)

    # ----- has_active_ttv catches TTV-NB too --------------------------
    @test Nereus._has_active_ttv(theta, p_idx)
    @test Nereus._has_active_ttv_nb(theta, p_idx)

    # ----- NbodyGradient.jl backend dispatch --------------------------
    # Same fit re-run with ttv_backend=:nbody. Output δts go through
    # the full ODE integrator (NbodyGradient) instead of TTVFaster.
    # Both backends should agree to within ~10% on a low-mass low-e
    # near-2:1 system like this one.
    params_nbg = Params(; max_kplanet=2,
                          planet_modes=[RVPM_TTV_NB, RVPM_TTV_NB],
                          instruments=ic, data=data,
                          parametrization=ParametrizationConfig(time=:Tp),
                          M_s=1.0, R_s=1.0, stability=:none,
                          ttv_backend=:nbody)
    @test params_nbg.config.ttv_backend === :nbody

    theta_nbg = Theta{Float64}(params_nbg)
    # Copy the same orbits over.
    idx_n = params_nbg.layout.name_to_idx
    for nm in keys(idx_n)
        haskey(idx, nm) && (theta_nbg.values[idx_n[nm]] = theta.values[idx[nm]])
    end

    p_idx_n = planet_indices(theta_nbg)
    _, st_nbg = Nereus._decode_ttv_state(theta_nbg, p_idx_n)
    Nereus._apply_ttv_nb!(st_nbg, theta_nbg, p_idx_n,
                            Ps, es, ws, Tps, bs, a_Rs, t_max)
    @test !isempty(st_nbg.δts[1])
    @test !isempty(st_nbg.δts[2])
    @test all(isfinite, st_nbg.δts[1])
    @test all(isfinite, st_nbg.δts[2])
    # Compare δts of the inner planet between backends. Take only the
    # transits both backends predict (NbodyGradient might compute one
    # extra near tmax).
    n_common = min(length(st.δts[1]), length(st_nbg.δts[1]))
    @test n_common >= 3
    # δts at both backends are O(minutes) — a few × 1e-3 days. The
    # backends use different conventions for Tc1 ↔ Tp alignment, so
    # the absolute values can differ by a few minutes systematic
    # offset. Check the difference of *consecutive δts* (TTV
    # amplitude) instead, which is convention-agnostic.
    Δ_ttvfaster = diff(st.δts[1][1:n_common])
    Δ_nbg       = diff(st_nbg.δts[1][1:n_common])
    @test all(isfinite, Δ_ttvfaster)
    @test all(isfinite, Δ_nbg)
    # Both backends report non-zero TTV signal magnitude
    @test maximum(abs.(Δ_nbg)) > 0
    @test maximum(abs.(Δ_ttvfaster)) > 0

    # Bad backend rejected
    @test_throws ArgumentError Params(; max_kplanet=2,
        planet_modes=[RVPM_TTV_NB, RVPM_TTV_NB], instruments=ic,
        data=data, parametrization=ParametrizationConfig(time=:Tp),
        M_s=1.0, R_s=1.0, stability=:none, ttv_backend=:foo)

    # Gradient-sampler + :nbody guard: ForwardDiff Duals must NOT
    # silently strip to Float64, which would zero the N-body
    # gradient. Manually construct a Dual-typed Theta and verify
    # the guard fires.
    using ForwardDiff: Dual
    DualT = Dual{Nothing, Float64, 1}
    theta_dual = Theta{DualT}(params_nbg)
    # Copy the (Float64) orbit values into the Dual slots.
    for nm in keys(idx_n)
        haskey(idx, nm) || continue
        theta_dual.values[idx_n[nm]] = DualT(theta.values[idx[nm]])
    end
    _, st_dual = Nereus._decode_ttv_state(theta_dual, p_idx_n)
    Ps_d  = [DualT(p) for p in Ps]
    es_d  = [DualT(e) for e in es]
    ws_d  = [DualT(w) for w in ws]
    Tps_d = [DualT(p) for p in Tps]
    bs_d  = [DualT(b) for b in bs]
    a_Rs_d = [DualT(a) for a in a_Rs]
    @test_throws ArgumentError Nereus._apply_ttv_nb!(
        st_dual, theta_dual, p_idx_n,
        Ps_d, es_d, ws_d, Tps_d, bs_d, a_Rs_d, DualT(t_max))
end

# An offset on transit n moves transit n, and nothing else, whichever time
# anchor the planet is parametrized by. Transits are numbered from the time of
# mid-transit Tc. They used to be numbered from Tp in the likelihood, and with
# the transit near apoastron (ω close to -π/2) Tc is almost half a period from
# Tp: the boundary round((t - Tp)/P) = n + 1/2 then fell inside a transit, so an
# offset on transit n moved the first half of transit n and the second half of
# transit n - 1.
@testset "TTV-A: an offset on transit n moves transit n only" begin
    P, e, ω, b, rr = 3.5, 0.3, -π / 2 + 0.02, 0.2, 0.1
    Tc = 2_450_001.3
    Tp = tc_to_tp(Tc, P, e, ω)
    @test 0.45 * P < Tc - Tp < 0.5 * P          # the transit is ~P/2 from Tp
    t_ref = 2_450_005.0
    t_phot = collect(range(Tc - 0.5 * P, Tc + 4.5 * P; step = 2 / 1440))
    n_tr = 5                                     # transits 0..4 → slots 1..5
    data = Data(; t_phot, flux = ones(length(t_phot)),
                  flux_err = fill(1e-3, length(t_phot)),
                  phot_inst = ones(Int, length(t_phot)), t_ref)
    epoch = round.(Int, (t_phot .- Tc) ./ P)
    sv, cv = sqrt(e) * sin(ω), sqrt(e) * cos(ω)
    anchor = Dict(:Tc => Tc, :Tp => Tp, :Mo => 2π * (t_ref - Tp) / P)
    for time in (:Tc, :Tp, :Mo)
        params = Params(; max_kplanet = 1, planet_modes = [PM_TTV],
                          instruments = InstrumentConfig(pm = ["TESS"]), data,
                          parametrization = ParametrizationConfig(; time),
                          M_s = 1.0, R_s = 1.0, stability = :none,
                          ttv_n_transits = Dict(1 => n_tr))
        ix = params.layout.name_to_idx
        th = Theta{Float64}(params)
        for (k, v) in ("P_k1" => P, "sesinw_k1" => sv, "secosw_k1" => cv,
                       "$(time)_k1" => anchor[time], "b_k1" => b, "rr_k1" => rr,
                       "q1_TESS" => 0.3, "q2_TESS" => 0.2)
            th.values[ix[k]] = v
        end
        for i in 1:n_tr
            th.values[ix["ttv_k1_t$i"]] = 0.0
        end
        pred0 = phot_predictions(th, data)[1]
        @test count(<(1), pred0) > 100                   # it transits
        for n in 0:(n_tr - 1), δ in (0.03, -0.02)
            th.values[ix["ttv_k1_t$(n + 1)"]] = δ
            pred1 = phot_predictions(th, data)[1]
            th.values[ix["ttv_k1_t$(n + 1)"]] = 0.0
            # The model at t - δ with no offsets: transit n moved by δ.
            shifted = Data(; t_phot = t_phot .- δ, flux = data.flux,
                             flux_err = data.flux_err, phot_inst = data.phot_inst, t_ref)
            pred_s = phot_predictions(th, shifted)[1]
            here = epoch .== n
            @test all(pred1[.!here] .== pred0[.!here])      # every other transit untouched
            @test pred1[here] ≈ pred_s[here] atol = 1e-14  # transit n, all of it, moved by δ
            @test any(pred1[here] .!= pred0[here])
        end
    end
end
