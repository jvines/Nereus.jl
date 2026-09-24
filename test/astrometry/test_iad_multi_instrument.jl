# Multi-instrument astrometry.
#
# Covers the four things the single-instrument design could not do:
#
#   1. Two intermediate-astrometry missions in one fit (Hipparcos IAD +
#      Gaia DR4 epoch astrometry), each tagged to its own instrument.
#   2. A per-instrument along-scan zero point, so a frame offset between
#      the two missions is absorbed instead of biasing the orbit.
#   3. Parallax and BOTH proper-motion components SHARED across missions
#      — the long-baseline lever that makes the joint fit worth doing.
#      Marginalising a free mu per mission would integrate that lever
#      away; the discrimination test below is what catches it.
#   4. Exact reduction: with one instrument the generalised code must
#      reproduce the previous 5x5 marginalisation BIT FOR BIT.

using Random: MersenneTwister
import LinearAlgebra

# ---------------------------------------------------------------------
# Shared synthetic scenario (the `iad_log_likelihood discriminates`
# fixture from test_iad_gost.jl, reproduced verbatim so the pinned
# goldens below stay comparable).
# ---------------------------------------------------------------------
function _multi_iad_scenario(; with_gaia::Bool)
    rng = MersenneTwister(42)
    M_pri = 1.0; M_sec = 0.05
    P_d = 5 * 365.25; e = 0.3; ω = 0.5; Ω_true = 1.2
    i_true = deg2rad(60); plx = 30.0; tp = 49000.0
    orb = Nereus.build_orbit(P_d, e, ω, Ω_true, i_true, M_pri, M_sec, tp, plx)
    n = 80
    t0 = jyear_to_mjd(1991.25)
    t_transits = collect(range(t0 - 365.25 * 1.4, t0 + 365.25 * 2.0, length = n))
    psi = 2π .* rand(rng, n)
    year_phase = 2π .* (t_transits .- t0) ./ 365.25
    parallax_factor = sin.(year_phase .- psi)
    abscissa = Vector{Float64}(undef, n); abscissa_err = fill(1.5, n)
    for j in 1:n
        Δra, Δdec = Nereus.star_reflex_offset(orb, t_transits[j], M_sec)
        # The parallactic wobble belongs in the abscissae — real ones have it,
        # and the likelihood now models it from the sampled `plx` instead of
        # marginalising a free ϖ that silently absorbed its absence.
        abscissa[j] = Nereus.along_scan_projection(Δra, Δdec, psi[j]) +
                      plx * parallax_factor[j] +
                      abscissa_err[j] * randn(rng)
    end
    iad = IADData(t = t_transits, abscissa = abscissa, abscissa_err = abscissa_err,
                  psi = psi, parallax_factor = parallax_factor,
                  pm_factor = (t_transits .- t0) ./ 365.25)
    kw = Dict{Symbol, Any}(:iad => iad)
    if with_gaia
        gost = GOSTData(t = collect(range(57000.0, 58000.0, length = 30)),
                        psi = mod.(2π .* (1:30) ./ 7, 2π),
                        parallax_factor = sin.(2π .* (1:30) ./ 365.25))
        kw[:gost] = gost
        cov = zeros(5, 5); for k in 1:5; cov[k, k] = (k <= 2 ? 0.05 : 0.02)^2; end
        # The published Gaia five-vector must be SELF-CONSISTENT with the
        # injected data, or it is not a constraint, it is a contradiction.
        # The synthetic IAD has no proper motion and no frame offset, so the
        # star's own vector is (0, 0, plx, 0, 0); what Gaia would publish is
        # that plus the orbit's induced five-parameter shift at its epoch.
        #
        # This was hardcoded (0.1, -0.2, 30.05, 1.5, -2.5). The orbit actually
        # induces (3.26, -0.40, 0.087, -3.12, -1.54) — the proper-motion terms
        # had the WRONG SIGN, asserting μ the abscissae contradict at σ=0.02.
        # The fit relieved ~48000 of χ² by tilting the orbit, so i = 30° beat
        # the true i = 60° by ~800 nats. The bit-for-bit pins never caught it
        # because they check exact values, not ordering — hence the
        # discrimination testset added below.
        Δq_true = gost_5param_fit(orb, gost, M_sec; t_ref = jyear_to_mjd(2016.0))
        kw[:gaia_dr3] = GaiaDR3Data(
            params = (Δq_true[1], Δq_true[2], plx + Δq_true[3],
                      Δq_true[4], Δq_true[5]),
            cov = cov, t_ref = jyear_to_mjd(2016.0))
    end
    data = Data(; t_rv = [49000.0, 49300.0], rv = [10.0, -10.0],
                rv_err = [1.0, 1.0], kw...)
    params = Params(max_kplanet = 1, planet_modes = [RVAS],
        instruments = InstrumentConfig(rv = ["X"]), data = data, stability = :none,
        M_s = M_pri, parametrization = ParametrizationConfig(mass = :a_driven),
        priors = Dict{String, PriorSpec}(
            "n_p" => FixedPrior(1.0), "a_k1" => LogUniformPrior(0.5, 100.0),
            "M_sec_k1" => LogUniformPrior(0.001, 0.5),
            "sesinw_k1" => UniformPrior(-1.0, 1.0),
            "secosw_k1" => UniformPrior(-1.0, 1.0), "Mo_k1" => UniformPrior(0.0, 2π),
            "inc_k1" => SinePrior(), "Omega_k1" => UniformPrior(0.0, 2π),
            "sigma_X" => LogUniformPrior(0.1, 10.0), "M_pri" => FixedPrior(M_pri)))
    theta = Theta(params)
    a_true = ((M_pri + M_sec) * (P_d / Nereus.KEPLER_YEAR_DAYS)^2)^(1/3)
    set_param!(theta, "n_p", 1); set_param!(theta, "a_k1", a_true)
    set_param!(theta, "M_sec_k1", M_sec)
    set_param!(theta, "sesinw_k1", sqrt(e) * sin(ω))
    set_param!(theta, "secosw_k1", sqrt(e) * cos(ω))
    set_param!(theta, "Mo_k1", 2π * (49150.0 - tp) / P_d)
    set_param!(theta, "inc_k1", i_true); set_param!(theta, "Omega_k1", Ω_true)
    set_param!(theta, "plx", plx); set_param!(theta, "gamma_X", 0.0)
    set_param!(theta, "sigma_X", 1.0)
    return data, theta, (; M_sec, i_true, Ω_true)
end

# The pinned log-likelihoods in this file are REGRESSION GOLDENS, compared to
# 1e-13 relative -- not bit for bit. They were recorded on macOS/aarch64; on
# Linux/x86_64, where CI runs, the same code lands an ulp away (4e-16 relative:
# libm and FMA contraction differ between the two), so `===` against a literal
# fails there for no reason in the code. Every change these pins exist to catch
# is far larger: the Kepler-year fix moved them 1.9e-5 relative, the GM_sun
# consolidation 1.1e-7. Reordered floating-point arithmetic (the reflex kernel,
# `_markley_sc`: 1e-16 to 1e-15) passes without a re-pin, as it should, and the
# measured cross-platform spread (4e-16) has 250x of headroom.
const PIN_RTOL = 1e-13

# Joint-path goldens. Named so the discrimination testset below and the
# pins above cannot drift apart silently.
#
# RE-PINNED 2026-09-23 (reflex kernel, GOST side): `gost_5param_fit` stopped
# calling `orbitsolve` per scan transit and now evaluates the orbit from the
# same hoisted Thiele-Innes constants `_iad_residuals!` uses
# (`_gost_5param_accumulate`, src/astrometry/projection.jl). The joint path
# reaches it twice — `_multi_iad_scenario` builds `Δq_true` with it, and
# `_iad_gaia_joint_log_likelihood` calls it per evaluation — so all four
# goldens below are exposed to the reordered multiply-adds. Three moved, by
# 1.4e-11, 3.4e-13 and 1.8e-12 absolute (4.0e-16, 1.7e-15 and 5.4e-16
# relative); JOINT_TRUTH rounds to the same Float64. The kernel is checked
# against `star_reflex_offset` independently in test_reflex_kernel.jl (worst
# relative 3.1e-14 over 10,000 orbit/epoch pairs) and `gost_5param_fit` against
# a verbatim copy of its own previous accumulation over 60 random scan
# geometries (worst relative 8.6e-16).
const JOINT_TRUTH = -141.2694442258537
const JOINT_3MSEC = -34001.345123468105
const JOINT_I30   = -204.91035261440564
const JOINT_OMEGA = -3328.2593093716064

@testset "astrometry — multi-instrument" begin

    # =================================================================
    # 1. Exact reduction of the n_inst == 1 path.
    #
    # RE-PINNED 2026-09-23 (_markley_sc): `_reflex_offset` stopped taking a
    # second `sincos(E)` after the Kepler solve and now uses the sine and cosine
    # Markley already computed for its own corrector (`_markley_sc`,
    # src/astrometry/projection.jl). E is returned BIT-IDENTICAL to
    # PlanetOrbits' `kepler_solver` -- pinned over 20,000 random (M, e) in
    # test_reflex_kernel.jl, 20000/20000 exact -- and only sin/cos differ, by
    # ~7 ulp, because they come from an angle-sum over the final correction
    # rather than a fresh libm call. On the residuals that is max |delta| =
    # 1.4e-14 mas against a Gaia along-scan sigma of 0.049 mas. Four pins moved
    # by 1-2 ulp: the two below and JOINT_3MSEC / JOINT_OMEGA above. Worth 12.3
    # us of a ~57 us likelihood evaluation (1.41x on the residual loop).
    #
    # RE-PINNED 2026-09-22 (reflex kernel): `_iad_residuals!` stopped calling
    # `orbitsolve` once per abscissa and now evaluates the orbit from its
    # Thiele-Innes constants, hoisted once per likelihood call
    # (`_reflex_kernel`, src/astrometry/projection.jl). Same Kepler solve --
    # PlanetOrbits' own, so E is bit-identical -- and an algebraic identity
    # after it, but the multiply-adds happen in a different order, so the low
    # bits move. Two of the four pins below shifted, by 2.2e-13 and 5.0e-14
    # absolute (1.1e-15 and 3.5e-16 relative); the other two round to the same
    # Float64. Agreement with `star_reflex_offset` is pinned independently over
    # 10,000 (orbit, epoch) pairs in test_reflex_kernel.jl, worst relative
    # 3.1e-14. NOTE the consequence the block comment below spells out: the
    # tracked reproduction artifacts were computed with the OLD accumulation
    # order and need regenerating.
    #
    # RE-PINNED 2026-09-22: `a_from_P` moved off the Julian year onto
    # KEPLER_YEAR_DAYS, so an orbit built from (a, M) finally has the period it
    # was given instead of one 1.89e-5 long (src/constants.jl). At the truth
    # parameters the value is unchanged to the last two digits — the scenario
    # builds its abscissae through the same conversion — and the off-truth pins
    # move by 1e-5 to 1e-4 relative, most at 3·M_sec, where the orbit is far
    # from the data and a phase shift costs the most.
    #
    # RE-PINNED 2026-09-20 when the parallax stopped being a marginalised
    # nuisance and became the sampled `astrom_plx(theta)` (see `_iad_n_q`).
    # The synthetic abscissae in `_multi_iad_scenario` also gained the
    # parallactic wobble they had always been missing. Both are intentional
    # model changes, so the old pins could not survive; the repo's tracked
    # reproduction artifacts were computed with the OLD likelihood and need
    # regenerating.
    #
    # These values were captured from the pre-generalisation
    # implementation and compared with `===` until 2026-09-23. That caught
    # reordered accumulations in the low bits -- and also failed on every
    # platform but the one the pins were recorded on (CI's Linux/x86_64 lands
    # an ulp away), and forced five re-pins in a week for arithmetic reorders
    # of 1e-16. They are now compared to PIN_RTOL (see its definition): a
    # change to the model or its constants still fails, a reorder does not.
    # =================================================================
    @testset "single instrument reproduces its pinned values" begin
        data, theta, tr = _multi_iad_scenario(with_gaia = false)
        @test iad_log_likelihood(theta, data) ≈ -141.44195364572727 rtol = PIN_RTOL
        set_param!(theta, "M_sec_k1", 3 * tr.M_sec)
        @test iad_log_likelihood(theta, data) ≈ -192.90983575750448 rtol = PIN_RTOL
        set_param!(theta, "M_sec_k1", tr.M_sec)
        set_param!(theta, "inc_k1", deg2rad(30))
        @test iad_log_likelihood(theta, data) ≈ -148.72611977105913 rtol = PIN_RTOL
        set_param!(theta, "inc_k1", tr.i_true)
        set_param!(theta, "Omega_k1", tr.Ω_true + 0.5)
        @test iad_log_likelihood(theta, data) ≈ -143.6809026397441 rtol = PIN_RTOL
    end

    @testset "IAD + Gaia DR3 joint path reproduces its pinned values" begin
        data, theta, tr = _multi_iad_scenario(with_gaia = true)
        @test iad_log_likelihood(theta, data) ≈ JOINT_TRUTH rtol = PIN_RTOL
        set_param!(theta, "M_sec_k1", 3 * tr.M_sec)
        @test iad_log_likelihood(theta, data) ≈ JOINT_3MSEC rtol = PIN_RTOL
        set_param!(theta, "M_sec_k1", tr.M_sec)
        set_param!(theta, "inc_k1", deg2rad(30))
        @test iad_log_likelihood(theta, data) ≈ JOINT_I30 rtol = PIN_RTOL
        set_param!(theta, "inc_k1", tr.i_true)
        set_param!(theta, "Omega_k1", tr.Ω_true + 0.5)
        @test iad_log_likelihood(theta, data) ≈ JOINT_OMEGA rtol = PIN_RTOL
    end

    # =================================================================
    # 1b. ORDERING, not just exact values.
    #
    # The pins above compare Float64s exactly and say nothing about
    # whether the truth is preferred. That gap let a joint path in which
    # i = 30° beat the true i = 60° by ~800 nats sit unnoticed: the pinned
    # numbers were "right" and the science was wrong. Assert the shape of
    # the likelihood surface, not only its values.
    # =================================================================
    @testset "joint path prefers the truth" begin
        data, theta, tr = _multi_iad_scenario(with_gaia = true)
        ll_truth = iad_log_likelihood(theta, data)
        for i_deg in (30, 45, 75)
            set_param!(theta, "inc_k1", deg2rad(i_deg))
            @test ll_truth > iad_log_likelihood(theta, data)
        end
        set_param!(theta, "inc_k1", tr.i_true)
        for f in (0.5, 2.0, 3.0)
            set_param!(theta, "M_sec_k1", f * tr.M_sec)
            @test ll_truth > iad_log_likelihood(theta, data)
        end
        set_param!(theta, "M_sec_k1", tr.M_sec)
        for dΩ in (-0.5, 0.5)
            set_param!(theta, "Omega_k1", tr.Ω_true + dΩ)
            @test ll_truth > iad_log_likelihood(theta, data)
        end
    end

    @testset "rank-deficient fallback reproduces its pinned value" begin
        iad = IADData(t = collect(48000.0:20.0:48200.0), abscissa = 0.3 .* sin.(1:11),
                      abscissa_err = fill(1.0, 11),
                      psi = collect(range(0, 2π, length = 11)))
        data = Data(t_rv = [49000.0, 49300.0], rv = [1.0, -1.0],
                    rv_err = [1.0, 1.0], iad = iad)
        params = Params(max_kplanet = 1, planet_modes = [RVAS],
            instruments = InstrumentConfig(rv = ["X"]), data = data,
            stability = :none, M_s = 1.0)
        theta = Theta(params)
        set_param!(theta, "n_p", 1); set_param!(theta, "P_k1", 365.25)
        set_param!(theta, "K_k1", 20.0)
        set_param!(theta, "sesinw_k1", 0.0); set_param!(theta, "secosw_k1", 0.0)
        set_param!(theta, "Mo_k1", 0.0); set_param!(theta, "inc_k1", deg2rad(60))
        set_param!(theta, "Omega_k1", deg2rad(30)); set_param!(theta, "plx", 25.0)
        set_param!(theta, "sigma_X", 5.0)
        # RE-PINNED 2026-09-23: this theta is K-driven, so M_sec comes from the
        # astrometric mass function, whose `_F_M_FACTOR` now uses the IAU GM_sun
        # (src/constants.jl) instead of 6.6743e-11 × 1.989e30 -- 3.0e-4 in GM,
        # 1.1e-7 in this value. With the old factor put back the previous pin
        # (-10.389529365014097) is reproduced bit for bit, so nothing else moved.
        @test iad_log_likelihood(theta, data) ≈ -10.38953053198153 rtol = PIN_RTOL
    end

    # =================================================================
    # 2. Container: instrument index + catalogue reference solution.
    # =================================================================
    @testset "IADData — instrument index defaults to single instrument" begin
        d = IADData(t = [48000.0, 48100.0], abscissa = [0.5, -0.3],
                    abscissa_err = [1.0, 1.2], psi = [0.0, π/4])
        @test d.inst == [1, 1]
        @test n_iad_inst(d) == 1
        @test d.ref_params == [(0.0, 0.0, 0.0, 0.0, 0.0)]
    end

    @testset "IADData — explicit instrument index and reference solutions" begin
        d = IADData(t = [48000.0, 48100.0, 57000.0, 57100.0],
                    abscissa = [0.5, -0.3, 0.1, 0.2],
                    abscissa_err = [1.0, 1.2, 0.1, 0.1],
                    psi = [0.0, π/4, 0.3, 1.1],
                    inst = [1, 1, 2, 2],
                    ref_params = [(0.0, 0.0, 35.25, 179.69, -138.40),
                                  (0.0, 0.0, 0.0, 0.0, 0.0)],
                    abscissa_kind = [:residual, :absolute])
        @test n_iad(d) == 4
        @test n_iad_inst(d) == 2
        @test d.inst == [1, 1, 2, 2]
        @test d.ref_params[1][3] == 35.25
        @test d.abscissa_kind == [:residual, :absolute]
    end

    @testset "IADData — instrument index validation" begin
        base = (t = [0.0, 1.0], abscissa = [0.0, 0.0], abscissa_err = [1.0, 1.0],
                psi = [0.0, 0.0])
        # Each guard is asserted ON ITS MESSAGE, not just on ArgumentError.
        # Several of them fire on overlapping inputs, and a bare
        # `@test_throws ArgumentError` lets a case drift onto a neighbouring
        # guard and go on passing after the one it was written for is gone.
        @test_throws ArgumentError IADData(; base..., inst = [1])
        @test_throws "1-based positive integers" IADData(; base..., inst = [0, 1])
        # A gap in the numbering means an instrument with no data and two
        # unconstrained zero-point columns — reject it rather than fall
        # back to a rank-deficient design at every likelihood call.
        @test_throws "gap-free" IADData(; base..., inst = [1, 3])
        # Two instruments over two transits: one transit each, which cannot
        # pin either instrument's two zero-point columns.
        @test_throws "at least 2 transits" IADData(; base..., inst = [1, 2])
        # :absolute contradicts a non-zero reference solution.
        @test_throws "abscissa_kind = :residual" IADData(; base...,
                                           ref_params = [(0.0, 0.0, 30.0, 1.0, 2.0)])

        # Four transits so the min-2 guard is satisfied and the NEXT guards
        # are the ones under test.
        base4 = (t = [0.0, 1.0, 2.0, 3.0], abscissa = zeros(4),
                 abscissa_err = ones(4), psi = [0.0, 0.3, 0.6, 0.9],
                 inst = [1, 1, 2, 2])
        @test_throws "per instrument, not per transit" IADData(; base4...,
                                           ref_params = [(0.0, 0.0, 0.0, 0.0, 0.0)])
        # The one that matters: an instrument storing residuals it cannot
        # reconstruct, combined with anything else. This is exactly the
        # container `_parse_van_leeuwen_iad` emits for a 7p/9p/VIM source,
        # and the parser's warning promises merge_iad will refuse it.
        @test_throws "all-zero ref_params" IADData(; base4...,
                                           abscissa_kind = [:residual, :absolute])
        @test_throws "all-zero ref_params" merge_iad(
            IADData(t = [0.0, 1.0], abscissa = [0.0, 0.0],
                    abscissa_err = [1.0, 1.0], psi = [0.0, 0.3],
                    abscissa_kind = :residual),
            IADData(t = [10.0, 11.0], abscissa = [0.0, 0.0],
                    abscissa_err = [1.0, 1.0], psi = [0.5, 0.8]))
    end

    @testset "merge_iad — concatenates sources into consecutive instruments" begin
        hip = IADData(t = [48000.0, 48100.0], abscissa = [0.5, -0.3],
                      abscissa_err = [1.0, 1.2], psi = [0.0, π/4],
                      parallax_factor = [0.3, 0.4], pm_factor = [-1.0, 0.5],
                      ref_params = [(0.0, 0.0, 35.25, 179.69, -138.40)],
                      abscissa_kind = :residual)
        gaia = IADData(t = [57000.0, 57100.0, 57200.0], abscissa = [0.1, 0.2, 0.3],
                       abscissa_err = [0.1, 0.1, 0.1], psi = [0.3, 1.1, 2.0],
                       parallax_factor = [0.2, -0.1, 0.5], pm_factor = [-0.5, 0.0, 0.5])
        m = merge_iad(hip, gaia)
        @test n_iad(m) == 5
        @test n_iad_inst(m) == 2
        @test m.inst == [1, 1, 2, 2, 2]
        @test m.t == vcat(hip.t, gaia.t)
        @test m.abscissa == vcat(hip.abscissa, gaia.abscissa)
        @test m.psi == vcat(hip.psi, gaia.psi)
        @test m.parallax_factor == vcat(hip.parallax_factor, gaia.parallax_factor)
        @test m.pm_factor == vcat(hip.pm_factor, gaia.pm_factor)
        @test m.ref_params == [(0.0, 0.0, 35.25, 179.69, -138.40),
                               (0.0, 0.0, 0.0, 0.0, 0.0)]
        @test m.abscissa_kind == [:residual, :absolute]
        # Merging a single source is the identity on the instrument axis.
        @test n_iad_inst(merge_iad(hip)) == 1

        # ...and splitting puts each mission back the way it went in.
        back1 = iad_for_inst(m, 1)
        back2 = iad_for_inst(m, 2)
        @test n_iad_inst(back1) == 1 && n_iad_inst(back2) == 1
        @test back1.t == hip.t && back1.abscissa == hip.abscissa
        @test back1.ref_params == hip.ref_params
        @test back1.abscissa_kind == hip.abscissa_kind
        @test back2.t == gaia.t && back2.abscissa == gaia.abscissa
        @test_throws ArgumentError iad_for_inst(m, 3)
        @test_throws ArgumentError iad_for_inst(m, 0)
    end

    @testset "solution ladder refuses multi-instrument input" begin
        # The 5/7/9-parameter ladder is a per-CATALOGUE question and its
        # columns 1-2 are ONE shared zero point. Handed two missions it used
        # to fit one frame to both without complaint, and since its product
        # is Bayes factors, an unmodelled frame offset reads as curvature.
        a = IADData(t = collect(48000.0:20.0:48200.0), abscissa = 0.3 .* sin.(1:11),
                    abscissa_err = fill(1.0, 11),
                    psi = collect(range(0, 2π, length = 11)),
                    parallax_factor = cos.(1:11), pm_factor = range(-1, 1, length = 11))
        b = IADData(t = collect(57000.0:20.0:57200.0), abscissa = 0.2 .* cos.(1:11),
                    abscissa_err = fill(0.5, 11),
                    psi = collect(range(0.3, 2π, length = 11)),
                    parallax_factor = sin.(1:11), pm_factor = range(-1, 1, length = 11))
        joined = merge_iad(a, b)
        @test_throws ArgumentError astrom_design(joined, 5)
        # Per mission it still works, which is the supported route.
        @test size(astrom_design(iad_for_inst(joined, 1), 5)) == (11, 5)
        @test size(astrom_design(iad_for_inst(joined, 2), 5)) == (11, 5)
    end

    # =================================================================
    # 3. The catalogue-reference reconstruction.
    #
    # Hipparcos abscissae are O-C against the catalogue five-parameter
    # solution; Gaia's are full abscissae. Sharing the parallax and the
    # proper motion across the two is only meaningful once both mean the
    # same thing, so the likelihood adds the catalogue solution back.
    # With ONE instrument that addition lands entirely in the column
    # span of the design matrix, which the flat-prior marginalisation
    # projects out — so it must not move the answer at all.
    # =================================================================
    @testset "reference reconstruction is a no-op for one instrument" begin
        data0, theta0, _ = _multi_iad_scenario(with_gaia = false)
        ll0 = iad_log_likelihood(theta0, data0)

        iad0 = data0.iad
        # A `:residual` instrument stores O−C against its catalogue solution,
        # so its abscissae do NOT contain the catalogue parallax — that sits
        # in `ref_params[3]`. Relabelling FULL abscissae as residuals while
        # leaving the parallax in them is not the same data, and now that the
        # parallax is modelled rather than marginalised the likelihood can
        # tell. Subtract it to build a genuine residual dataset.
        # The position and PM entries need no such treatment: they lie in the
        # span of the design and the marginalisation projects them out.
        ref_cat = (12.0, -7.0, 30.0, 179.69, -138.40)
        iad_ref = IADData(t = iad0.t,
                          abscissa = iad0.abscissa .-
                                     ref_cat[3] .* iad0.parallax_factor,
                          abscissa_err = iad0.abscissa_err, psi = iad0.psi,
                          parallax_factor = iad0.parallax_factor,
                          pm_factor = iad0.pm_factor,
                          ref_params = [ref_cat],
                          abscissa_kind = :residual)
        data_ref = Data(t_rv = data0.t_rv, rv = data0.rv, rv_err = data0.rv_err,
                        iad = iad_ref)
        params = Params(max_kplanet = 1, planet_modes = [RVAS],
            instruments = InstrumentConfig(rv = ["X"]), data = data_ref,
            stability = :none, M_s = 1.0,
            parametrization = ParametrizationConfig(mass = :a_driven),
            priors = theta0.params.config.priors)
        theta_ref = Theta(params)
        for nm in params.layout.unfrozen_names
            idx = get(Dict(theta0.params.layout.name_to_idx), nm, 0)
            idx == 0 || set_param!(theta_ref, nm, theta0.values[idx])
        end
        # Equal to within floating-point reassociation rather than `===`:
        # the parallax term now moves across the subtraction (taken off the
        # abscissae here, off the model there), so the two paths do the same
        # arithmetic in a different order. The invariant itself still holds —
        # declaring a reference solution cannot change the answer.
        @test iad_log_likelihood(theta_ref, data_ref) ≈ ll0 rtol = 1e-12
    end

    # =================================================================
    # 4. Per-instrument zero point.
    #
    # Shift instrument 2's whole frame by a constant (dra0, ddec0). That
    # shift lies in the span of instrument 2's OWN position columns, so a
    # correctly split design absorbs it exactly and the likelihood does
    # not move. With a single shared zero point it cannot be absorbed and
    # the answer changes — which is the bias this whole change removes.
    # =================================================================
    @testset "per-instrument zero point absorbs a frame offset" begin
        rng = MersenneTwister(7)
        n1, n2 = 40, 40
        t1 = collect(range(jyear_to_mjd(1990.0), jyear_to_mjd(1993.0), length = n1))
        t2 = collect(range(jyear_to_mjd(2015.0), jyear_to_mjd(2018.0), length = n2))
        psi1 = 2π .* rand(rng, n1); psi2 = 2π .* rand(rng, n2)
        e1 = jyear_to_mjd(1991.25); e2 = jyear_to_mjd(2016.5)
        pf1 = sin.(2π .* (t1 .- e1) ./ 365.25 .- psi1)
        pf2 = sin.(2π .* (t2 .- e2) ./ 365.25 .- psi2)
        pm1 = (t1 .- e1) ./ 365.25; pm2 = (t2 .- e2) ./ 365.25
        a1 = randn(rng, n1); a2 = randn(rng, n2)

        base = IADData(t = vcat(t1, t2), abscissa = vcat(a1, a2),
                       abscissa_err = fill(1.0, n1 + n2), psi = vcat(psi1, psi2),
                       parallax_factor = vcat(pf1, pf2), pm_factor = vcat(pm1, pm2),
                       inst = vcat(fill(1, n1), fill(2, n2)))
        # Same data, instrument 2 shifted by (dra0, ddec0) = (3.0, -2.0) mas.
        shift = vcat(zeros(n1), 3.0 .* sin.(psi2) .+ (-2.0) .* cos.(psi2))
        shifted = IADData(t = base.t, abscissa = base.abscissa .+ shift,
                          abscissa_err = base.abscissa_err, psi = base.psi,
                          parallax_factor = base.parallax_factor,
                          pm_factor = base.pm_factor, inst = base.inst)

        function _ll(iad)
            data = Data(t_rv = [55000.0, 55100.0], rv = [0.0, 0.0],
                        rv_err = [1.0, 1.0], iad = iad)
            params = Params(max_kplanet = 1, planet_modes = [RVAS],
                instruments = InstrumentConfig(rv = ["X"]), data = data,
                stability = :none, M_s = 1.0)
            theta = Theta(params)
            set_param!(theta, "n_p", 1); set_param!(theta, "P_k1", 3000.0)
            set_param!(theta, "K_k1", 30.0)
            set_param!(theta, "sesinw_k1", 0.0); set_param!(theta, "secosw_k1", 0.0)
            set_param!(theta, "Mo_k1", 0.4); set_param!(theta, "inc_k1", deg2rad(55))
            set_param!(theta, "Omega_k1", 0.9); set_param!(theta, "plx", 25.0)
            set_param!(theta, "sigma_X", 3.0)
            return iad_log_likelihood(theta, data)
        end
        @test _ll(base) ≈ _ll(shifted) rtol = 1e-9
    end

    # =================================================================
    # 5. The whole point: a shared proper motion across a 25-yr gap.
    #
    # Inject a companion far longer in period than either mission's own
    # baseline. Within one mission its reflex is degenerate with a linear
    # sky path and gets marginalised away, so neither mission alone can
    # say much about the mass. The signal lives in the proper-motion
    # DIFFERENCE between the two epochs, and only a decomposition that
    # shares mu across instruments can see it.
    #
    # If someone "generalises" this by giving each instrument its own
    # free 5-vector, the joint discrimination collapses to the sum of the
    # two single-mission ones and this test fails. That is its job.
    # =================================================================
    @testset "shared proper motion beats either mission alone" begin
        rng = MersenneTwister(11)
        M_pri = 1.0; M_sec = 0.08
        P_d = 90 * 365.25              # 90 yr: 25x the Hipparcos baseline
        e = 0.2; ω = 0.7; Ω = 1.0; inc = deg2rad(55); plx = 25.0
        tp = jyear_to_mjd(1970.0)
        orb = Nereus.build_orbit(P_d, e, ω, Ω, inc, M_pri, M_sec, tp, plx)

        n1, n2 = 60, 60
        t1 = collect(range(jyear_to_mjd(1989.9), jyear_to_mjd(1993.2), length = n1))
        t2 = collect(range(jyear_to_mjd(2014.6), jyear_to_mjd(2017.5), length = n2))
        psi1 = 2π .* rand(rng, n1); psi2 = 2π .* rand(rng, n2)
        e1 = jyear_to_mjd(1991.25); e2 = jyear_to_mjd(2016.0)
        pf1 = sin.(2π .* (t1 .- e1) ./ 365.25 .- psi1)
        pf2 = sin.(2π .* (t2 .- e2) ./ 365.25 .- psi2)
        pm1 = (t1 .- e1) ./ 365.25; pm2 = (t2 .- e2) ./ 365.25
        σ1 = fill(1.5, n1); σ2 = fill(0.2, n2)       # Gaia is the sharper mission

        # Parallactic wobble included: real abscissae carry it, and the
        # likelihood now models it from the sampled plx rather than
        # marginalising a free ϖ that hid its absence.
        function _absc(ts, psis, σs, pfs)
            out = Vector{Float64}(undef, length(ts))
            for j in eachindex(ts)
                Δra, Δdec = Nereus.star_reflex_offset(orb, ts[j], M_sec)
                out[j] = Nereus.along_scan_projection(Δra, Δdec, psis[j]) +
                         plx * pfs[j] + σs[j] * randn(rng)
            end
            return out
        end
        a1 = _absc(t1, psi1, σ1, pf1); a2 = _absc(t2, psi2, σ2, pf2)

        hip  = IADData(t = t1, abscissa = a1, abscissa_err = σ1, psi = psi1,
                       parallax_factor = pf1, pm_factor = pm1)
        gaia = IADData(t = t2, abscissa = a2, abscissa_err = σ2, psi = psi2,
                       parallax_factor = pf2, pm_factor = pm2)
        joint = merge_iad(hip, gaia)

        # Profile the likelihood in M_sec: truth vs a 3x-too-heavy companion.
        function _discrimination(iad)
            data = Data(t_rv = [49000.0, 57000.0], rv = [0.0, 0.0],
                        rv_err = [1.0, 1.0], iad = iad)
            params = Params(max_kplanet = 1, planet_modes = [RVAS],
                instruments = InstrumentConfig(rv = ["X"]), data = data,
                stability = :none, M_s = M_pri,
                parametrization = ParametrizationConfig(mass = :a_driven),
                priors = Dict{String, PriorSpec}(
                    "n_p" => FixedPrior(1.0), "a_k1" => LogUniformPrior(0.5, 200.0),
                    "M_sec_k1" => LogUniformPrior(0.001, 1.0),
                    "sesinw_k1" => UniformPrior(-1.0, 1.0),
                    "secosw_k1" => UniformPrior(-1.0, 1.0),
                    "Mo_k1" => UniformPrior(0.0, 2π), "inc_k1" => SinePrior(),
                    "Omega_k1" => UniformPrior(0.0, 2π),
                    "sigma_X" => LogUniformPrior(0.1, 10.0),
                    "M_pri" => FixedPrior(M_pri)))
            theta = Theta(params)
            a_true = ((M_pri + M_sec) * (P_d / Nereus.KEPLER_YEAR_DAYS)^2)^(1/3)
            set_param!(theta, "n_p", 1); set_param!(theta, "a_k1", a_true)
            set_param!(theta, "sesinw_k1", sqrt(e) * sin(ω))
            set_param!(theta, "secosw_k1", sqrt(e) * cos(ω))
            set_param!(theta, "Mo_k1", mod(2π * (53000.0 - tp) / P_d, 2π))
            set_param!(theta, "inc_k1", inc); set_param!(theta, "Omega_k1", Ω)
            set_param!(theta, "plx", plx); set_param!(theta, "gamma_X", 0.0)
            set_param!(theta, "sigma_X", 1.0)
            set_param!(theta, "M_sec_k1", M_sec)
            ll_truth = iad_log_likelihood(theta, data)
            set_param!(theta, "M_sec_k1", 3 * M_sec)
            ll_wrong = iad_log_likelihood(theta, data)
            return ll_truth - ll_wrong
        end

        d_hip   = _discrimination(hip)
        d_gaia  = _discrimination(gaia)
        d_joint = _discrimination(joint)

        @test isfinite(d_hip) && isfinite(d_gaia) && isfinite(d_joint)
        # Neither mission alone can say anything: a 90-yr orbit over a
        # 3-yr baseline IS a linear sky path, and the marginalisation eats
        # it. Measured: 0.58 nats for Hipparcos, 1.10 for Gaia.
        @test d_hip  < 5
        @test d_gaia < 5
        # Shared mu across the 25-yr gap: measured 207 nats. A per-mission
        # mu would give d_joint ≈ d_hip + d_gaia ≈ 1.7, so the factor-of-10
        # margin below is the test that catches that mistake.
        @test d_joint > 50
        @test d_joint > 10 * (d_hip + d_gaia)
    end

    # =================================================================
    # 5b. Conventions.
    #
    # Every test above injects a stellar reflex and nothing else, so none
    # of them contains any parallax or proper-motion signal at all — they
    # cannot catch a swapped RA/Dec basis, a sign-flipped parallax factor,
    # or a mis-anchored time origin. This one injects a known sky path with
    # per-instrument origins and reads the marginal solution back.
    # =================================================================
    @testset "marginal solution recovers injected parallax, PM and zero points" begin
        rng = MersenneTwister(23)
        ϖ, μα, μδ = 27.5, 143.0, -88.0                # mas, mas/yr, mas/yr
        zp = [(0.0, 0.0), (4.0, -6.5)]                # per-instrument frame offsets
        n1, n2 = 50, 50
        e1 = jyear_to_mjd(1991.25); e2 = jyear_to_mjd(2017.5)   # different origins
        t1 = collect(range(e1 - 550, e1 + 700, length = n1))
        t2 = collect(range(e2 - 500, e2 + 500, length = n2))
        psi1 = 2π .* rand(rng, n1); psi2 = 2π .* rand(rng, n2)
        pf1 = sin.(2π .* (t1 .- e1) ./ 365.25 .- psi1)
        pf2 = sin.(2π .* (t2 .- e2) ./ 365.25 .- psi2)
        pm1 = (t1 .- e1) ./ 365.25                    # years from ITS OWN epoch
        pm2 = (t2 .- e2) ./ 365.25

        function sky(psis, pfs, pms, m)
            [zp[m][1] * sin(psis[j]) + zp[m][2] * cos(psis[j]) +
             ϖ * pfs[j] + μα * sin(psis[j]) * pms[j] + μδ * cos(psis[j]) * pms[j]
             for j in eachindex(psis)]
        end
        iad = IADData(t = vcat(t1, t2),
                      abscissa = vcat(sky(psi1, pf1, pm1, 1), sky(psi2, pf2, pm2, 2)),
                      abscissa_err = fill(1.0, n1 + n2),
                      psi = vcat(psi1, psi2),
                      parallax_factor = vcat(pf1, pf2),
                      pm_factor = vcat(pm1, pm2),
                      inst = vcat(fill(1, n1), fill(2, n2)))

        # The parallax is NOT a marginalised column any more — it is the
        # sampled `astrom_plx(theta)`, and `_iad_residuals!` subtracts its
        # term from the data before the solve. Mirror that here.
        n_q = Nereus._iad_n_q(2)
        @test n_q == 6                      # 2 shared PM + 2 zero points × 2
        r = iad.abscissa .- ϖ .* iad.parallax_factor
        A = zeros(n_q, n_q); v = zeros(n_q)
        Nereus._iad_normal_equations!(A, v, iad, r, iad.pm_factor,
                                      Nereus._iad_pos_cols(2))
        q = LinearAlgebra.Symmetric(A) \ v
        # Shared block: one proper motion for the star, recovered across two
        # missions with DIFFERENT time origins. Columns 3-4, not 4-5.
        @test q[3] ≈ μα atol = 1e-8
        @test q[4] ≈ μδ atol = 1e-8
        # Per-instrument zero points, each in its own frame.
        @test q[1] ≈ zp[1][1] atol = 1e-8
        @test q[2] ≈ zp[1][2] atol = 1e-8
        @test q[5] ≈ zp[2][1] atol = 1e-8
        @test q[6] ≈ zp[2][2] atol = 1e-8

        # And the parallax must NOT be absorbable by the remaining design:
        # feeding the UNsubtracted abscissae through leaves the zero points
        # and PM biased, which is exactly the leakage the old free-ϖ column
        # hid. If this ever passes, the parallax has a column again.
        A2 = zeros(n_q, n_q); v2 = zeros(n_q)
        Nereus._iad_normal_equations!(A2, v2, iad, iad.abscissa, iad.pm_factor,
                                      Nereus._iad_pos_cols(2))
        q2 = LinearAlgebra.Symmetric(A2) \ v2
        @test !isapprox(q2[3], μα; atol = 1e-6)
    end

    # =================================================================
    # 5c. Residual vs absolute abscissae must describe the same star.
    #
    # Hipparcos stores O−C residuals, Gaia stores full abscissae. The same
    # physical data expressed either way must give the same likelihood —
    # if it does not, a joint fit is comparing a catalogue CORRECTION with
    # a full quantity. This is what made the IAD + Gaia DR3 path unusable
    # with real loader output before: `load_gaia_dr3` reports an absolute
    # parallax while `fetch_hip_iad` reports residuals, and the two blocks
    # pulled the shared parallax toward 30 mas and toward 0 respectively.
    # =================================================================
    @testset "residual and absolute conventions agree" begin
        rng = MersenneTwister(3)
        n = 60
        t0 = jyear_to_mjd(1991.25)
        t = collect(range(t0 - 365.25*1.5, t0 + 365.25*1.9, length = n))
        psi = 2π .* rand(rng, n)
        plxf = sin.(2π .* (t .- t0) ./ 365.25 .- psi)
        pmf  = (t .- t0) ./ 365.25
        σ    = fill(1.5, n)
        ϖ_cat, μα_cat, μδ_cat = 30.0, 180.0, -138.0   # a high-proper-motion star
        s = sin.(psi); c = cos.(psi)
        cat_path = ϖ_cat .* plxf .+ μα_cat .* s .* pmf .+ μδ_cat .* c .* pmf
        res = σ .* randn(rng, n)                      # residuals: noise only
        gost = GOSTData(t = collect(range(57000.0, 58000.0, length = 30)),
                        psi = mod.(2π .* (1:30) ./ 7, 2π),
                        parallax_factor = sin.(2π .* (1:30) ./ 365.25))
        cov = zeros(5, 5); for k in 1:5; cov[k, k] = (k <= 2 ? 0.3 : 0.05)^2; end
        # Position offsets zero — the documented "PM + parallax only" form,
        # and what `load_gaia_dr3` produces by default.
        gaia = GaiaDR3Data(params = (0.0, 0.0, ϖ_cat, μα_cat, μδ_cat),
                           cov = cov, t_ref = jyear_to_mjd(2016.0))

        function _ll(absc; kind, ref)
            iad = IADData(t = t, abscissa = absc, abscissa_err = σ, psi = psi,
                          parallax_factor = plxf, pm_factor = pmf,
                          ref_params = [ref], abscissa_kind = kind)
            data = Data(t_rv = [49000.0, 49300.0], rv = [0.0, 0.0],
                        rv_err = [1.0, 1.0], iad = iad, gost = gost, gaia_dr3 = gaia)
            params = Params(max_kplanet = 1, planet_modes = [RVAS],
                instruments = InstrumentConfig(rv = ["X"]), data = data,
                stability = :none, M_s = 1.0)
            theta = Theta(params)
            set_param!(theta, "n_p", 0)
            set_param!(theta, "P_k1", 365.25); set_param!(theta, "K_k1", 0.0)
            set_param!(theta, "sesinw_k1", 0.0); set_param!(theta, "secosw_k1", 0.0)
            set_param!(theta, "Mo_k1", 0.0); set_param!(theta, "inc_k1", deg2rad(60))
            set_param!(theta, "Omega_k1", 0.0); set_param!(theta, "plx", ϖ_cat)
            set_param!(theta, "sigma_X", 1.0)
            return iad_log_likelihood(theta, data)
        end

        ll_res = _ll(res; kind = :residual,
                     ref = (0.0, 0.0, ϖ_cat, μα_cat, μδ_cat))
        ll_abs = _ll(res .+ cat_path; kind = :absolute,
                     ref = (0.0, 0.0, 0.0, 0.0, 0.0))
        @test ll_res ≈ ll_abs rtol = 1e-9
        # And the fit is actually good: chi2 of order n, not of order 1e7.
        # ln L = -chi2/2 - sum(log sigma) - ... , so bound it loosely.
        @test ll_res > -0.5 * 4n - sum(log, σ) - 0.5 * n * log(2π)

        # Same again with a SUPPLIED Gaia position (the caller asserting a
        # common frame), which takes the other branch: the full 5x5
        # constraint including the position transport. Here the reference
        # solution has to be transported to the Gaia epoch before it is
        # subtracted -- subtracting it untransported leaves mu_cat*Delta_t,
        # about 4500 mas for this star against a 0.3 mas position error.
        Δt = (jyear_to_mjd(2016.0) - sum(t)/n) / 365.25
        gaia_pos = GaiaDR3Data(params = (μα_cat*Δt, μδ_cat*Δt,
                                         ϖ_cat, μα_cat, μδ_cat),
                               cov = cov, t_ref = jyear_to_mjd(2016.0))
        function _ll_pos(absc; kind, ref)
            iad = IADData(t = t, abscissa = absc, abscissa_err = σ, psi = psi,
                          parallax_factor = plxf, pm_factor = pmf,
                          ref_params = [ref], abscissa_kind = kind)
            data = Data(t_rv = [49000.0, 49300.0], rv = [0.0, 0.0],
                        rv_err = [1.0, 1.0], iad = iad, gost = gost,
                        gaia_dr3 = gaia_pos)
            params = Params(max_kplanet = 1, planet_modes = [RVAS],
                instruments = InstrumentConfig(rv = ["X"]), data = data,
                stability = :none, M_s = 1.0)
            theta = Theta(params)
            set_param!(theta, "n_p", 0)
            set_param!(theta, "P_k1", 365.25); set_param!(theta, "K_k1", 0.0)
            set_param!(theta, "sesinw_k1", 0.0); set_param!(theta, "secosw_k1", 0.0)
            set_param!(theta, "Mo_k1", 0.0); set_param!(theta, "inc_k1", deg2rad(60))
            set_param!(theta, "Omega_k1", 0.0); set_param!(theta, "plx", ϖ_cat)
            set_param!(theta, "sigma_X", 1.0)
            return iad_log_likelihood(theta, data)
        end
        @test _ll_pos(res; kind = :residual,
                      ref = (0.0, 0.0, ϖ_cat, μα_cat, μδ_cat)) ≈
              _ll_pos(res .+ cat_path; kind = :absolute,
                      ref = (0.0, 0.0, 0.0, 0.0, 0.0)) rtol = 1e-9
    end

    # =================================================================
    # 5d. Two imagers, two error budgets.
    #
    # `RelAstromData.inst` indexes `InstrumentConfig.as_names`, and each
    # imager gets its own `sigma_as_<name>` jitter. Before this, GPI and
    # SPHERE observing the same companion shared one error model, which
    # means the better-calibrated one absorbs the other's systematics.
    # =================================================================
    @testset "relative astrometry — per-imager jitter" begin
        relast = RelAstromData(
            t       = [55000.0, 55090.0, 55180.0, 55270.0],
            ra_off  = [10.0, 8.0, -5.0, -10.0],
            dec_off = [0.0, 6.0, 9.0, 0.0],
            ra_err  = [1.0, 1.0, 1.0, 1.0],
            dec_err = [1.0, 1.0, 1.0, 1.0],
            inst    = [1, 1, 2, 2])
        @test relast.inst == [1, 1, 2, 2]
        @test n_relast_inst(relast) == 2
        # Defaults to one imager when not given.
        @test n_relast_inst(RelAstromData(t = [55000.0], ra_off = [1.0],
                                          dec_off = [1.0], ra_err = [1.0],
                                          dec_err = [1.0])) == 1

        data = Data(t_rv = [55000.0, 55100.0], rv = [0.0, 0.0],
                    rv_err = [1.0, 1.0], relastrom = relast)

        function _params(as_names)
            Params(max_kplanet = 1, planet_modes = [RVAS],
                   instruments = InstrumentConfig(rv = ["X"], as = as_names),
                   data = data, stability = :none, M_s = 1.0)
        end
        function _set!(theta)
            set_param!(theta, "n_p", 1); set_param!(theta, "P_k1", 900.0)
            set_param!(theta, "K_k1", 50.0)
            set_param!(theta, "sesinw_k1", 0.0); set_param!(theta, "secosw_k1", 0.0)
            set_param!(theta, "Mo_k1", 0.3); set_param!(theta, "inc_k1", deg2rad(60))
            set_param!(theta, "Omega_k1", 0.5); set_param!(theta, "plx", 25.0)
            set_param!(theta, "sigma_X", 5.0)
            return theta
        end

        # No astrometric instruments named: no slot, no change at all.
        p0 = _params(String[])
        @test n_as_instruments(p0.config.instruments) == 0
        @test isempty(p0.layout.systemic.as_jitter)
        ll_plain = relastrom_log_likelihood(_set!(Theta(p0)), data)

        # Two imagers named: one jitter slot each.
        p2 = _params(["GPI", "SPHERE"])
        @test n_as_instruments(p2.config.instruments) == 2
        @test "sigma_as_GPI" in p2.layout.names
        @test "sigma_as_SPHERE" in p2.layout.names
        th = _set!(Theta(p2))
        set_param!(th, "sigma_as_GPI", 0.0)
        set_param!(th, "sigma_as_SPHERE", 0.0)
        @test as_jitter(th, 1) == 0.0
        # Zero jitter reproduces the no-instrument answer exactly: the
        # likelihood must not take a sqrt round-trip it does not need.
        @test relastrom_log_likelihood(th, data) === ll_plain

        # Inflating ONE imager changes the answer, and inflating the other
        # by the same amount changes it differently — the two budgets are
        # genuinely separate. (Equal here only by coincidence of the
        # fixture would be a false pass, so assert they differ.)
        set_param!(th, "sigma_as_GPI", 5.0)
        ll_gpi = relastrom_log_likelihood(th, data)
        set_param!(th, "sigma_as_GPI", 0.0)
        set_param!(th, "sigma_as_SPHERE", 5.0)
        ll_sph = relastrom_log_likelihood(th, data)
        @test ll_gpi != ll_plain
        @test ll_sph != ll_plain
        @test ll_gpi != ll_sph

        # Published in milliarcseconds, not m/s. `sigma_as_GPI` starts with
        # "sigma", which the science tables otherwise read as RV jitter.
        @test Nereus.sci_param_unit("sigma_as_GPI", p2) == ("mas", false)
        @test Nereus.sci_param_unit("sigma_X", p2) == ("m/s", false)

        # Sharing collapses them onto one slot.
        p_sh = Params(max_kplanet = 1, planet_modes = [RVAS],
                      instruments = InstrumentConfig(rv = ["X"],
                                                     as = ["GPI", "SPHERE"]),
                      data = data, stability = :none, M_s = 1.0,
                      sharing = Dict(:as_jitter => [["GPI", "SPHERE"]]))
        @test p_sh.layout.systemic.as_jitter[1] == p_sh.layout.systemic.as_jitter[2]
    end

    # =================================================================
    # 6. Guards.
    # =================================================================
    @testset "under-determined multi-instrument design returns 0" begin
        # Two instruments need 3 + 2*2 = 7 rows before the marginalisation
        # is defined; 6 rows must bail out the way 4 rows do for one.
        iad = IADData(t = collect(48000.0:10.0:48050.0), abscissa = zeros(6),
                      abscissa_err = fill(1.0, 6),
                      psi = collect(range(0, π, length = 6)),
                      inst = [1, 1, 1, 2, 2, 2])
        data = Data(t_rv = [49000.0, 49300.0], rv = [1.0, -1.0],
                    rv_err = [1.0, 1.0], iad = iad)
        params = Params(max_kplanet = 1, planet_modes = [RVAS],
            instruments = InstrumentConfig(rv = ["X"]), data = data,
            stability = :none, M_s = 1.0)
        theta = Theta(params)
        set_param!(theta, "n_p", 1); set_param!(theta, "P_k1", 365.25)
        set_param!(theta, "K_k1", 20.0); set_param!(theta, "sesinw_k1", 0.0)
        set_param!(theta, "secosw_k1", 0.0); set_param!(theta, "Mo_k1", 0.0)
        set_param!(theta, "inc_k1", deg2rad(60))
        set_param!(theta, "Omega_k1", deg2rad(30)); set_param!(theta, "plx", 25.0)
        set_param!(theta, "sigma_X", 5.0)
        @test iad_log_likelihood(theta, data) == 0.0
    end
end
