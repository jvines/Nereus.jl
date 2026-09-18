# Per-planet astrometric coupling as a trans-dimensional axis.
#
# Every trans-dim exoplanet code samples K, the number of companions. With
# heterogeneous data the honest model index is (K, S_1..S_K) where S_k is the
# set of observables that actually constrain companion k. `PlanetDataSources`
# already expresses that per planet — it was just fixed at construction.
#
# WHY IT MATTERS: a companion the RV establishes beyond doubt can be
# astrometrically undetected. With the coupling forced on, the fit reports an
# inclination — hence a "dynamical mass" — whether or not the astrometry
# constrained one, and prior volume gets published as a measurement. That is the
# Hipparcos-era failure mode.
#
# Layout is built at the MAXIMAL mode and masked, because a layout cannot change
# block type at run time — the same trick planets and noise already use. With
# the flag off, inc/Omega carry their priors, which integrate to 1 and leave the
# marginal likelihood untouched, so configurations stay evidence-comparable.
using Test
using Nereus
using Nereus: TransDimState, activate_as!, deactivate_as!, is_as_active,
              is_planet_as_active, copy_into!, Theta, set_param!,
              HGCAData, mjd_epochs, astrom_log_likelihood, planet_indices
using Random, Statistics

@testset "astrometric coupling mask" begin

    @testset "mask defaults on and survives copying" begin
        td = TransDimState(max_planets = 3, n_noise = 2)
        @test all(td.as_active)                       # unchanged behaviour
        deactivate_as!(td, 2)
        @test !is_as_active(td, 2) && is_as_active(td, 1)
        # BOTH copy paths must carry it, or the mask is silently lost mid-chain
        @test copy(td).as_active == td.as_active
        dst = TransDimState(max_planets = 3, n_noise = 2)
        copy_into!(dst, td)
        @test dst.as_active == td.as_active
        activate_as!(td, 2)
        @test is_as_active(td, 2)
        @test_throws BoundsError deactivate_as!(td, 9)
    end

    @testset "untracked and fixed-dim default to coupled" begin
        td = TransDimState(max_planets = 1, n_noise = 0)
        @test is_as_active(td, 5)                     # out of range ⇒ true
    end

    # ---- the mask must actually gate the astrometric likelihood ----------
    @testset "decoupling removes the reflex from the astrometry" begin
        t_rv = collect(55000.0:20.0:55400.0)
        hgca = HGCAData(epochs = mjd_epochs((1991.25, 2004.6, 2016.0)),
                        pmra = (5.0, 4.95, 4.9), pmdec = (-3.0, -3.05, -3.1),
                        sigma_pmra = (0.05, 0.05, 0.05),
                        sigma_pmdec = (0.05, 0.05, 0.05),
                        plx = 25.0, plx_err = 0.05, hip_id = 1)
        d = Data(t_rv = t_rv, rv = 30 .* randn(MersenneTwister(2), length(t_rv)),
                 rv_err = fill(5.0, length(t_rv)),
                 rv_inst = ones(Int, length(t_rv)), hgca = hgca)
        p = Params(max_kplanet = 1, planet_modes = [RVAS],
                   instruments = InstrumentConfig(rv = ["A"]), data = d,
                   stability = :none, M_s = 1.0)
        th = Theta{Float64}(p)
        th.td = TransDimState(max_planets = 1, n_noise = 0)
        Nereus.activate_planet!(th.td, 1)
        for (i, s) in enumerate(p.layout.unfrozen_idx)
            lo, hi = bounds(p.layout.unfrozen_priors[i])
            th.values[s] = clamp((lo + hi) / 2, lo, hi)
        end
        for (k, v) in ("P_k1" => 3000.0, "K_k1" => 80.0, "inc_k1" => 1.0,
                       "Omega_k1" => 1.2, "plx" => 25.0, "m_pri" => 1.0,
                       "sesinw_k1" => 0.0, "secosw_k1" => 0.0)
            haskey(p.layout.name_to_idx, k) && set_param!(th, k, v)
        end

        ll_on = astrom_log_likelihood(th, d)
        deactivate_as!(th.td, 1)
        ll_off = astrom_log_likelihood(th, d)
        @test isfinite(ll_on) && isfinite(ll_off)
        @test ll_on != ll_off               # the mask is genuinely consulted

        # decoupled, the astrometry must not care about inc/Omega at all
        set_param!(th, "inc_k1", 2.0); set_param!(th, "Omega_k1", 4.0)
        @test astrom_log_likelihood(th, d) ≈ ll_off atol = 1e-10

        # recoupled, it must care again
        activate_as!(th.td, 1)
        @test astrom_log_likelihood(th, d) != ll_off
    end

    @testset "decoupling does not touch the RV" begin
        # The whole point: the companion still exists and the RV still sees it.
        t_rv = collect(55000.0:20.0:55400.0)
        d = Data(t_rv = t_rv, rv = 30 .* randn(MersenneTwister(5), length(t_rv)),
                 rv_err = fill(5.0, length(t_rv)), rv_inst = ones(Int, length(t_rv)))
        p = Params(max_kplanet = 1, planet_modes = [RV_ONLY],
                   instruments = InstrumentConfig(rv = ["A"]), data = d,
                   stability = :none)
        th = Theta{Float64}(p)
        th.td = TransDimState(max_planets = 1, n_noise = 0)
        Nereus.activate_planet!(th.td, 1)
        for (i, s) in enumerate(p.layout.unfrozen_idx)
            lo, hi = bounds(p.layout.unfrozen_priors[i])
            th.values[s] = clamp((lo + hi) / 2, lo, hi)
        end
        a = Nereus.rv_log_likelihood(th, d)
        deactivate_as!(th.td, 1)
        @test Nereus.rv_log_likelihood(th, d) ≈ a atol = 1e-12
    end
end
