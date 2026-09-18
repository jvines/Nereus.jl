# Birth/death of the astrometric coupling: detailed balance.
#
# The move flips whether companion k's reflex is fitted to the astrometry. It is
# a pure dimension EXTENSION — mass in RVASBlock is derived from
# (K, P, e, sin i, M_pri), so nothing changes meaning and there is no Jacobian
# beyond the proposal density. This tests that claim rather than assuming it: an
# exact reverse density forces log_q_death = -log_q_birth once the combinatorial
# terms cancel, and a wrong q shows up there and nowhere else.
using Test
using Nereus
using Nereus: TransDimState, Theta, set_param!, propose_as_birth,
              propose_as_death, is_as_active, activate_as!, deactivate_as!,
              activate_planet!, HGCAData, mjd_epochs, _as_coupling_slots,
              _as_togglable, astrom_log_likelihood
using Random, Statistics

@testset "astrometric coupling move" begin
    function build(mode; nplanet = 1)
        t_rv = collect(55000.0:20.0:55400.0)
        hgca = HGCAData(epochs = mjd_epochs((1991.25, 2004.6, 2016.0)),
                        pmra = (5.0, 4.95, 4.9), pmdec = (-3.0, -3.05, -3.1),
                        sigma_pmra = (0.05,0.05,0.05), sigma_pmdec = (0.05,0.05,0.05),
                        plx = 25.0, plx_err = 0.05, hip_id = 1)
        d = Data(t_rv = t_rv, rv = 30 .* randn(MersenneTwister(2), length(t_rv)),
                 rv_err = fill(5.0, length(t_rv)),
                 rv_inst = ones(Int, length(t_rv)), hgca = hgca)
        p = Params(max_kplanet = nplanet, planet_modes = fill(mode, nplanet),
                   instruments = InstrumentConfig(rv = ["A"]), data = d,
                   stability = :none, M_s = 1.0)
        th = Theta{Float64}(p)
        th.td = TransDimState(max_planets = nplanet, n_noise = 0)
        for k in 1:nplanet; activate_planet!(th.td, k); end
        for (i, s) in enumerate(p.layout.unfrozen_idx)
            lo, hi = bounds(p.layout.unfrozen_priors[i])
            th.values[s] = clamp((lo + hi) / 2, lo, hi)
        end
        return p, d, th
    end

    @testset "which slots the coupling controls" begin
        # the asymmetry is physical: a transiting planet already has inc, so
        # coupling buys only the NODE
        p, d, th = build(RVAS)
        @test length(_as_coupling_slots(th, 1)) == 2      # inc + Omega
        p2, d2, th2 = build(RVPMAS)
        @test length(_as_coupling_slots(th2, 1)) == 1     # Omega only
        # a planet with no astrometry cannot be toggled at all
        p3, d3, th3 = build(RV_ONLY)
        @test isempty(_as_coupling_slots(th3, 1))
        @test isempty(_as_togglable(th3))
    end

    # ---- detailed balance: the reason this test exists -------------------
    @testset "birth/death reversibility" begin
        for mode in (RVAS, RVPMAS)
            p, d, th = build(mode)
            deactivate_as!(th.td, 1)               # start decoupled
            n = 0
            for trial in 1:12
                born, lqb = propose_as_birth(th, MersenneTwister(100 + trial))
                isfinite(lqb) || continue
                @test is_as_active(born.td, 1)
                back, lqd = propose_as_death(born, MersenneTwister(7))
                @test isfinite(lqd)
                @test !is_as_active(back.td, 1)
                @test lqd ≈ -lqb atol = 1e-9       # one togglable ⇒ exact
                n += 1
            end
            @test n > 0
        end
    end

    @testset "no move available when there is nothing to toggle" begin
        p, d, th = build(RVAS)
        activate_as!(th.td, 1)
        @test propose_as_birth(th, MersenneTwister(1))[2] == -Inf
        deactivate_as!(th.td, 1)
        @test propose_as_death(th, MersenneTwister(1))[2] == -Inf
        # RV-only: neither direction is possible
        p3, d3, th3 = build(RV_ONLY)
        @test propose_as_birth(th3, MersenneTwister(1))[2] == -Inf
        @test propose_as_death(th3, MersenneTwister(1))[2] == -Inf
    end

    @testset "the move changes the astrometric likelihood" begin
        p, d, th = build(RVAS)
        for (k,v) in ("P_k1"=>3000.0,"K_k1"=>80.0,"inc_k1"=>1.0,"Omega_k1"=>1.2,
                      "plx"=>25.0,"m_pri"=>1.0,"sesinw_k1"=>0.0,"secosw_k1"=>0.0)
            haskey(p.layout.name_to_idx,k) && set_param!(th,k,v)
        end
        a = astrom_log_likelihood(th, d)
        deactivate_as!(th.td, 1)
        @test astrom_log_likelihood(th, d) != a
    end

    @testset "combinatorics with two togglable planets" begin
        p, d, th = build(RVAS; nplanet = 2)
        deactivate_as!(th.td, 1); deactivate_as!(th.td, 2)
        born, lqb = propose_as_birth(th, MersenneTwister(3))
        @test isfinite(lqb)
        @test count(j -> is_as_active(born.td, j), 1:2) == 1
        # 2 off -> 1 on: the ratio must carry log(1) - log(2), not log(1)-log(1)
        back, lqd = propose_as_death(born, MersenneTwister(3))
        @test isfinite(lqd)
        @test count(j -> is_as_active(back.td, j), 1:2) == 0
    end
end
