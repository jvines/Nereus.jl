# The transit window with its cos i term, the per-cadence reach of a supersampled
# cadence, and the sub-sample gate (src/transit_likelihood.jl). A cadence or a
# sub-sample is skipped only when its flux is exactly 1, so the likelihood is the
# one that computes every cadence, to the bit. Checked:
#
#   1. data-free, random orbits (e up to 0.9999, b down to 1e-12 of grazing, k up
#      to 0.9): no time outside the bound has the planet in front of the star with
#      z < 1 + k, by the likelihood's kernels (`_sky_orbit`) or by
#      `planet_sky_position` and `sky_separation`;
#   2. circular orbits: the bound is the transit, T14/2 plus the margins;
#   3. per cadence, random orbits and exposures (none, 20 s to 1 h, some longer
#      than the period): the transit product with the likelihood's windows is
#      === the product with no window, ordinary and gravity-darkened, and no
#      cadence or sub-sample the gates skip is in contact;
#   4. the same on the fixture target with 30-min exposures and with TTVs.
using Test
using Nereus
using Random
using Nereus: _transit_window_halfwidth, _set_transit_windows!, _phot_transit_product,
              _sky_orbit, _sky_position, _sky_separation_signed, QuadLimbDark,
              gd_context, tp_to_tc, tc_to_tp, kipping_q_to_u, planet_sky_position

include(joinpath(@__DIR__, "fixtures", "transit_gate_target.jl"))

twc_fold(t, Tc, P) = (Δ = t - Tc; Δ - P * round(Δ / P))

# In contact (in front, z < 1 + k) by any of the four z kernels.
function twc_contact(t, o)
    orb = _sky_orbit(o.P, o.e, o.ω, o.Tp, o.b, o.aR)
    x, y, zl = _sky_position(orb, t)
    z, sw = _sky_separation_signed(orb, t)
    x2, y2, zl2 = planet_sky_position(t, o.P, o.e, o.ω, o.Tp, o.b, o.aR)
    z2, sw2 = _sky_separation_signed(t, o.P, o.e, o.ω, o.Tp, o.b, o.aR)
    return (zl > 0 && hypot(x, y) < 1 + o.k) | (sw > 0 && z < 1 + o.k) |
           (zl2 > 0 && hypot(x2, y2) < 1 + o.k) | (sw2 > 0 && z2 < 1 + o.k)
end

function twc_orbit(rng; Pmin = 0.3)
    P  = exp(log(Pmin) + rand(rng) * (log(300) - log(Pmin)))
    u  = rand(rng)
    e  = u < 0.2 ? 0.0 : u < 0.5 ? min(1 - 10^(-4 * rand(rng)), 0.99989) : rand(rng)
    ω  = 2π * rand(rng) - π
    k  = exp(log(1e-3) + rand(rng) * (log(0.9) - log(1e-3)))
    aR = exp(log(1.5) + rand(rng) * (log(300) - log(1.5)))
    b  = rand(rng) < 0.25 ? (1 + k) * (1 - 10.0^(-12 * rand(rng))) : (1 + k) * rand(rng)
    Tp = 2_460_000.0 + 50 * rand(rng)
    return (; P, e, ω, k, aR, b, Tp, Tc = tp_to_tc(Tp, P, e, ω))
end

@testset "transit window: nothing in contact is skipped" begin

    @testset "data-free: no contact outside the bound" begin
        rng = MersenneTwister(77)
        n_fin = 0; n_out = 0; bad = 0
        for _ in 1:20_000
            o = twc_orbit(rng)
            hw = _transit_window_halfwidth(o.P, o.e, o.ω, o.k, o.aR,
                                           max(abs(o.Tc), abs(o.Tp)), o.b)
            isfinite(hw) || continue
            n_fin += 1
            @test hw < o.P / 2
            for j in 1:60
                # the whole orbit, and just outside the bound (1e-12 to 1e-3 of it)
                Δ = j <= 20 ? (rand(rng) - 0.5) * o.P : hw * (1 + 10.0^(-3 - 9 * rand(rng)))
                t = o.Tc + (rand(rng) < 0.5 ? -Δ : Δ) + o.P * rand(rng, -400:400)
                abs(twc_fold(t, o.Tc, o.P)) > hw || continue
                n_out += 1
                twc_contact(t, o) && (bad += 1)
            end
        end
        @info "window with cos i, data-free" n_fin n_out
        @test n_fin > 10_000 && n_out > 600_000
        @test bad == 0
    end

    @testset "circular: the bound is the transit" begin
        for (b, k, aR) in ((0.0, 0.1, 10.0), (0.75, 0.116, 6.8), (1.05, 0.1, 10.0),
                           (0.3, 0.6, 4.0))
            P = 4.0
            hw = _transit_window_halfwidth(P, 0.0, 0.3, k, aR, 0.0, b)
            sini = sqrt(1 - (b / aR)^2)
            t14_half = asin(sqrt((1 + k)^2 - b^2) / (aR * sini)) / (2π) * P
            @test hw ≈ t14_half rtol = 1e-6
            @test hw >= t14_half
        end
        # without b, the looser b = 0 bound
        @test _transit_window_halfwidth(4.0, 0.0, 0.3, 0.1, 10.0, 0.0) ==
              _transit_window_halfwidth(4.0, 0.0, 0.3, 0.1, 10.0, 0.0, 0.0)
        @test _transit_window_halfwidth(4.0, 0.0, 0.3, 0.1, 10.0, 0.0, 0.8) <
              _transit_window_halfwidth(4.0, 0.0, 0.3, 0.1, 10.0, 0.0)
        # a b that is not finite drops the term; a b past grazing leaves the bound finite
        @test _transit_window_halfwidth(4.0, 0.0, 0.3, 0.1, 10.0, 0.0, NaN) ==
              _transit_window_halfwidth(4.0, 0.0, 0.3, 0.1, 10.0, 0.0)
        @test isfinite(_transit_window_halfwidth(4.0, 0.0, 0.3, 0.1, 10.0, 0.0, 50.0))
    end

    @testset "per cadence, random orbits and exposures" begin
        rng = MersenneTwister(5)
        ld = QuadLimbDark([0.4, 0.25])
        exps = [0.0, 20.0, 120.0, 121.5, 200.0, 600.0, 1800.0, 3600.0] ./ 86_400
        n_orb = 0; n_cad = 0; n_skip_cad = 0; n_skip_sub = 0; n_contact = 0
        same = true; bad = 0
        for it in 1:4_000
            # a few very short periods, so that an exposure can exceed P
            o = twc_orbit(rng; Pmin = it % 10 == 0 ? 0.02 : 0.3)
            ts = max(abs(o.Tc), abs(o.Tp))
            win = _set_transit_windows!([Inf], 1, [true], [0], nothing, [o.P], [o.e], [o.ω],
                                        [o.k], [o.aR], [o.Tc], [o.Tp], 0.0, [o.b])
            hw = win[1]
            isfinite(hw) || continue
            n_orb += 1
            # gravity darkening on half the orbits: the GD kernel's z and z_los
            gd = nothing; ctx = nothing
            if isodd(it)
                gd = (i_star = 0.3 + 2.5 * rand(rng), λs = [2π * rand(rng) - π],
                      ω_frac = 0.9 * rand(rng), on = [true])
                ctx = reshape([gd_context(gd.i_star, gd.λs[1], gd.ω_frac, 0.2)], 1, 1)
            end
            for _ in 1:80
                texp = exps[rand(rng, 1:length(exps))]
                # near the window edge, as far as the cadence's reach, or anywhere
                Δ = rand(rng) < 0.7 ? (hw + texp / 2) * (1 + 0.2 * (rand(rng) - 0.5)) :
                                      (rand(rng) - 0.5) * o.P
                t = o.Tc + (rand(rng) < 0.5 ? -Δ : Δ) + o.P * rand(rng, -200:200)
                pr(w) = _phot_transit_product(t, texp, ld, 1, [true], [o.P], [o.e], [o.ω],
                                              [o.Tp], [o.b], [o.aR], [o.k], [o.Tc], w, [0],
                                              nothing, 2, gd, ctx, 1)
                same &= pr(win) === pr([Inf])
                n_cad += 1
                # what the gates skip, sub-sample by sub-sample
                ns = Nereus._phot_n_super_point(texp)
                integ = ns > 1 && texp > 0
                d0 = twc_fold(t, o.Tc, o.P)
                cad_skip = abs(d0) > hw + (integ ? texp / 2 : 0.0)
                n_skip_cad += cad_skip
                for s in 1:(integ ? ns : 1)
                    off = integ ? ((2s - ns - 1) / (2 * ns)) * texp : 0.0
                    skip = cad_skip || (integ && abs(twc_fold(t + off, o.Tc, o.P)) > hw)
                    n_skip_sub += skip && !cad_skip
                    c = twc_contact(t + off, o)
                    n_contact += c
                    (skip && c) && (bad += 1)
                end
            end
        end
        @info "per-cadence gates" n_orb n_cad n_skip_cad n_skip_sub n_contact
        @test n_orb > 2_000 && n_skip_cad > 20_000 && n_skip_sub > 20_000 && n_contact > 20_000
        @test same
        @test bad == 0
    end

    for (exposure, ttv) in ((true, false), (true, true), (false, true))
        @testset "fixture target, exposure = $exposure, ttv = $ttv" begin
            tg = transit_gate_target(; exposure, ttv)
            data = tg.data; n = length(data.t_phot)
            th = Nereus.Theta{Float64}(tg.params)
            sys = tg.params.layout.systemic
            n_super = Nereus._phot_n_super(data)
            n_cmp = 0
            for v in transit_gate_points(tg, 60; seed = 43)
                th.values .= v
                P = Nereus.planet_P(th, 1); e, ω = Nereus.planet_e_w(th, 1)
                (0 <= e < 1) || continue
                Tp = tc_to_tp(Nereus.planet_time_anchor(th, 1), P, e, ω)
                b, rr = Nereus.planet_b_rr(th, 1)
                b < 1 + rr || continue
                aR = Nereus.rho_s_to_a_Rs(Nereus.rho_s(th), P)
                Tc = tp_to_tc(Tp, P, e, ω)
                _, ttv_state = Nereus._decode_ttv_state(th, Nereus.planet_indices(th))
                r_for_j = [ttv ? 1 : 0]
                win = _set_transit_windows!([Inf], 1, [true], r_for_j, ttv_state, [P], [e],
                                            [ω], [rr], [aR], [Tc], [Tp], 0.0, [b])
                lds = [QuadLimbDark(collect(kipping_q_to_u(th.values[sys.ld_q1[ix]],
                                                           th.values[sys.ld_q2[ix]]))) for ix in 1:2]
                same = true
                for i in 1:n
                    texp = isempty(data.exposure_times) ? 0.0 : data.exposure_times[i]
                    pr(w) = _phot_transit_product(data.t_phot[i], texp, lds[data.phot_inst[i]],
                                                  1, [true], [P], [e], [ω], [Tp], [b], [aR],
                                                  [rr], [Tc], w, r_for_j, ttv_state, n_super)
                    same &= pr(win) === pr([Inf])
                end
                @test same
                n_cmp += 1
            end
            @test n_cmp > 20
        end
    end
end
