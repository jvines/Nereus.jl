# The photo-eccentric reparametrization, density = :rho_circ (src/density.jl).
#
# A transiting planet with a free eccentricity samples the circular-equivalent
# density ρ_circ = ρ★ g³, g = (1 + e sin ω)/√(1 − e²), in place of ρ★ (the
# anchor, when ρ★ is sampled) or of its two e-coordinates (as ρ_circ and the
# ellipse coordinate η). The assertions pin that it is the same model: the
# likelihood is the same function at matching physical points, the Jacobian is
# the map's, the prior of ρ★, e, ω and b sampled in the new coordinates is the
# old prior, and everything outside its support is -Inf, never NaN. Then where
# it applies, what it reports, and what it refuses.

using Nereus, MCMCChains, Random, Statistics, LinearAlgebra, Test
using Nereus: Theta, UniformPrior, LogUniformPrior, NormalPrior, FixedPrior

# A light curve of a 3.5 d planet (b = 0.3, Rp/R★ = 0.1) round a ρ★ = 1 star,
# and of a second, 7.9 d one; times in days, Gaussian noise. Flux by direct
# integration over the occulted disc, quadratic limb darkening.
function _dr_flux(t, P, Tc, rp, a_Rs, b; u = (0.40, 0.25), n = 300)
    out = ones(length(t))
    k = (0:n-1) .+ 0.5; rr = rp .* sqrt.(k ./ n); th = k .* 2.399963
    norm = π * (1 - u[1] / 3 - u[2] / 6)
    for (i, ti) in enumerate(t)
        ph = 2π * (ti - Tc) / P
        cos(ph) > 0 || continue
        x, y = a_Rs * sin(ph), b * cos(ph)
        lost = 0.0
        for j in 1:n
            px = x + rr[j] * cos(th[j]); py = y + rr[j] * sin(th[j]); r2 = px^2 + py^2
            r2 < 1 || continue
            μ = sqrt(1 - r2); lost += 1 - u[1] * (1 - μ) - u[2] * (1 - μ)^2
        end
        out[i] = 1 - lost * π * rp^2 / n / norm
    end
    return out
end
function _dr_lc(; two = false, seed = 1)
    rng = MersenneTwister(seed)
    t = Float64[]
    for i in 0:2; append!(t, 59000.5 + 3.5i - 0.3 .+ 0.6 .* rand(rng, 100)); end
    two && for i in 0:2; append!(t, 59002.1 + 7.9i - 0.3 .+ 0.6 .* rand(rng, 100)); end
    sort!(t)
    f = _dr_flux(t, 3.5, 59000.5, 0.1, 9.70, 0.3)
    two && (f .*= _dr_flux(t, 7.9, 59002.1, 0.07, 9.70 * (7.9 / 3.5)^(2 / 3), 0.5))
    f .+= 4e-4 .* randn(rng, length(t))
    return Dict("TESS" => (t = t, flux = f, flux_err = fill(4e-4, length(t))))
end
const _DR_LC1 = _dr_lc()
const _DR_LC2 = _dr_lc(; two = true)
const _DR_PRI1 = Dict("P_k1" => UniformPrior(3.4, 3.6), "rr_k1" => UniformPrior(0.02, 0.3))
const _DR_PRI2 = merge(_DR_PRI1, Dict("P_k2" => UniformPrior(7.8, 8.0),
                                      "rr_k2" => UniformPrior(0.02, 0.3)))
_dr_target(np = 1; pri = np == 1 ? _DR_PRI1 : _DR_PRI2, kw...) =
    Nereus._target_from([Dict{String,Any}("source" => "PM",
                                          "data" => np == 1 ? _DR_LC1 : _DR_LC2)], np;
                        priors = Dict{String,Any}(pri), kw...)

# Two-sample Kolmogorov-Smirnov statistic.
function _dr_ks(a, b)
    a = sort(a); b = sort(b); na, nb = length(a), length(b)
    i = j = 1; d = 0.0
    while i <= na && j <= nb
        a[i] <= b[j] ? (i += 1) : (j += 1)
        d = max(d, abs((i - 1) / na - (j - 1) / nb))
    end
    return d
end

@testset "photo-eccentric reparametrization (density = :rho_circ)" begin

    @testset "ParametrizationConfig" begin
        @test ParametrizationConfig().density === :rho_circ
        @test ParametrizationConfig(density = "rho_star").density === :rho_star
        @test_throws ArgumentError ParametrizationConfig(density = :rho)
        pc = Nereus.resolve_time_anchor(ParametrizationConfig(density = :rho_star), nothing)
        @test pc.density === :rho_star
        @test Nereus._with_t_ref_c(pc, 1.0).density === :rho_star
    end

    @testset "where it applies" begin
        # ρ★ fixed by (M★, R★): the planet samples (ρ_circ, η).
        g = _dr_target(; M_s = 1.0, R_s = 1.0).params
        @test g.layout.density.mode === :given
        @test "rho_circ_k1" in g.layout.unfrozen_names && "eta_k1" in g.layout.unfrozen_names
        @test !("sesinw_k1" in g.layout.unfrozen_names) && !("secosw_k1" in g.layout.unfrozen_names)
        @test Nereus.density_derived_names(g) == ["sesinw_k1", "secosw_k1"]
        @test length(g.layout.unfrozen_names) ==
              length(Nereus.without_density_reparam(g).layout.unfrozen_names)
        # ρ★ sampled: the first planet's ρ_circ replaces it; its e stays.
        a = _dr_target(; use_rho_s = true).params
        @test a.layout.density.mode === :anchor
        @test "rho_circ_k1" in a.layout.unfrozen_names && !("eta_k1" in a.layout.unfrozen_names)
        @test "sesinw_k1" in a.layout.unfrozen_names && !("rho_s" in a.layout.unfrozen_names)
        @test Nereus.density_derived_names(a) == ["rho_s"]
        # Two planets sharing a sampled ρ★: anchor + one (ρ_circ, η) planet.
        a2 = _dr_target(2; use_rho_s = true).params
        @test a2.layout.density.mode === :anchor
        @test "rho_circ_k2" in a2.layout.unfrozen_names && "eta_k2" in a2.layout.unfrozen_names
        @test Nereus.density_derived_names(a2) == ["sesinw_k2", "secosw_k2", "rho_s"]  # layout order
        # Opt-out, a circular orbit, a fixed e, a/R★ sampled, no transits: inert.
        @test !Nereus.density_active(_dr_target(; M_s = 1.0, R_s = 1.0, density = :rho_star).params)
        circ = merge(_DR_PRI1, Dict("sesinw_k1" => FixedPrior(0.0), "secosw_k1" => FixedPrior(0.0)))
        @test !Nereus.density_active(_dr_target(; pri = circ, use_rho_s = true).params)
        half = merge(_DR_PRI1, Dict("secosw_k1" => FixedPrior(0.2)))
        @test !Nereus.density_active(_dr_target(; pri = half, use_rho_s = true).params)
        rng = MersenneTwister(2); t = sort(59000 .+ 100 .* rand(rng, 40))
        rv = Dict("HARPS" => (t = t, rv = 10 .* sin.(2π .* t ./ 7.3) .+ randn(rng, 40),
                              rv_err = fill(1.0, 40)))
        prv = Nereus._target_from([Dict{String,Any}("source" => "RV", "data" => rv)],
                                  (k1 = (P = UniformPrior(5.0, 10.0), K = UniformPrior(0.0, 30.0)),);
                                  M_s = 1.0).params
        @test !Nereus.density_active(prv)
        @test Nereus.without_density_reparam(prv) === prv
        # A prior on a sampled ρ_circ / η slot is refused: theirs is carried.
        @test_throws ArgumentError _dr_target(; pri = merge(_DR_PRI1,
            Dict("rho_circ_k1" => LogUniformPrior(0.1, 10.0))), M_s = 1.0, R_s = 1.0)
    end

    @testset "the Jacobian is the map's" begin
        for ew in (:sesinw, :esinw, :ew)
            t0 = _dr_target(; M_s = 1.0, R_s = 1.0)
            pc = ParametrizationConfig(ew = ew, time = :phic,
                                       t_ref_c = t0.params.config.parametrization.t_ref_c)
            P = Params(max_kplanet = 1, planet_modes = t0.params.config.planet_modes,
                       instruments = t0.params.config.instruments, parametrization = pc,
                       priors = Dict{String,PriorSpec}(_DR_PRI1), data = t0.data,
                       M_s = 1.0, R_s = 1.0, stability = :none)
            tgt = NereusTarget(P, t0.data); d = P.layout.density
            th = Theta{Float64}(P); rng = MersenneTwister(7)
            i1, i2 = d.derived[d.old_e[1][1]], d.derived[d.old_e[1][2]]
            n = 0
            for _ in 1:60
                Nereus.set_unfrozen!(th, Nereus._draw_from_prior(tgt, rng))
                rc, η = th.values[d.rho_circ[1]], th.values[d.eta[1]]
                f(r, h) = (th.values[d.rho_circ[1]] = r; th.values[d.eta[1]] = h;
                           Nereus.density_physical!(th); (th.values[i1], th.values[i2]))
                h1, h2 = 1e-6 * rc, 1e-6
                A = f(rc + h1, η); B = f(rc - h1, η); C = f(rc, η + h2); D = f(rc, η - h2)
                J = [(A[1] - B[1]) / 2h1 (C[1] - D[1]) / 2h2; (A[2] - B[2]) / 2h1 (C[2] - D[2]) / 2h2]
                f(rc, η)
                c = cbrt(rc)                               # ρ★ = 1
                x, y = Nereus._ellipse_xy(c, η)
                0.01 < hypot(x, y) < 0.98 || continue
                _, _, lj = Nereus._old_from_xy(ew, x, y, Nereus._dplan_wlo(d, d.old_e[1][2]))
                @test log(abs(det(J))) ≈ lj + Nereus._ellipse_logjac(c, η) + log(c / 3rc) atol = 1e-6
                n += 1
            end
            @test n > 30
        end
    end

    @testset "log L at matching physical points" begin
        for (np, kw) in ((1, (; M_s = 1.0, R_s = 1.0)), (1, (; use_rho_s = true)),
                         (2, (; M_s = 1.0, R_s = 1.0)), (2, (; use_rho_s = true)))
            tn = _dr_target(np; kw...); pn = tn.params; data = tn.data
            p0 = Nereus.without_density_reparam(pn, data)
            @test !Nereus.density_active(p0)
            to = NereusTarget(p0, data)
            ws(p) = Nereus.PTWorkspace(p, np, 0; n_obs = 0, n_phot = length(data.t_phot))
            wn, wo = ws(pn), ws(p0)
            rng = MersenneTwister(3)
            for _ in 1:80
                x0 = Nereus._draw_from_prior(to, rng)
                xn = Nereus.density_new_x(x0, p0, pn)
                # the round trip, and the same orbit slot for slot
                @test Nereus.density_old_x(xn, pn, p0) ≈ x0 rtol = 1e-10
                tho = Theta{Float64}(p0); Nereus.set_unfrozen!(tho, x0)
                thn = Theta{Float64}(pn); Nereus.set_unfrozen!(thn, xn)
                for k in 1:np
                    @test all(Nereus.planet_e_w(thn, k) .≈ Nereus.planet_e_w(tho, k))
                    @test all(isapprox.(Nereus.planet_geometry(thn, k, 3.5),
                                        Nereus.planet_geometry(tho, k, 3.5); rtol = 1e-12))
                end
                lo = Nereus.transit_log_likelihood(tho, data)
                @test Nereus.transit_log_likelihood(thn, data) ≈ lo rtol = 1e-11
                @test Nereus.transit_log_likelihood(thn, data, wn) ≈
                      Nereus.transit_log_likelihood(tho, data, wo) rtol = 1e-11
            end
        end
    end

    @testset "the prior of ρ★, e, ω and b is the old one" begin
        # An affine-invariant ensemble on log_prior alone, in the new
        # coordinates of the slots the reparametrization couples (and b), the
        # rest held; against i.i.d. draws of the old prior (e < 1).
        function prior_draws(tgt, slots; nw = 48, nstep = 3000, thin = 10, seed = 1)
            p = tgt.params; pt = tgt.transform; rng = MersenneTwister(seed)
            x0 = Nereus._draw_from_prior(tgt, rng)
            pos = [findfirst(==(s), p.layout.unfrozen_names) for s in slots]
            th = Theta{Float64}(p)
            fw(q, x) = Nereus.transform_forward([x], [pt.type_ids[q]], [pt.lowers[q]], [pt.uppers[q]])[1]
            lj(q, y) = Nereus.transform_logabsdetjac_inv([y], [pt.type_ids[q]], [pt.lowers[q]], [pt.uppers[q]])
            function point(y)
                x = copy(x0)
                for (j, q) in enumerate(pos); x[q] = Nereus._bounded_coord(y[j], q, pt); end
                return x
            end
            function lpy(y)
                Nereus.set_unfrozen!(th, point(y))
                l = Nereus.log_prior(th)
                isfinite(l) || return -Inf
                return l + sum(lj(q, y[j]) for (j, q) in enumerate(pos))
            end
            D = length(pos); Y = zeros(D, nw); L = zeros(nw)
            for w in 1:nw
                xx = Nereus._draw_from_prior(tgt, rng)
                Y[:, w] = [fw(q, xx[q]) for q in pos]; L[w] = lpy(Y[:, w])
            end
            out = Vector{Vector{Float64}}()
            for s in 1:nstep, w in 1:nw
                o = rand(rng, 1:nw-1); o >= w && (o += 1)
                z = (rand(rng) + 1)^2 / 2
                yp = Y[:, o] .+ z .* (Y[:, w] .- Y[:, o]); lp = lpy(yp)
                if log(rand(rng)) < (D - 1) * log(z) + lp - L[w]
                    Y[:, w] = yp; L[w] = lp
                end
                (s > nstep ÷ 5 && s % thin == 0) && push!(out, point(Y[:, w]))
            end
            return out, x0
        end
        function phys(p, x)
            th = Theta{Float64}(p); Nereus.set_unfrozen!(th, x)
            K = p.config.max_kplanet
            return (p.config.parametrization.use_rho_s ? Nereus.rho_s(th) : NaN,
                    [Nereus.planet_e_w(th, k) for k in 1:K], [Nereus.planet_b_rr(th, k)[1] for k in 1:K])
        end
        for (np, kw, slots) in ((1, (; M_s = 1.0, R_s = 1.0), ["rho_circ_k1", "eta_k1", "b_k1"]),
                                (1, (; use_rho_s = true), ["rho_circ_k1", "sesinw_k1", "secosw_k1", "b_k1"]),
                                (2, (; use_rho_s = true), ["rho_circ_k1", "sesinw_k1", "secosw_k1",
                                                          "rho_circ_k2", "eta_k2", "b_k2"]))
            tgt = _dr_target(np; kw...); p = tgt.params
            S, x0 = prior_draws(tgt, slots)
            p0 = Nereus.without_density_reparam(p, tgt.data)
            rng = MersenneTwister(99)
            old = Any[]
            oslots = vcat([s for s in slots if !Nereus.is_density_reparam_name(s)],
                          Nereus.density_derived_names(p))
            base = Nereus.density_old_x(x0, p, p0)
            while length(old) < 100_000
                x = copy(base)
                for s in oslots
                    q = findfirst(==(s), p0.layout.unfrozen_names)
                    x[q] = quantile(p0.layout.unfrozen_priors[q], rand(rng))
                end
                th = Theta{Float64}(p0); Nereus.set_unfrozen!(th, x)
                all(Nereus.planet_e_w(th, k)[1] < 1 for k in 1:np) && push!(old, phys(p0, x))
            end
            new = [phys(p, x) for x in S]
            # Critical D at p = 0.001 for the ensemble's draws counted at a quarter
            # of their number (they are correlated), against 1e5 i.i.d. draws.
            n = length(new) ÷ 4; m = length(old)
            dcrit = 1.95 * sqrt((n + m) / (n * m))
            if p.config.parametrization.use_rho_s
                @test _dr_ks(first.(new), first.(old)) < dcrit
            end
            for k in 1:np
                @test _dr_ks([q[2][k][1] for q in new], [q[2][k][1] for q in old]) < dcrit
                @test _dr_ks([q[2][k][2] for q in new], [q[2][k][2] for q in old]) < dcrit
                "b_k$k" in slots &&
                    @test _dr_ks([q[3][k] for q in new], [q[3][k] for q in old]) < dcrit
            end
        end
    end

    @testset "the prior integrates to one over the sampled box" begin
        # The surrogate is uniform in (log ρ_circ, η) on its box, so the mean of
        # exp(density_logprior_correction) over a midpoint grid of that box is
        # the reparametrized prior's mass, which must be the old prior's: 1, as
        # nothing in these e-priors reaches e = 1. Box (esinw, ecosw) priors, flat
        # in (x, y), so the integrand is bounded; |esinw|, |ecosw| ≤ 0.3 stops at
        # e_max = 0.42, whose η is sampled on an arc (`_density_eta_range`), and
        # 0.6 at e_max = 0.85. The grid's error is the box edges' O(1/n).
        box(p, nm) = Nereus.bounds(p.layout.unfrozen_priors[findfirst(==(nm), p.layout.unfrozen_names)])
        t0 = _dr_target(; M_s = 1.0, R_s = 1.0)
        for a in (0.3, 0.6)
            pc = ParametrizationConfig(ew = :esinw, time = :phic,
                                       t_ref_c = t0.params.config.parametrization.t_ref_c)
            pri = merge(_DR_PRI1, Dict("esinw_k1" => UniformPrior(-a, a),
                                       "ecosw_k1" => UniformPrior(-a, a)))
            P = Params(max_kplanet = 1, planet_modes = t0.params.config.planet_modes,
                       instruments = t0.params.config.instruments, parametrization = pc,
                       priors = Dict{String,PriorSpec}(pri), data = t0.data,
                       M_s = 1.0, R_s = 1.0, stability = :none)
            d = P.layout.density
            @test d.mode === :given && d.eta[1] > 0
            ηlo, ηhi = box(P, "eta_k1")
            @test (ηlo, ηhi) == Nereus._density_eta_range(Nereus._density_e_max(
                :esinw, pri["esinw_k1"], pri["ecosw_k1"]))
            @test ηhi - ηlo < 2π
            lo, hi = box(P, "rho_circ_k1")
            th = Theta{Float64}(P)
            Nereus.set_unfrozen!(th, Nereus._draw_from_prior(NereusTarget(P, t0.data), MersenneTwister(4)))
            n = 1200; m = 0.0
            for i in 1:n, j in 1:n
                th.values[d.rho_circ[1]] = exp(log(lo) + (i - 0.5) / n * log(hi / lo))
                th.values[d.eta[1]] = ηlo + (j - 0.5) / n * (ηhi - ηlo)
                Δ = Nereus.density_logprior_correction(th)
                isfinite(Δ) && (m += exp(Δ))
            end
            @test m / n^2 ≈ 1 atol = 5e-3
        end

        # The sesinw box the GP-on-photometry test uses: every draw of the old
        # prior maps inside the arc, at a finite prior.
        pri = merge(_DR_PRI1, Dict("sesinw_k1" => UniformPrior(-0.3, 0.3),
                                   "secosw_k1" => UniformPrior(-0.3, 0.3)))
        tg = _dr_target(; pri, M_s = 1.0, R_s = 1.0); pg = tg.params
        p0 = Nereus.without_density_reparam(pg, tg.data); t0 = NereusTarget(p0, tg.data)
        ηlo, ηhi = box(pg, "eta_k1")
        @test (ηlo, ηhi) == Nereus._density_eta_range(0.18)
        q = findfirst(==("eta_k1"), pg.layout.unfrozen_names)
        th = Theta{Float64}(pg); rng = MersenneTwister(5)
        for _ in 1:2000
            xn = Nereus.density_new_x(Nereus._draw_from_prior(t0, rng), p0, pg)
            @test ηlo < xn[q] < ηhi
            Nereus.set_unfrozen!(th, xn)
            @test isfinite(Nereus.log_prior(th))
        end

        # The anchor: ∫ π(ρ★) dρ★ over ρ_circ's box at a fixed orbit.
        ta = _dr_target(; use_rho_s = true); pa = ta.params; da = pa.layout.density
        lo, hi = box(pa, "rho_circ_k1")
        th = Theta{Float64}(pa)
        for seed in 1:3
            Nereus.set_unfrozen!(th, Nereus._draw_from_prior(ta, MersenneTwister(seed)))
            n = 200_000; m = 0.0
            for i in 1:n
                th.values[da.rho_circ[1]] = exp(log(lo) + (i - 0.5) / n * log(hi / lo))
                Δ = Nereus.density_logprior_correction(th)
                isfinite(Δ) && (m += exp(Δ))
            end
            @test m / n ≈ 1 atol = 5e-3
        end
    end

    @testset "outside the support: -Inf, never NaN" begin
        pri = merge(_DR_PRI1, Dict("rho_s" => NormalPrior(1.0, 0.1, 0.5, 1.5)))
        tgt = _dr_target(; pri, use_rho_s = true); p = tgt.params; d = p.layout.density
        th = Theta{Float64}(p)
        Nereus.set_unfrozen!(th, Nereus._draw_from_prior(tgt, MersenneTwister(5)))
        e, w = Nereus.planet_e_w(th, 1)
        g = Nereus._duration_factor(e, sin(w))
        for (rho, ok) in ((0.3, false), (0.49, false), (0.51, true), (1.0, true),
                          (1.49, true), (1.6, false), (1e-30, false), (1e30, false))
            th.values[d.rho_circ[1]] = rho * g^3
            lp = Nereus.log_prior(th)
            @test !isnan(lp)
            @test isfinite(lp) == ok
            @test Nereus.rho_s(th) ≈ rho rtol = 1e-12
        end
        # an anchor orbit with e ≥ 1, and an η beyond its interval
        th.values[d.rho_circ[1]] = g^3
        th.values[p.layout.name_to_idx["sesinw_k1"]] = 0.9
        th.values[p.layout.name_to_idx["secosw_k1"]] = 0.9
        @test Nereus.log_prior(th) == -Inf
        tg = _dr_target(; M_s = 1.0, R_s = 1.0); pg = tg.params; dg = pg.layout.density
        tt = Theta{Float64}(pg); Nereus.set_unfrozen!(tt, Nereus._draw_from_prior(tg, MersenneTwister(6)))
        tt.values[dg.eta[1]] = 3π / 2 + 0.1
        @test Nereus.log_prior(tt) == -Inf
        lp, ll = Nereus._logdensity_parts(NereusTarget(pg, tg.data; unconstrained = false),
                                          Nereus.unfrozen_values(tt))
        @test lp == -Inf && !isnan(ll)
    end

    @testset "names, chains and summaries" begin
        tgt = _dr_target(; M_s = 1.0, R_s = 1.0); p = tgt.params
        th = Theta{Float64}(p)
        rng = MersenneTwister(8)
        X = reduce(vcat, permutedims(Nereus._draw_from_prior(tgt, rng)) for _ in 1:50)
        cube = cat(X[1:25, :], X[26:50, :]; dims = 3)        # iter × param × chain
        ch = MCMCChains.Chains(cube, Symbol.(p.layout.unfrozen_names))
        ch2 = Nereus.with_density_columns(ch, p)
        @test Set(names(ch2, :parameters)) ==
              union(Set(Symbol.(p.layout.unfrozen_names)), Set([:sesinw_k1, :secosw_k1]))
        V = ch2.value.data; nm = names(ch2)
        for c in 1:2, i in 1:25
            Nereus.set_unfrozen!(th, [V[i, findfirst(==(Symbol(n)), nm), c] for n in p.layout.unfrozen_names])
            e, w = Nereus.planet_e_w(th, 1)
            se = V[i, findfirst(==(:sesinw_k1), nm), c]; sc = V[i, findfirst(==(:secosw_k1), nm), c]
            @test se^2 + sc^2 ≈ e rtol = 1e-10
            @test atan(se, sc) ≈ w atol = 1e-10
            # get_param reads a derived slot at the sampled point; set_param!
            # writes one as the physical coordinate it is
            @test Nereus.get_param(th, "sesinw_k1") ≈ se rtol = 1e-10
        end
        Nereus.set_param!(th, "sesinw_k1", 0.3); Nereus.set_param!(th, "secosw_k1", -0.2)
        @test Nereus.planet_e_w(th, 1)[1] ≈ 0.13 rtol = 1e-10
        @test Nereus.with_density_columns(ch2, p) === ch2
        s = Nereus._summarise_params(ch2, p)
        @test haskey(s, "sesinw_k1") && haskey(s, "rho_circ_k1") && haskey(s, "eta_k1")
        mktempdir() do dir
            f = joinpath(dir, "chains.nc")
            Nereus.save_chains(f, ch2, p; data = tgt.data)
            back, meta = Nereus.load_chains(f)
            @test meta["density"] == "rho_circ"
            @test meta["derived_param_names"] == "sesinw_k1,secosw_k1"
            @test Array(back[:sesinw_k1]) ≈ Array(ch2[:sesinw_k1])
            @test Array(back[:rho_circ_k1]) ≈ Array(ch2[:rho_circ_k1])
        end
    end

    @testset "the ρ_circ transform maps its whole box" begin
        tgt = _dr_target(; M_s = 1.0, R_s = 1.0); pt = tgt.transform
        q = findfirst(==("rho_circ_k1"), tgt.params.layout.unfrozen_names)
        @test pt.type_ids[q] == Nereus.TRANSFORM_LOGLOGIT
        lo, hi = pt.lowers[q], pt.uppers[q]
        @test hi / lo > 1e30
        for x in (1e-3, 0.5, 1.0, 2.0, 1e3)
            y = Nereus.transform_forward([x], [pt.type_ids[q]], [lo], [hi])[1]
            @test Nereus.transform_inverse([y], [pt.type_ids[q]], [lo], [hi])[1] ≈ x rtol = 1e-10
            h = 1e-6
            xp = Nereus.transform_inverse([y + h], [pt.type_ids[q]], [lo], [hi])[1]
            xm = Nereus.transform_inverse([y - h], [pt.type_ids[q]], [lo], [hi])[1]
            @test Nereus.transform_logabsdetjac_inv([y], [pt.type_ids[q]], [lo], [hi]) ≈
                  log((xp - xm) / 2h) atol = 1e-6
        end
    end

    @testset "checkpoints" begin
        pn = _dr_target(; M_s = 1.0, R_s = 1.0); po = _dr_target(; M_s = 1.0, R_s = 1.0, density = :rho_star)
        fn = Nereus.run_fingerprint(pn.params, pn.data; seed = 1)
        fo = Nereus.run_fingerprint(po.params, po.data; seed = 1)
        @test fn.density === :rho_circ && !haskey(fo, :density)
        mktempdir() do dir
            f = joinpath(dir, "ck.jls")
            Nereus.write_checkpoint(f, "pt_emcee", fo, (; step = 1))
            err = try Nereus.read_checkpoint(f, "pt_emcee", fn); nothing catch e; e end
            @test err isa ArgumentError && occursin("before this model sampled", err.msg)
            Nereus.write_checkpoint(f, "pt_emcee", fn, (; step = 1))
            err = try Nereus.read_checkpoint(f, "pt_emcee", fo); nothing catch e; e end
            @test err isa ArgumentError && occursin("density = :rho_circ", err.msg)
            @test Nereus.read_checkpoint(f, "pt_emcee", fn).step == 1
        end
        # A model the reparametrization leaves alone keeps its fingerprint.
        rng = MersenneTwister(2); t = sort(59000 .+ 100 .* rand(rng, 40))
        rv = Dict("HARPS" => (t = t, rv = 10 .* sin.(2π .* t ./ 7.3) .+ randn(rng, 40),
                              rv_err = fill(1.0, 40)))
        fr(d) = (tg = Nereus._target_from([Dict{String,Any}("source" => "RV", "data" => rv)],
                     (k1 = (P = UniformPrior(5.0, 10.0), K = UniformPrior(0.0, 30.0)),);
                     M_s = 1.0, density = d); Nereus.run_fingerprint(tg.params, tg.data; seed = 1))
        @test fr(:rho_circ) == fr(:rho_star)
    end

    @testset "engines that run ρ★, (e, ω)" begin
        tgt = _dr_target(; M_s = 1.0, R_s = 1.0)
        t0 = Nereus.without_density_reparam(tgt)
        @test !Nereus.density_active(t0.params) && t0.transform !== nothing
        @test "sesinw_k1" in t0.params.layout.unfrozen_names
        @test all(in(Nereus._DENSITY_FREE_ENGINES), ("pt", "transdim_pt_emcee", "rjmcmc", "moms",
                                                     "daedalus", "ess"))
        # MAP: found in ρ★, (e, ω) and returned as a point of this target.
        r = Nereus.sample_map(tgt; n_starts = 1, maxiter = 30)
        th = Theta{Float64}(tgt.params); Nereus.set_unfrozen!(th, r.x_map)
        @test r.param_names == tgt.params.layout.unfrozen_names
        @test isfinite(Nereus.log_prior(th))
        # run_job hands such an engine, and everything written from its chains,
        # the model in ρ★ and (e, ω): the chains file names what it holds.
        lc = _DR_LC1["TESS"]
        u(lo, hi) = Dict("type" => "UniformPrior", "args" => [lo, hi])
        cfg = Dict{String,Any}("seed" => 1, "output_dir" => mktempdir(),
            "star" => Dict("M_s" => 1.0, "R_s" => 1.0),
            "priors" => Dict("P_k1" => u(3.4, 3.6), "rr_k1" => u(0.02, 0.3)),
            "model" => Dict("max_kplanet" => 1, "planet_modes" => ["PM_ONLY"]),
            "data" => Dict("transit_photometry" => [Dict("instrument" => "TESS",
                "values" => Dict("bjd" => lc.t, "flux" => lc.flux, "flux_err" => lc.flux_err))]),
            "sampler" => Dict("name" => "ess", "kwargs" => Dict("n_samples" => 40,
                              "n_burnin" => 20, "n_chains" => 2)),
            "output" => Dict("plots" => String[], "ppc" => false,
                             "detection_limits" => false, "loo" => false))
        Nereus.run_job(cfg)
        ch, meta = Nereus.load_chains(joinpath(cfg["output_dir"], "chains.nc"))
        @test !haskey(meta, "density") && !haskey(meta, "derived_param_names")
        @test :sesinw_k1 in names(ch) && !(:rho_circ_k1 in names(ch))
        @test occursin("sesinw_k1", meta["param_names"])
    end
end
