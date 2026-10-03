# The workspace transit likelihood must include gravity darkening.
#
# Every sampler that keeps a PTWorkspace per walker -- pt_emcee, ESS, rjmcmc,
# transdim_pt_emcee, nested -- calls `transit_log_likelihood(theta, data, ws)`.
# That method had no gravity darkening at all: at a fixed :GD point it returned
# the same value for every i_star, λ and β, so a GD fit whose cadences are all
# <= 2 min (no supersampling, so the cached path is taken) sampled i_star from
# its prior. The non-workspace method had it right and is the reference here.
#
# Per cadence the two methods now compute a :GD planet's flux with the same
# calls on the same arguments, so they agree bit for bit wherever they also sum
# in the same order: up to `_PHOT_REDUCE_CHUNK` cadences, where both are one
# sequential pass. Above it the non-workspace method sums fixed chunks and the
# workspace one a single pass (an order every non-GD fit shares, so it is left
# as it was), and they agree to rounding. A non-GD planet's row in the
# workspace uses its own closed form for z, which differs from `sky_separation`
# in the last bits, so a fit that mixes :GD and ordinary planets also agrees to
# rounding only.
using Test
using Nereus
using Random

const GDW_BJD0 = 2_460_000.0
const GDW_P, GDW_TC = 4.137, GDW_BJD0 + 1.31

# A contiguous space-based block (instrument 1) and ground nights inside it
# (instrument 2), concatenated, so t_phot is not in time order. `exp1`/`exp2`
# are the exposures in seconds; `nothing` leaves `exposure_times` empty.
function gdw_data(; t1 = 0.0, t2 = 3.0, step1 = 120.0, step2 = 60.0, exp1 = 120.0,
                  exp2 = 60.0, nights = (2.31, 5.12, 8.47), seed = 3)
    rng = MersenneTwister(seed)
    tA = collect(range(GDW_BJD0 + t1, GDW_BJD0 + t2; step = step1 / 86_400))
    tB = Float64[]
    for c in nights
        t1 <= c <= t2 && append!(tB, range(GDW_BJD0 + c - 0.16, GDW_BJD0 + c + 0.16;
                                           step = step2 / 86_400))
    end
    t = vcat(tA, tB)
    inst = vcat(fill(1, length(tA)), fill(2, length(tB)))
    ph = @. mod(t - GDW_TC + GDW_P / 2, GDW_P) - GDW_P / 2
    σ = [i == 1 ? 6e-4 : 1.2e-3 for i in inst]
    flux = @. 1.0 - 0.006 * exp(-0.5 * (ph / 0.04)^2) + σ * randn(rng)
    expo = exp1 === nothing ? Float64[] :
           [i == 1 ? exp1 : exp2 for i in inst] ./ 86_400
    return Data(; t_phot = t, flux = flux, flux_err = σ, phot_inst = inst,
                  exposure_times = expo)
end

function gdw_with_rv(d::Data; seed = 5)
    rng = MersenneTwister(seed)
    t_rv = sort(GDW_BJD0 .+ 40 .* rand(rng, 30))
    rv = 80 .* sin.(2π .* (t_rv .- GDW_BJD0) ./ GDW_P) .+ 5 .* randn(rng, 30)
    return Data(; t_rv = t_rv, rv = rv, rv_err = fill(5.0, 30), rv_inst = fill(1, 30),
                  t_phot = d.t_phot, flux = d.flux, flux_err = d.flux_err,
                  phot_inst = d.phot_inst, exposure_times = d.exposure_times)
end

function gdw_target(modes; data, beta_free = true)
    priors = Dict{String, PriorSpec}("rho_s" => LogUniformPrior(0.05, 5.0),
                                     "v_sin_i_star" => UniformPrior(2_000.0, 150_000.0),
                                     "i_star" => UniformPrior(0.02, π - 0.02))
    for (k, P, Tc) in ((1, GDW_P, GDW_TC), (2, 6.71, GDW_BJD0 + 2.05))
        k <= length(modes) || continue
        merge!(priors, Dict{String, PriorSpec}(
            "P_k$k" => UniformPrior(P - 0.2, P + 0.2),
            "Tc_k$k" => UniformPrior(Tc - 0.3, Tc + 0.3),
            "b_k$k" => UniformPrior(0.0, 1.2), "rr_k$k" => UniformPrior(0.01, 0.25),
            "sesinw_k$k" => UniformPrior(-0.7, 0.7), "secosw_k$k" => UniformPrior(-0.7, 0.7)))
        Nereus.has_gd(modes[k]) && (priors["lambda_k$k"] = UniformPrior(-π, π))
    end
    if beta_free
        priors["gd_beta_TESS"] = UniformPrior(0.08, 0.32)
        priors["gd_beta_GROUND"] = UniformPrior(0.08, 0.32)
    end
    rv = !isempty(data.t_rv)
    ic = rv ? InstrumentConfig(rv = ["HARPS"], pm = ["TESS", "GROUND"]) :
              InstrumentConfig(pm = ["TESS", "GROUND"])
    params = Params(; max_kplanet = length(modes), planet_modes = collect(modes),
                      instruments = ic, data = data,
                      parametrization = ParametrizationConfig(time = :Tc, use_rho_s = true),
                      priors = priors, stability = :none, M_s = 1.6, R_s = 1.47)
    return NereusTarget(params, data)
end

gdw_ws(tg) = Nereus.PTWorkspace(tg.params, tg.params.config.max_kplanet,
                                length(tg.params.config.noise_models);
                                n_obs = length(tg.data.t_rv), n_phot = length(tg.data.t_phot))

# Independent prior draws (bounded space) with a finite log prior.
function gdw_points(tg, n; seed)
    rng = MersenneTwister(seed)
    L = tg.params.layout
    th = Nereus.Theta{Float64}(tg.params)
    pts = Vector{Vector{Float64}}()
    while length(pts) < n
        for (j, idx) in enumerate(L.unfrozen_idx)
            th.values[idx] = Nereus.quantile(L.unfrozen_priors[j], rand(rng))
        end
        isfinite(Nereus.log_prior(th)) && push!(pts, copy(th.values))
    end
    return pts
end

# A transiting point near the synthetic dip, every GD term well away from zero.
function gdw_point!(th, tg)
    vals = Dict("P_k1" => GDW_P, "Tc_k1" => GDW_TC, "b_k1" => 0.3, "rr_k1" => 0.08,
                "rho_s" => 0.3, "sesinw_k1" => 0.1, "secosw_k1" => -0.05,
                "lambda_k1" => 0.7, "v_sin_i_star" => 90_000.0, "i_star" => 0.8,
                "gd_beta_TESS" => 0.2, "gd_beta_GROUND" => 0.12,
                "P_k2" => 6.71, "Tc_k2" => GDW_BJD0 + 2.30, "b_k2" => 0.5, "rr_k2" => 0.06,
                "sesinw_k2" => 0.0, "secosw_k2" => 0.0, "lambda_k2" => -1.9)
    for nm in tg.params.layout.unfrozen_names
        haskey(vals, nm) && Nereus.set_param!(th, nm, vals[nm])
    end
    return th
end

ll_ws(th, tg, ws) = Nereus.transit_log_likelihood(th, tg.data, ws)
ll_direct(th, tg) = Nereus.transit_log_likelihood(th, tg.data)

@testset "workspace transit likelihood with gravity darkening" begin

    @testset "the workspace likelihood sees i_star" begin
        tg = gdw_target([Nereus.PM_GD]; data = gdw_data())
        th = gdw_point!(Nereus.Theta{Float64}(tg.params), tg)
        ws = gdw_ws(tg)
        vals = Float64[]
        for i_star in (0.3, 0.8, 1.4)
            Nereus.set_param!(th, "i_star", i_star)
            v = ll_ws(th, tg, ws)
            push!(vals, v)
            @test isfinite(v)
            @test v === ll_direct(th, tg)
        end
        # One value per i_star: the photometry constrains it.
        @test allunique(vals)
    end

    # Bit-for-bit agreement at independent prior draws, for every way a :GD
    # fit reaches the workspace likelihood. One workspace per configuration,
    # carried across the draws, so every draw refreshes the cache from the
    # previous one; each draw is evaluated twice (the second a cache hit) and
    # every fifth also on a fresh workspace.
    configs = [
        # 2-min and 1-min cadences, both exposures at or under 2 min: cached path.
        ("PM_GD 2-min + 1-min", [Nereus.PM_GD], gdw_data()),
        ("PM_GD, no exposure times", [Nereus.PM_GD], gdw_data(; exp1 = nothing)),
        ("PM_GD, 20-s cadence", [Nereus.PM_GD],
            gdw_data(; t1 = 0.95, t2 = 1.75, step1 = 20.0, exp1 = 20.0, nights = ())),
        # One 10-min band switches on supersampling: the workspace method hands
        # over to the non-workspace one, and must still agree.
        ("PM_GD, 10-min ground band", [Nereus.PM_GD], gdw_data(; exp2 = 600.0)),
        ("RVPM_GD with RVs", [Nereus.RVPM_GD], gdw_with_rv(gdw_data())),
        ("RVPM_RM_GD with RVs", [Nereus.RVPM_RM_GD], gdw_with_rv(gdw_data())),
        ("two :GD planets", [Nereus.PM_GD, Nereus.PM_GD], gdw_data(; seed = 8)),
        ("β fixed at von Zeipel", [Nereus.PM_GD], gdw_data(; seed = 9)),
    ]
    @testset "=== the non-workspace likelihood: $name" for (ci, (name, modes, data)) in
                                                             enumerate(configs)
        tg = gdw_target(modes; data, beta_free = !startswith(name, "β fixed"))
        @test length(data.t_phot) <= Nereus._PHOT_REDUCE_CHUNK
        th = Nereus.Theta{Float64}(tg.params)
        ws = gdw_ws(tg)
        n_eq = n_eq2 = n_eqf = n_fin = 0
        pts = gdw_points(tg, 40; seed = 100 + ci)
        for (k, v) in enumerate(pts)
            th.values .= v
            a = ll_ws(th, tg, ws)
            b = ll_ws(th, tg, ws)
            r = ll_direct(th, tg)
            n_fin += isfinite(r)
            n_eq += a === r
            n_eq2 += b === r
            k % 5 == 0 && (n_eqf += ll_ws(th, tg, gdw_ws(tg)) === r)
        end
        @test n_fin >= 30                 # mostly transiting draws
        @test n_eq == length(pts)
        @test n_eq2 == length(pts)
        @test n_eqf == length(pts) ÷ 5
    end

    # The flux cache must key on every gravity-darkening input. Move one at a
    # time and come back: each value must be the non-workspace value at that
    # point, so a term left out of the key shows up as a stale value.
    @testset "the flux cache keys on i_star, λ, v sin i and β" begin
        tg = gdw_target([Nereus.PM_GD, Nereus.PM_GD]; data = gdw_data())
        th = gdw_point!(Nereus.Theta{Float64}(tg.params), tg)
        ws = gdw_ws(tg)
        base = copy(th.values)
        v0 = ll_ws(th, tg, ws)
        @test v0 === ll_direct(th, tg)
        h0 = copy(ws.transit_flux_hash)
        # Each move changes the keys of exactly the rows whose flux it changes:
        # both rows for a property of the star, one row for a planet's λ.
        @testset "$nm" for (nm, δ, rows) in (("i_star", 0.05, [1, 2]),
                                              ("v_sin_i_star", 7_000.0, [1, 2]),
                                              ("gd_beta_TESS", 0.03, [1, 2]),
                                              ("gd_beta_GROUND", -0.02, [1, 2]),
                                              ("lambda_k1", 0.3, [1]),
                                              ("lambda_k2", -0.4, [2]))
            Nereus.set_param!(th, nm, Nereus.get_param(th, nm) + δ)
            v = ll_ws(th, tg, ws)
            @test v === ll_direct(th, tg)
            @test v != v0
            @test findall(ws.transit_flux_hash[1:2] .!= h0[1:2]) == rows
            th.values .= base
            @test ll_ws(th, tg, ws) === v0
            @test ws.transit_flux_hash == h0
        end
        # A move that leaves every flux row as it was (the TESS jitter) reuses
        # the cached gravity-darkened rows, and still gives the reference value.
        Nereus.set_param!(th, "jitter_TESS", Nereus.get_param(th, "jitter_TESS") + 1e-4)
        v = ll_ws(th, tg, ws)
        @test ws.transit_flux_hash == h0
        @test v === ll_direct(th, tg)
        @test v != v0
    end

    @testset "the ends of i_star are closed on the workspace path too" begin
        tg = gdw_target([Nereus.PM_GD]; data = gdw_data())
        th = gdw_point!(Nereus.Theta{Float64}(tg.params), tg)
        ws = gdw_ws(tg)
        for i_star in (0.0, π)
            Nereus.set_param!(th, "i_star", i_star)
            @test ll_direct(th, tg) == -Inf
            @test ll_ws(th, tg, ws) == -Inf
        end
    end

    # More cadences than one reduction chunk: the two methods sum in different
    # orders and agree to rounding, 1e-12 relative. The 20-s band also puts
    # more than 2000 cadences in the window, the size at which the refresh of a
    # row is split across threads; the value must not depend on that. The
    # mixed fit is held to 1e-10 instead: its ordinary planet's row is the
    # workspace's own z, which already differed from the non-workspace one by
    # up to ~1e-11 relative for an ordinary fit, before gravity darkening.
    @testset "agrees to rounding above one reduction chunk" begin
        for (modes, data, rtol) in (([Nereus.PM_GD], gdw_data(; t2 = 9.0), 1e-12),
                                    ([Nereus.PM_GD], gdw_data(; t1 = 0.4, t2 = 2.2,
                                         step1 = 20.0, exp1 = 20.0, nights = ()), 1e-12),
                                    ([Nereus.PM_GD, PM_ONLY], gdw_data(; seed = 8), 1e-10))
            tg = gdw_target(modes; data)
            @test length(modes) == 2 || length(data.t_phot) > Nereus._PHOT_REDUCE_CHUNK
            th = Nereus.Theta{Float64}(tg.params)
            ws = gdw_ws(tg)
            worst = 0.0
            n_fin = 0
            for v in gdw_points(tg, 25; seed = 7)
                th.values .= v
                a = ll_ws(th, tg, ws)
                r = ll_direct(th, tg)
                @test isfinite(a) == isfinite(r)
                isfinite(r) || continue
                n_fin += 1
                worst = max(worst, abs(a - r) / abs(r))
                # Same bits with the refresh run serially, as inside a sampler
                # task that has every thread busy.
                s = fetch(Threads.@spawn begin
                    Nereus._serial_inner_loops!()
                    ll_ws(th, tg, gdw_ws(tg))
                end)
                @test s === a
            end
            @test n_fin >= 15
            @test worst <= rtol
        end
    end
end
