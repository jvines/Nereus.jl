#!/usr/bin/env julia
# Regenerate the figures shown in the top-level README.
#
# Two published systems, public data only:
#
#   Gaia-4 b  — joint RV + Gaia DR4 epoch astrometry.
#               RV: Stefansson et al. 2025, AJ 169, 107 (HPF, NEID, FIES).
#               Astrometry: Gaia DR4 pre-release epoch astrometry (RC3,
#               2026-06-26), source 1457486023639239296.
#   WASP-47   — joint RV + transit, four planets (e, b, d transit; c is
#               RV-only). Photometry: K2 Campaign 3 EVEREST long cadence.
#               RV: Neveu-VanMalle et al. 2016 (CORALIE), Dai et al. 2015
#               (PFS), Sinukoff et al. 2017 (HIRES), Vanderburg et al. 2017
#               (HARPS-N), Bryant & Bayliss 2022 (ESPRESSO).
#
# Usage:
#   julia -t auto --project=. docs/readme/make_figures.jl [gaia4|wasp47|all]
#
#   env NEREUS_GAIA_DR4_XML=<path>   reuse a local DR4 VOTable (else fetched)
#   env README_STEPS / README_BURNIN / README_TEMPS / README_WALKERS
#                                    WASP-47 sampler budget (defaults below)
#   env README_WORK=<dir>            where chains go (default: a temp dir)
#
# Figures are written next to this script.

using Nereus
using CairoMakie
using DelimitedFiles
using Statistics: median
using Printf

const HERE  = @__DIR__
const DATA  = normpath(joinpath(HERE, "..", "..", "test", "data"))
const WORK  = get(ENV, "README_WORK", mktempdir())
const STEPS   = parse(Int, get(ENV, "README_STEPS",   "6000"))
const BURNIN  = parse(Int, get(ENV, "README_BURNIN",  "4000"))
const TEMPS   = parse(Int, get(ENV, "README_TEMPS",   "24"))
const WALKERS = parse(Int, get(ENV, "README_WALKERS", "128"))

run_sampler(target; kw...) = sample_pt_emcee(target, target.data;
                                             n_temps = TEMPS, n_walkers = WALKERS,
                                             n_steps = STEPS, n_burnin = BURNIN,
                                             seed = 42, show_progress = false, kw...)

# Plotters write to WORK/models/; the README takes copies under stable names.
keep(src, dst) = cp(joinpath(WORK, "models", src), joinpath(HERE, dst); force = true)

function report(name, res, t0)
    @printf("%s: %d steps in %.1f min, log Z = %.2f, min swap acceptance = %.3f\n",
            name, STEPS, (time() - t0) / 60, res.log_evidence,
            minimum(res.acceptance_swap))
end

# =============================================================================
# Gaia-4 b: RV + DR4 epoch astrometry
# =============================================================================

function gaia4()
    # RV, BJD -> MJD to share the astrometry's time axis.
    tb = Dict{String, Vector{Float64}}()
    for line in eachline(joinpath(DATA, "gaia4_rv.dat"))
        s = strip(line); (isempty(s) || startswith(s, "#")) && continue
        t, v, e, ins = split(s)
        append!(get!(() -> Float64[], tb, ins),
                (parse(Float64, t) - 2_400_000.5, parse(Float64, v), parse(Float64, e)))
    end
    rvset(ins) = let v = reshape(tb[ins], 3, :)
        (data = (t = v[1, :], rv = v[2, :], rv_err = v[3, :]),
         sigma = LogUniformPrior(0.5, 100.0))
    end

    xml = get(ENV, "NEREUS_GAIA_DR4_XML", "")
    (isempty(xml) || !isfile(xml)) && (xml = fetch_gaia_dr4_prerelease())
    src = read_gaia_epoch_votable(xml, 1457486023639239296)

    # Stefansson et al. 2025: M_* = 0.644 M_sun, Gaia parallax 13.628 ± 0.021 mas.
    target = build_target(
        M_pri = 0.644,
        planets = (b = (
            a      = LogUniformPrior(0.3, 4.0),     # AU
            M_sec  = LogUniformPrior(0.001, 0.05),  # M_sun
            sesinw = UniformPrior(-1.0, 1.0),
            secosw = UniformPrior(-1.0, 1.0),
            Mo     = UniformPrior(0.0, 2π),
            inc    = SinePrior(),
            Omega  = UniformPrior(0.0, 2π),
        ),),
        rv  = (HPF = rvset("HPF"), NEID = rvset("NEID"), FIES = rvset("FIES")),
        iad = src.iad,
        plx = NormalPrior(13.628, 0.021),
        M_s = 0.644,
    )
    @printf("Gaia-4: %d RVs, %d DR4 abscissae, %d free parameters\n",
            length(target.data.t_rv), length(src.iad.t), n_unfrozen(target.params))

    # Pigeons, not sample_pt_emcee: from prior draws the ensemble PT settles
    # in an e ≈ 0.96 basin whose likelihood sits below the e ≈ 0.33 solution
    # (with 16 and with 24 temperatures alike).
    t0 = time()
    c, log_Z = sample_pt(target; n_rounds = 12, n_chains = 8, seed = 42,
                         show_report = false)
    @printf("Gaia-4: Pigeons 12 rounds in %.1f min\n", (time() - t0) / 60)
    save_chains(joinpath(WORK, "gaia4_chains.nc"), c, target.params;
                data = target.data, log_evidence = log_Z)

    p, d = target.params, target.data
    plot_epoch_astrometry_orbit(c, p, d; planet_idx = 1, output = WORK)
    plot_rv_phasefold(c, p, d; planet = 1, output = WORK)
    keep("epoch_astrometry_orbit_K1.png", "gaia4_dr4_orbit.png")
    keep("RV_phasefold_K1.png",           "gaia4_rv.png")
end

# =============================================================================
# WASP-47: RV + K2 transits, four planets
# =============================================================================

# Published solution (Vanderburg et al. 2017; Bryant & Bayliss 2022), BJD_TDB. P and Tc centre the
# priors; with K, Rp/R★ and b they locate the transits for clipping and give
# the sampler its starting point. `half` is half the transit window, days.
const W47 = (e = (P = 0.789592,  Tc = 2457011.34862,  K = 4.55,  rr = 0.0146, b = 0.16, half = 0.05),
             b = (P = 4.1591289, Tc = 2457007.932131, K = 140.8, rr = 0.1020, b = 0.14, half = 0.10),
             d = (P = 9.030525,  Tc = 2457006.36927,  K = 4.26,  rr = 0.0288, b = 0.15, half = 0.11))

in_transit(t, pl) = abs.(mod.(t .- pl.Tc .+ pl.P / 2, pl.P) .- pl.P / 2) .< pl.half

function wasp47_target()
    # --- photometry: K2 C3 EVEREST, notch-detrended -------------------------
    lc = load_tess_lc(joinpath(DATA, "wasp47_k2_c03_everest.csv"))
    nd = detrend_notch(lc.t, lc.flux, lc.flux_err; window = 1.0)
    f  = nd.flux_detrended
    # Clip at 5σ about a floor of 1, lowered inside each transit window by the
    # limb-darkened depth (1.3 (Rp/R★)²). `nd.outlier` is a 3.5σ clip about
    # the trend and removes d's and b's transits with the outliers.
    depth = sum(1.3 * pl.rr^2 .* in_transit(lc.t, pl) for pl in W47)
    out   = isfinite.(f) .& (depth .== 0)
    σ     = 1.4826 * median(abs.(f[out] .- median(f[out])))
    fin   = isfinite.(f) .& (f .- 1 .< 5σ) .& (f .- 1 .> -(depth .+ 5σ))
    @printf("WASP-47: %d K2 points kept of %d\n", count(fin), count(isfinite.(f)))
    tph, fph, eph = lc.t[fin], f[fin], lc.flux_err[fin]
    k2_exposure = 29.4 / (60 * 24)                     # long cadence, days

    # --- RV ------------------------------------------------------------------
    raw, hdr = readdlm(joinpath(DATA, "wasp47_rv.csv"), ',', Any, '\n'; header = true)
    col(n) = findfirst(==(n), vec(hdr))
    bjd  = Float64.(raw[:, col("bjd")])
    flag = [strip(string(x)) for x in raw[:, col("flag")]]
    # Drop points the source papers flag (Rossiter–McLaughlin nights and two
    # anomalous ESPRESSO points) and any velocity taken during a transit of b
    # (T14 ≈ 3.6 h): the RM anomaly is not in the model.
    in_b = in_transit(bjd, merge(W47.b, (half = 0.09,)))
    # One unflagged ESPRESSO velocity sits 93 m/s off every orbit; its same-night
    # neighbour 0.13 d earlier fits to a few m/s. Kept, it inflates the ESPRESSO
    # jitter to ~20 m/s and drags P_c to ~613 d.
    bad  = (String.(raw[:, col("instrument")]) .== "ESPRESSO") .& (abs.(bjd .- 2458755.64037) .< 1e-4)
    ok   = (flag .== "") .& .!in_b .& .!bad
    @printf("WASP-47: %d RVs kept of %d (%d flagged, %d during a transit of b, %d outlier)\n",
            count(ok), length(ok), count(flag .!= ""), count(in_b .& (flag .== "")), count(bad))
    bjd, rv, err = bjd[ok], Float64.(raw[ok, col("rv")]), Float64.(raw[ok, col("rv_err")])
    inst_str = String.(raw[ok, col("instrument")])
    inst = sort(unique(inst_str))
    rv_inst = [findfirst(==(s), inst) for s in inst_str]

    data = Data(; t_rv = bjd, rv = rv, rv_err = err, rv_inst = rv_inst,
                t_phot = tph, flux = fph, flux_err = eph,
                phot_inst = ones(Int, length(tph)),
                exposure_times = fill(k2_exposure, length(tph)))

    # --- priors ----------------------------------------------------------------
    # Time anchor φ_c = 2π (Tc − t_ref_c) / P, circular, so the RV-only c has
    # exactly one copy of its orbit in the prior. A Tc window cannot do that
    # while P_c is free. The transiting planets get Tc ± 0.05 d as a phase
    # window; t_ref_c is arbitrary, so pick it near the K2 midpoint such that
    # no window straddles the ±π wrap.
    φδ(pl, t) = (rem2pi(tc_to_phic(pl.Tc, pl.P, t), RoundNearest), 2π * 0.05 / pl.P)
    margin(t) = minimum(π - abs(φ) - δ for (φ, δ) in (φδ(pl, t) for pl in W47))
    cands   = median(tph) .+ (-2.0:0.005:2.0)
    t_ref_c = cands[argmax(margin.(cands))]
    margin(t_ref_c) > 0 || error("no t_ref_c keeps the phase windows off ±π")
    priors = Dict{String, PriorSpec}()
    for (k, (pl, Kmax)) in enumerate(((W47.e, 30.0), (W47.b, 300.0), (W47.d, 30.0)))
        φ, δ = φδ(pl, t_ref_c)
        priors["P_k$k"]      = UniformPrior(pl.P * (1 - 1e-3), pl.P * (1 + 1e-3))
        priors["phic_k$k"]   = UniformPrior(φ - δ, φ + δ)
        priors["K_k$k"]      = UniformPrior(0.0, Kmax)
        priors["sesinw_k$k"] = UniformPrior(-0.7, 0.7)
        priors["secosw_k$k"] = UniformPrior(-0.7, 0.7)
        priors["b_k$k"]      = UniformPrior(0.0, 1.0)
        priors["rr_k$k"]     = UniformPrior(0.005, 0.2)
    end
    # c: non-transiting, ~600 d.
    priors["P_k4"]      = LogUniformPrior(400.0, 800.0)
    priors["phic_k4"]   = UniformPrior(-π, π)
    priors["K_k4"]      = UniformPrior(0.0, 100.0)
    priors["sesinw_k4"] = UniformPrior(-0.9, 0.9)
    priors["secosw_k4"] = UniformPrior(-0.9, 0.9)
    for (i, n) in enumerate(inst)
        v = rv[rv_inst .== i]
        span = max(maximum(v) - minimum(v), 1.0)
        priors["gamma_$n"] = UniformPrior(median(v) - 3span, median(v) + 3span)
        priors["sigma_$n"] = ModJeffreysPrior(0.1, 50.0)
    end
    priors["offset_K2"] = NormalPrior(0.0, 1e-3, -5e-3, 5e-3)
    priors["jitter_K2"] = LogUniformPrior(1e-6, 1e-3)
    priors["q1_K2"]     = UniformPrior(0.0, 1.0)
    priors["q2_K2"]     = UniformPrior(0.0, 1.0)
    priors["rho_s"]     = UniformPrior(0.1, 3.0)          # solar units

    params = Params(;
        max_kplanet     = 4,
        planet_modes    = [RVPM, RVPM, RVPM, RV_ONLY],    # e, b, d, c
        instruments     = InstrumentConfig(rv = inst, pm = ["K2"]),
        data            = data,
        M_s             = 1.04,                           # Vanderburg et al. 2017
        R_s             = 1.137,
        parametrization = ParametrizationConfig(time = :phic, geom = :b_rr,
                                                use_rho_s = true, t_ref_c = t_ref_c),
        priors          = priors,
        external_priors = [ExternalPrior(:ecc, NormalPrior(0.0, 0.3), true)],
    )
    @printf("WASP-47: %d RVs / %d instruments, %d K2 points, %d free parameters\n",
            length(bjd), length(inst), length(tph), n_unfrozen(params))
    return NereusTarget(params, data; unconstrained = true)
end

# Starting point: the literature solution (Vanderburg et al. 2017; Bryant &
# Bayliss 2022), with c's phase and ω from a grid, polished by MAP. Walkers
# started from the prior do not settle in the 43-dimensional joint mode within
# this budget: the K2 transits make the likelihood steep (log L ≈ 2e4), the
# tempered rungs barely swap, and K_e, K_d, P_c and the jitters stay off.
# Stranded-walker pruning is off: every walker starts in a 1e-3 ball about
# one point, so pruning only clones walkers onto each other and the ensemble
# never widens to the posterior (P_c came out at ±0.7 d instead of ±4 d).
function wasp47_start(target)
    nm, pri, tr = target.params.layout.unfrozen_names, target.params.config.priors, target.transform
    x = [(pri[n].lo + pri[n].hi) / 2 for n in nm]    # P, φ, γ: centred priors
    set!(n, v) = (x[findfirst(==(n), nm)] = v)
    for (k, pl) in enumerate(W47)
        set!("K_k$k", pl.K); set!("rr_k$k", pl.rr); set!("b_k$k", pl.b)
    end
    for n in nm; startswith(n, "sigma_") && set!(n, 2.0); end
    set!("jitter_K2", 3e-4); set!("rho_s", 0.71)
    set!("P_k4", 588.8); set!("K_k4", 31.0)
    lpost(θ) = (y = Nereus.transform_forward(θ, tr);
                Nereus.LogDensityProblems.logdensity(target, y) - Nereus.transform_logabsdetjac_inv(y, tr))
    setc!(φ, ω) = (set!("phic_k4", φ); set!("sesinw_k4", √0.3 * sin(ω)); set!("secosw_k4", √0.3 * cos(ω)); x)
    _, φ, ω = maximum((lpost(setc!(φ, ω)), φ, ω)
                      for φ in range(-π, π; length = 73)[1:72], ω in range(0, 2π; length = 25)[1:24])
    setc!(φ, ω)
    m = sample_map(target; init = Nereus.transform_forward(x, tr), n_starts = 1)
    @printf("WASP-47: MAP log posterior %.1f\n", m.log_posterior)
    return m.x_map
end

function wasp47()
    target = wasp47_target()
    params, data = target.params, target.data
    t0 = time()
    res = run_sampler(target; init = wasp47_start(target), prune_stranded = false)
    report("WASP-47", res, t0)
    save_chains(joinpath(WORK, "wasp47_chains.nc"), res.chains, params;
                data = data, log_evidence = res.log_evidence)

    c = res.chains
    # Phase windows hold each full transit: e's 1.4 h spans ±0.038 in phase.
    windows = ((-0.1, 0.1), (-0.04, 0.04), (-0.025, 0.025))
    for (k, name) in enumerate(("e", "b", "d"))
        plot_pm_phasefold(c, params, data; planet = k, output = WORK,
                          phase_window = windows[k])
        keep("Transit_phasefold_K$(k)_K2.png", "wasp47_transit_$name.png")
    end
    for (k, name) in enumerate(("e", "b", "d", "c"))
        plot_rv_phasefold(c, params, data; planet = k, output = WORK)
        keep("RV_phasefold_K$k.png", "wasp47_rv_$name.png")
    end
    return res
end

# =============================================================================

which = isempty(ARGS) ? "all" : ARGS[1]
which in ("gaia4", "all")  && gaia4()
which in ("wasp47", "all") && wasp47()
println("figures in ", HERE, "; chains in ", WORK)
