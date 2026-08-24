# Joint RV + Gaia DR4 epoch-astrometry fit of HD 114762 b — the sin(i) break.
#
# HD 114762 b was the first exoplanet candidate (Latham+ 1989): RV gives
# M sin i ≈ 11 M_J. But RV cannot measure i. Kiefer+ 2019 used Hipparcos–Gaia
# astrometry to show the orbit is near face-on (i ≈ 7°), so the true mass is
# ≈ 0.1 M_sun — a low-mass STAR, not a planet. This fit reproduces that result
# from PUBLIC RVs + the Gaia DR4 (pre-release) epoch astrometry, jointly, in
# Nereus's RVAS mode: RV pins K (∝ M_sec·sin i), the DR4 abscissae pin a0
# (∝ M_sec) and i, and together they separate M_sec from sin i.
#
# Data:
#   RV  — California Legacy Survey (Rosenthal+ 2021, ApJS 255, 8, table6):
#         24 Keck/HIRES + 35 Lick velocities, ~29 yr baseline (public, CDS).
#   AST — Gaia DR4 pre-release epoch astrometry, source 3937211745905473024.
#
# Usage:  julia -t auto --project=. test/astrometry/fit_HD114762_rv_dr4_epoch.jl
#   env NEREUS_GAIA_DR4_XML=<path>   reuse a local VOTable
#   env HD114762_RV=<path>           override the bundled RV file
#                                    (4-col: BJD RV[m/s] eRV[m/s] inst)
#   env HD114762_ROUNDS=<n>          PT rounds (default 12; 3 = smoke)

using Nereus
using MCMCChains
using Statistics: median, quantile
using Printf

const HD114762_SID = 3937211745905473024
const M_PRI  = 0.83          # M_sun (Kiefer+ 2019)
const PLX    = 25.36         # mas (Gaia DR3)
const PLX_ERR = 0.30

# --- RV data (BJD → MJD), split by instrument -------------------------------
# Ships with the package (California Legacy Survey, Rosenthal+ 2021, public),
# so the fit runs with no environment set up. HD114762_RV overrides it.
const _RV_BUNDLED = normpath(joinpath(@__DIR__, "..", "data", "hd114762_rv.dat"))
rvfile = get(ENV, "HD114762_RV", _RV_BUNDLED)
isfile(rvfile) || error("RV file not found: $rvfile (set HD114762_RV to override)")
tb = Dict{String, Vector{Float64}}()
for line in eachline(rvfile)
    s = strip(line); (isempty(s) || startswith(s, "#")) && continue
    t, rv, er, ins = split(s)
    d = get!(() -> Float64[], tb, ins)          # per-instrument column store
    append!(d, (parse(Float64, t) - 2_400_000.5, parse(Float64, rv), parse(Float64, er)))
end
mkrv(ins) = let v = reshape(tb[ins], 3, :)
    (t = v[1, :], rv = v[2, :], rv_err = v[3, :])
end
hires = mkrv("j"); lick = mkrv("lick")
@printf("RV: %d HIRES + %d Lick, baseline %.1f yr\n",
        length(hires.t), length(lick.t),
        (max(maximum(hires.t), maximum(lick.t)) -
         min(minimum(hires.t), minimum(lick.t))) / 365.25)

# --- DR4 epoch astrometry ---------------------------------------------------
xml = get(ENV, "NEREUS_GAIA_DR4_XML", "")
(isempty(xml) || !isfile(xml)) && (xml = fetch_gaia_dr4_prerelease())
src = read_gaia_epoch_votable(xml, HD114762_SID)
@printf("AST: %d DR4 along-scan abscissae\n", length(src.iad.t))

# --- joint RVAS model -------------------------------------------------------
target = build_target(
    M_pri = M_PRI,
    planets = (b = (
        a      = LogUniformPrior(0.30, 0.45),      # AU ⇒ P≈60–105 d, brackets the
                                                   #   35-yr-established P=83.92 d
                                                   #   (targeted mass/i fit, not a
                                                   #    blind period search)
        M_sec  = LogUniformPrior(0.003, 0.5),      # M_sun — allow the stellar regime
        sesinw = UniformPrior(-1.0, 1.0),
        secosw = UniformPrior(-1.0, 1.0),
        Mo     = UniformPrior(0.0, 2π),
        inc    = SinePrior(),
        Omega  = UniformPrior(0.0, 2π),
    ),),
    rv = (
        HIRES = (data = hires, sigma = LogUniformPrior(0.5, 50.0)),
        Lick  = (data = lick,  sigma = LogUniformPrior(0.5, 50.0)),
    ),
    iad = src.iad,
    plx = NormalPrior(PLX, PLX_ERR),
    M_s = M_PRI,
    # LINEAR trend for the wide outer M-dwarf HD 114762 B. NOTE: a *quadratic*
    # trend (d2vdt2) over the 29-yr baseline is a near-unconstrained degenerate
    # direction that produces a railed, mode-less posterior (NaN Laplace
    # evidence, orders-of-magnitude sampler-logZ disagreement) — see
    # diag_HD114762_logz.jl. The correct model for B is a 2nd Keplerian, not a
    # higher-order polynomial. Recovery (P, stellar mass) is robust either way;
    # only the EVIDENCE is broken by the quadratic term.
    trend_order = parse(Int, get(ENV, "HD114762_TREND", "1")),
)
println("Free params: ", n_unfrozen(target.params))
println("Unfrozen: ", join(target.params.layout.unfrozen_names, ", "))

nrounds = parse(Int, get(ENV, "HD114762_ROUNDS", "12"))
@printf("\nPigeons PT: %d rounds × 8 chains\n", nrounds)
t0 = time()
chains, log_Z = sample_pt(target; n_rounds = nrounds, n_chains = 8, seed = 42,
                          show_report = false)
@printf("Done in %.1f min, log Z = %.2f\n\n", (time() - t0) / 60, log_Z)

# Persist the posterior so plots_HD114762_dr4.jl can render without refitting.
# NEREUS_OUTDIR keeps trend_order variants from clobbering each other.
let out = get(ENV, "NEREUS_OUTDIR", joinpath(@__DIR__, "plots_HD114762_dr4"))
    mkpath(out)
    save_chains(joinpath(out, "chains.nc"), chains, target.params;
                data = target.data, log_evidence = log_Z)
    println("chains → ", joinpath(out, "chains.nc"))
end

# --- per-channel likelihood decomposition at the MAP ------------------------
# A pathological log Z means one channel's residuals dwarf its errors. Split
# RV vs astrometry to see which, and report reduced χ² + railed jitters.
let names = target.params.layout.unfrozen_names
    th = Theta(target.params)
    medof = nm -> median(vec(Array(chains[:, Symbol(nm), :])))
    for nm in names
        set_param!(th, string(nm), medof(nm))
    end
    # NOTE: `rv_log_likelihood` is the TOTAL likelihood — likelihood.jl:131
    # returns `ll + lp_ext + ll_astrom + ll_ifloor`. Using it as the "RV
    # channel" double-counts astrometry into RV and inflates RV −2logL/N by
    # ~10× (measured 79.5 vs the true 6.8 on this target, 2026-08-07).
    # `_rv_log_likelihood_core` is the actual RV-only term.
    ll_rv  = Nereus._rv_log_likelihood_core(th, target.data)
    ll_ast = Nereus.astrom_log_likelihood(th, target.data)
    n_rv, n_ast = length(target.data.t_rv), length(src.iad.t)
    @printf("posterior-median per-channel logL:  RV = %.1f (N=%d)   ASTROM = %.1f (N=%d)\n",
            ll_rv, n_rv, ll_ast, n_ast)
    @printf("  → RV −2logL/N ≈ %.1f   ASTROM −2logL/N ≈ %.1f\n",
            -2ll_rv / n_rv, -2ll_ast / n_ast)
    # −2logL/N carries the log(2πσ²) normalisation, so it is NOT a reduced χ².
    # Report χ²/N too — that is the number to judge goodness-of-fit on.
    let (pred, var) = Nereus.rv_predictions(th, target.data)
        chi2 = sum(abs2.(target.data.rv .- pred) ./ var)
        @printf("  → RV χ²/N = %.2f  (this is the goodness-of-fit number)\n", chi2 / n_rv)
    end
    for nm in ("sigma_HIRES", "sigma_Lick")
        Symbol(nm) in names && @printf("  %s(median) = %.2f m/s\n", nm, medof(nm))
    end
end

# --- posterior → physical ---------------------------------------------------
a_v   = vec(Array(chains[:, :a_k1, :]))
M_sec = vec(Array(chains[:, :M_sec_k1, :]))
ses   = vec(Array(chains[:, :sesinw_k1, :]))
sec   = vec(Array(chains[:, :secosw_k1, :]))
inc_v = vec(Array(chains[:, :inc_k1, :]))
e_v   = ses .^ 2 .+ sec .^ 2
M_J   = M_sec .* 1047.57
M_sol = M_sec
Msini = M_J .* sin.(inc_v)                         # what RV-alone would report
P_d   = [365.25 * sqrt(a_v[j]^3 / (M_PRI + M_sec[j])) for j in eachindex(a_v)]
i_deg = rad2deg.(inc_v)
i_fold = [x > 90 ? 180 - x : x for x in i_deg]      # fold to [0,90] for reporting

q(x, p) = quantile(x, p)
band(x) = (q(x, 0.5), q(x, 0.84) - q(x, 0.5), q(x, 0.5) - q(x, 0.16))
println("Posterior medians [+1σ, −1σ]      (Kiefer+ 2019):")
let (m, hi, lo) = band(P_d);    @printf("  P_days   = %8.2f [+%.2f, −%.2f]   (83.92)\n", m, hi, lo) end
let (m, hi, lo) = band(e_v);    @printf("  e        = %8.3f [+%.3f, −%.3f]   (0.335)\n", m, hi, lo) end
let (m, hi, lo) = band(Msini);  @printf("  M sin i  = %8.2f [+%.2f, −%.2f] M_J (≈11, the RV-only value)\n", m, hi, lo) end
let (m, hi, lo) = band(i_fold); @printf("  i_deg    = %8.2f [+%.2f, −%.2f]   (6.2 +1.9 −1.3)\n", m, hi, lo) end
let (m, hi, lo) = band(M_sol);  @printf("  M_true   = %8.3f [+%.3f, −%.3f] M_sun (0.103 +0.030 −0.025, a STAR)\n", m, hi, lo) end
let (m, hi, lo) = band(M_J);    @printf("           = %8.1f [+%.1f, −%.1f] M_J   (108 +31 −26)\n", m, hi, lo) end
# Kiefer+ 2019 (arXiv:1910.07835) abstract: i = 6.2 +1.9/−1.3 deg,
# M_b = 108 +31/−26 M_Jup. Quote the INTERVAL, not "≈0.10 M_sun": the ±28%
# uncertainty is what decides whether a recovery agrees, and comparing against
# the bare central value makes a ~1σ-consistent fit look 40% discrepant.
let mj = q(M_J, 0.5), dev = (mj - 108.0) / (mj > 108 ? 31.0 : 26.0)
    @printf("  → M_true is %+.2fσ from Kiefer+ 2019\n", dev)
end

# --- the point: RV-only M sin i is planetary; joint mass is stellar ----------
Mtrue_med = q(M_sol, 0.5); i_med = q(i_fold, 0.5); msini_med = q(Msini, 0.5)
ok = true
chk(nm, c) = (global ok; ok &= c; @printf("  [%s] %s\n", c ? "PASS" : "FAIL", nm))
println("\n=== sin(i) break: is HD 114762 b a planet or a star? ===")
chk("P within 2% of 83.92 d",                 abs(q(P_d, 0.5) - 83.92) / 83.92 < 0.02)
chk("M sin i ≈ planetary (5–20 M_J)",          5 < msini_med < 20)
chk("inclination low (i < 25°)",               i_med < 25)
chk("true mass STELLAR (> 50 M_J = 0.048 M_⊙)", Mtrue_med * 1047.57 > 50)
@printf("\n  → RV alone: M sin i = %.1f M_J (looks like a giant planet)\n", msini_med)
@printf("  → RV + DR4 astrometry: i = %.1f° ⇒ M = %.3f M_⊙ (a low-mass star)\n", i_med, Mtrue_med)
println(ok ? "\n✅ JOINT RV+DR4-ASTROMETRY REPRODUCES THE HD 114762 b sin(i) BREAK" :
             "\n❌ outside tolerance — investigate")
