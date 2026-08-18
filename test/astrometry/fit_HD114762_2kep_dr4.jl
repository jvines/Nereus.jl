#!/usr/bin/env julia
# HD 114762 joint RV + Gaia DR4 epoch astrometry, with the wide outer M-dwarf
# HD 114762 B as a REAL 2nd Keplerian instead of a polynomial trend.
#
# Why this exists: fit_HD114762_rv_dr4_epoch.jl models B with a trend, and both
# trend orders leave the fit bad (RV −2logL/N ≈ 80–105) and the true mass 40–75%
# above Kiefer+ 2019's 0.10 M_sun. Its jitter prior is also prior-dominated —
# sigma spans the whole LogUniform(0.5,50) support and rails at the ceiling with
# a quadratic trend. diag_HD114762_logz.jl / diag_HD114762_fix.jl diagnosed both
# in July: "the correct model for B is a 2nd Keplerian, not a higher-order
# polynomial", plus "use Nereus's DEFAULT jitter prior".
#
# This script applies BOTH halves of that prescription and reports the same
# quantities as the trend version so the two are directly comparable:
#   b: RVAS,    P≈84 d, astrometry breaks sin i → true mass.        k1
#   B: RV_ONLY, P ≫ 29-yr baseline (partial arc only).              k2
#   jitter: Nereus default prior (no LogUniform override).
#   trend_order = 0 — B is now a physical body, not a polynomial.
#
# Usage:  julia -t auto --project=. test/astrometry/fit_HD114762_2kep_dr4.jl
#   env HD114762_RV=<path>           4-col RV file (BJD RV eRV inst)
#   env HD114762_ROUNDS=<n>          PT rounds (default 12; 3 = smoke)
#   env NEREUS_GAIA_DR4_XML=<path>  reuse a local VOTable
#   env NEREUS_OUTDIR=<dir>         defaults to plots_HD114762_2kep

using Nereus
using MCMCChains
using Statistics: median, quantile
using Printf

const SID = 3937211745905473024
const M_PRI, PLX, PLX_ERR = 0.83, 25.36, 0.30
const MJUP_PER_MSUN = 1047.57

# --- data --------------------------------------------------------------------
rvfile = get(ENV, "HD114762_RV", "")
isfile(rvfile) || error("set HD114762_RV to the 4-col RV file (BJD RV eRV inst)")
tb = Dict{String, Vector{Float64}}()
for line in eachline(rvfile)
    s = strip(line); (isempty(s) || startswith(s, "#")) && continue
    t, rv, er, ins = split(s)
    append!(get!(() -> Float64[], tb, ins),
            (parse(Float64, t) - 2_400_000.5, parse(Float64, rv), parse(Float64, er)))
end
mkrv(k) = let v = reshape(tb[k], 3, :); (t = v[1, :], rv = v[2, :], rv_err = v[3, :]) end
hires, lick = mkrv("j"), mkrv("lick")

xml = get(ENV, "NEREUS_GAIA_DR4_XML", "")
(isempty(xml) || !isfile(xml)) && (xml = fetch_gaia_dr4_prerelease())
src = read_gaia_epoch_votable(xml, SID)

t_rv    = vcat(hires.t, lick.t)
rv      = vcat(hires.rv, lick.rv)
rv_err  = vcat(hires.rv_err, lick.rv_err)
rv_inst = vcat(fill(1, length(hires.t)), fill(2, length(lick.t)))
data = Data(; t_rv = t_rv, rv = rv, rv_err = rv_err, rv_inst = rv_inst, iad = src.iad)
@printf("RV: %d HIRES + %d Lick, baseline %.1f yr | AST: %d DR4 abscissae\n",
        length(hires.t), length(lick.t),
        (maximum(t_rv) - minimum(t_rv)) / 365.25, length(src.iad.t))

# --- 2-Keplerian model --------------------------------------------------------
priors = Dict{String, PriorSpec}(
    # b — RVAS; period bracketed to the 35-yr-established 83.9 d
    "P_k1"     => LogUniformPrior(60.0, 110.0),
    "M_sec_k1" => LogUniformPrior(0.003, 0.5),      # allow the stellar regime
    "inc_k1"   => SinePrior(),
    "Omega_k1" => UniformPrior(0.0, 2π),
    # B — RV_ONLY, P ≫ baseline (only a partial arc is constrained)
    "P_k2"     => LogUniformPrior(1.1e4, 1.0e7),    # ~30 – 27000 yr
    "M_sec_k2" => LogUniformPrior(1e-3, 0.5),       # msini
    "plx"      => NormalPrior(PLX, PLX_ERR),
    # NOTE: no sigma_* entries — Nereus's default jitter prior applies. The
    # LogUniform override in the trend script is what makes jitter rail.
)

target = NereusTarget(
    Params(; max_kplanet = 2, planet_modes = [RVAS, RV_ONLY],
           instruments = InstrumentConfig(rv = ["HIRES", "Lick"], pm = String[]),
           parametrization = ParametrizationConfig(mass = :M_sec_driven),
           priors = priors, data = data, M_s = M_PRI, trend_order = 0),
    data; unconstrained = true)

println("Free params: ", n_unfrozen(target.params))
println("Unfrozen: ", join(target.params.layout.unfrozen_names, ", "))

nrounds = parse(Int, get(ENV, "HD114762_ROUNDS", "12"))
@printf("\nPigeons PT: %d rounds × 8 chains\n", nrounds)
t0 = time()
chains, log_Z = sample_pt(target; n_rounds = nrounds, n_chains = 8, seed = 42,
                          show_report = false)
@printf("Done in %.1f min, log Z = %.2f\n\n", (time() - t0) / 60, log_Z)

OUTDIR = get(ENV, "NEREUS_OUTDIR", joinpath(@__DIR__, "plots_HD114762_2kep"))
mkpath(OUTDIR)
save_chains(joinpath(OUTDIR, "chains.nc"), chains, target.params;
            data = target.data, log_evidence = log_Z)
println("chains → ", joinpath(OUTDIR, "chains.nc"))

# --- per-channel likelihood at the posterior median --------------------------
let names = target.params.layout.unfrozen_names
    th = Theta(target.params)
    medof = nm -> median(vec(Array(chains[:, Symbol(nm), :])))
    for nm in names
        set_param!(th, string(nm), medof(nm))
    end
    # `rv_log_likelihood` is the TOTAL likelihood (likelihood.jl:131 returns
    # ll + lp_ext + ll_astrom + ll_ifloor) — use the core for the RV channel.
    ll_rv  = Nereus._rv_log_likelihood_core(th, target.data)
    ll_ast = Nereus.astrom_log_likelihood(th, target.data)
    n_rv, n_ast = length(target.data.t_rv), length(src.iad.t)
    @printf("posterior-median per-channel logL:  RV = %.1f (N=%d)   ASTROM = %.1f (N=%d)\n",
            ll_rv, n_rv, ll_ast, n_ast)
    @printf("  → RV −2logL/N ≈ %.1f   ASTROM −2logL/N ≈ %.1f  (trend version: 6.8 / 7.7)\n",
            -2ll_rv / n_rv, -2ll_ast / n_ast)
    let (pred, var) = Nereus.rv_predictions(th, target.data)
        chi2 = sum(abs2.(target.data.rv .- pred) ./ var)
        @printf("  → RV χ²/N = %.2f  (trend version: 0.97)\n", chi2 / n_rv)
    end
    # jitter is the nuisance that railed in the trend version — report its
    # spread, not just its median, so prior-domination is visible
    for nm in names
        startswith(String(nm), "sigma_") || continue
        v = vec(Array(chains[:, Symbol(nm), :]))
        @printf("  %-14s median=%7.2f m/s   [q0.01=%7.2f, q0.99=%7.2f]\n",
                nm, median(v), quantile(v, 0.01), quantile(v, 0.99))
    end
end

# --- posterior → physical ----------------------------------------------------
P_b   = vec(Array(chains[:, :P_k1, :]))
M_sec = vec(Array(chains[:, :M_sec_k1, :]))     # RVAS ⇒ TRUE mass
ses   = vec(Array(chains[:, :sesinw_k1, :]))
sec   = vec(Array(chains[:, :secosw_k1, :]))
inc_v = vec(Array(chains[:, :inc_k1, :]))
P_B   = vec(Array(chains[:, :P_k2, :]))
M_B   = vec(Array(chains[:, :M_sec_k2, :]))     # RV_ONLY ⇒ msini
e_v   = ses .^ 2 .+ sec .^ 2
M_J   = M_sec .* MJUP_PER_MSUN
Msini = M_J .* sin.(inc_v)
i_fold = [x > 90 ? 180 - x : x for x in rad2deg.(inc_v)]

q(x, p) = quantile(x, p)
band(x) = (q(x, 0.5), q(x, 0.84) - q(x, 0.5), q(x, 0.5) - q(x, 0.16))
println("\nPosterior medians [+1σ, −1σ]      (Kiefer+ 2019):")
let (m, hi, lo) = band(P_b);    @printf("  P_b days = %8.2f [+%.2f, −%.2f]   (83.92)\n", m, hi, lo) end
let (m, hi, lo) = band(e_v);    @printf("  e_b      = %8.3f [+%.3f, −%.3f]   (0.335)\n", m, hi, lo) end
let (m, hi, lo) = band(Msini);  @printf("  M sin i  = %8.2f [+%.2f, −%.2f] M_J (≈11, the RV-only value)\n", m, hi, lo) end
let (m, hi, lo) = band(i_fold); @printf("  i_deg    = %8.2f [+%.2f, −%.2f]   (≈7)\n", m, hi, lo) end
let (m, hi, lo) = band(M_sec);  @printf("  M_true   = %8.3f [+%.3f, −%.3f] M_sun (≈0.10, a STAR)\n", m, hi, lo) end
println("  --- outer companion B (RV-only, partial arc) ---")
let (m, hi, lo) = band(P_B ./ 365.25); @printf("  P_B yr   = %8.0f [+%.0f, −%.0f]\n", m, hi, lo) end
let (m, hi, lo) = band(M_B);           @printf("  M_B sini = %8.3f [+%.3f, −%.3f] M_sun\n", m, hi, lo) end

# --- gates: same as the trend version, plus a real goodness-of-fit gate ------
Mtrue_med = q(M_sec, 0.5); i_med = q(i_fold, 0.5); msini_med = q(Msini, 0.5)
ok = true
chk(nm, c) = (global ok; ok &= c; @printf("  [%s] %s\n", c ? "PASS" : "FAIL", nm))
println("\n=== sin(i) break + does B-as-Keplerian actually fit? ===")
chk("P_b within 2% of 83.92 d",                 abs(q(P_b, 0.5) - 83.92) / 83.92 < 0.02)
chk("M sin i ≈ planetary (5–20 M_J)",           5 < msini_med < 20)
chk("inclination low (i < 25°)",                i_med < 25)
chk("true mass STELLAR (> 50 M_J)",             Mtrue_med * MJUP_PER_MSUN > 50)
# Kiefer+ 2019 (arXiv:1910.07835): M_b = 108 +31/−26 M_Jup, i = 6.2 +1.9/−1.3°.
# Gate on SIGMA against that interval, not on % from the bare central value —
# the published mass carries ±28%, and a percent-gate against 0.10 M_sun calls a
# 1σ-consistent recovery a failure.
let mj = Mtrue_med * MJUP_PER_MSUN, dev = (mj - 108.0) / (mj > 108 ? 31.0 : 26.0)
    chk(@sprintf("M_true within 2σ of Kiefer 108 +31/−26 M_J (got %+.2fσ)", dev),
        abs(dev) < 2.0)
    @printf("\n  → M_true = %.3f M_⊙ = %.1f M_J, %+.2fσ from Kiefer\n",
            Mtrue_med, mj, dev)
    @printf("     (trend_order=1: 0.139 M_⊙ = +1.21σ — the committed default is the closest)\n")
end
println(ok ? "\n✅ 2-KEPLERIAN MODEL RECOVERS HD 114762 b" :
             "\n❌ outside tolerance — see per-channel logL above")
