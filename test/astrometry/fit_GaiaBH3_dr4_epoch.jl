#!/usr/bin/env julia
# Recover the Gaia BH3 astrometric orbit — a 33 M_sun dormant BLACK HOLE — from
# Gaia DR4 (pre-release) epoch astrometry ALONE, no RV.
#
# This is the third and last orbital system in the 12-source pre-release bundle
# (the other nine are 3 CRF3 QSOs + 6 ordinary stars, all sitting on the ~1 mas
# single-star floor — they are the null controls in
# test/validation/plot_gaia_dr4_epoch_validation.jl, not fit targets).
#
# Why it matters: it extends Nereus's validated companion-mass range over four
# decades — Gaia-4 b at 11 M_Jup, HD 114762 B at 0.14 M_sun, BH3 at 32 M_sun —
# and it is the largest astrometric signal in the bundle by far (81 mas
# single-star post-fit RMS vs a ~1 mas floor, i.e. 78×).
#
# SAME DATA as the discovery. Panuzzo et al. 2024 (A&A, arXiv:2404.10486) found
# BH3 in these very abscissae: their window is JD 2456941.6–2458869.4, ours is
# MJD 56958–58819. Their Fig. 2 shows a single-star model leaving ±15 mas AL
# residuals and the binary model ±0.5 mas.
#
# Published astrometric-only solution (their Table 2, centre column):
#   P = 4194.7 ± 112.3 d   e = 0.7262 ± 0.0056   i = 110.659 ± 0.107 deg
#   a0 = 27.07 ± 0.56 mas  Omega = 136.200 deg   omega = 77.77 deg
#   f_M = 32.03 ± 0.64 M_sun            parallax = 1.6747 ± 0.0094 mas
#   visible giant M_* = 0.76 ± 0.05 M_sun (their Table 3)
#
# NOTE the period is 11.5 yr against a 5.1 yr baseline — less than half an
# orbit. It is nevertheless constrained because e = 0.73 puts periastron
# (2018.2, JD 2458177) INSIDE the window, and that is where essentially all the
# angular motion happens. Expect P and a to be the loosest recovered elements.
#
# Usage:  julia -t auto --project=. test/astrometry/fit_GaiaBH3_dr4_epoch.jl
#   env NEREUS_GAIA_DR4_XML=<path>  reuse a local VOTable (else it fetches)
#   env BH3_ROUNDS=<n>               PT rounds (default 12; 3 = smoke)

using Nereus
using MCMCChains
using Statistics: median, quantile
using Printf

const BH3_SID = 4318465066420528000
const M_PRI   = 0.76            # M_sun, the visible metal-poor giant
const PLX     = 1.6747          # mas
const PLX_ERR = 0.0094

# Panuzzo+ 2024 Table 2, astrometric solution
const PUB = (P = 4194.7, e = 0.7262, i = 110.659, M = 32.03, a0 = 27.07)

# --- data --------------------------------------------------------------------
xml = get(ENV, "NEREUS_GAIA_DR4_XML", "")
(isempty(xml) || !isfile(xml)) && (xml = fetch_gaia_dr4_prerelease())
src = read_gaia_epoch_votable(xml, BH3_SID)
@printf("Gaia BH3: %d along-scan abscissae, G=%.2f, span %.2f yr, ref (α0,δ0)=(%.4f, %.4f)\n",
        length(src.iad.t), src.g_mag,
        (maximum(src.iad.t) - minimum(src.iad.t)) / 365.25, src.ra0, src.dec0)

# --- model: one dark massive companion, DR4 epoch astrometry only ------------
# a_rel for P=4194.7 d and M_tot=32.8 M_sun is ≈16.3 AU; the prior brackets it
# generously because P is the element the short baseline hurts most.
target = build_target(
    M_pri = M_PRI,
    planets = (BH = (
        a      = LogUniformPrior(3.0, 80.0),      # AU  (truth ≈ 16.3)
        M_sec  = LogUniformPrior(0.5, 100.0),     # M_sun (truth 32.0) — BH regime
        sesinw = UniformPrior(-1.0, 1.0),
        secosw = UniformPrior(-1.0, 1.0),
        Mo     = UniformPrior(0.0, 2π),
        inc    = SinePrior(),                     # i>90° ⇒ retrograde, reachable
        Omega  = UniformPrior(0.0, 2π),
    ),),
    iad = src.iad,
    plx = NormalPrior(PLX, PLX_ERR),
    M_s = M_PRI,
)
println("Free params: ", n_unfrozen(target.params))
println("Unfrozen: ", join(target.params.layout.unfrozen_names, ", "))

nrounds = parse(Int, get(ENV, "BH3_ROUNDS", "12"))
@printf("\nPigeons PT: %d rounds × 8 chains\n", nrounds)
t0 = time()
chains, log_Z = sample_pt(target; n_rounds = nrounds, n_chains = 8, seed = 42,
                          show_report = false)
@printf("Done in %.1f min, log Z = %.2f\n\n", (time() - t0) / 60, log_Z)

OUTDIR = get(ENV, "NEREUS_OUTDIR", joinpath(@__DIR__, "plots_GaiaBH3_dr4"))
mkpath(OUTDIR)
save_chains(joinpath(OUTDIR, "chains.nc"), chains, target.params;
            data = target.data, log_evidence = log_Z)
println("chains → ", joinpath(OUTDIR, "chains.nc"))

# --- posterior → physical ----------------------------------------------------
a_v   = vec(Array(chains[:, :a_k1, :]))
M_sec = vec(Array(chains[:, :M_sec_k1, :]))
ses   = vec(Array(chains[:, :sesinw_k1, :]))
sec   = vec(Array(chains[:, :secosw_k1, :]))
inc_v = vec(Array(chains[:, :inc_k1, :]))
plx_v = vec(Array(chains[:, :plx, :]))
e_v   = ses .^ 2 .+ sec .^ 2
P_d   = [365.25 * sqrt(a_v[j]^3 / (M_PRI + M_sec[j])) for j in eachindex(a_v)]
# photocentre semi-major axis: a0 = a_rel · M_sec/(M_pri+M_sec) · plx  (dark companion)
a0_v  = [a_v[j] * M_sec[j] / (M_PRI + M_sec[j]) * plx_v[j] for j in eachindex(a_v)]

q(x, p) = quantile(x, p)
band(x) = (q(x, 0.5), q(x, 0.84) - q(x, 0.5), q(x, 0.5) - q(x, 0.16))
println("Posterior medians [+1σ, −1σ]      (Panuzzo+ 2024 astrometric):")
let (m, hi, lo) = band(P_d);   @printf("  P_days  = %9.1f [+%.1f, −%.1f]   (4194.7 ± 112.3)\n", m, hi, lo) end
let (m, hi, lo) = band(M_sec); @printf("  M_sec   = %9.2f [+%.2f, −%.2f] M_sun (32.03 ± 0.64)\n", m, hi, lo) end
let (m, hi, lo) = band(e_v);   @printf("  e       = %9.4f [+%.4f, −%.4f]   (0.7262 ± 0.0056)\n", m, hi, lo) end
let (m, hi, lo) = band(rad2deg.(inc_v)); @printf("  i_deg   = %9.3f [+%.3f, −%.3f]   (110.659 ± 0.107)\n", m, hi, lo) end
let (m, hi, lo) = band(a0_v);  @printf("  a0_mas  = %9.2f [+%.2f, −%.2f]   (27.07 ± 0.56)\n", m, hi, lo) end

# --- recovery gate -----------------------------------------------------------
Pm, Mm, em = q(P_d, 0.5), q(M_sec, 0.5), q(e_v, 0.5)
im, a0m = rad2deg(q(inc_v, 0.5)), q(a0_v, 0.5)
ok = true
chk(nm, c) = (global ok; ok &= c; @printf("  [%s] %s\n", c ? "PASS" : "FAIL", nm))
println("\n=== recovery vs published Gaia BH3 (astrometric solution) ===")
chk("P within 10% of 4194.7 d (baseline < half an orbit)", abs(Pm - PUB.P) / PUB.P < 0.10)
chk("M_sec within 15% of 32.03 M_sun — a STELLAR-MASS BH",  abs(Mm - PUB.M) / PUB.M < 0.15)
chk("e within 0.06 of 0.7262",                              abs(em - PUB.e) < 0.06)
chk("i within 6° of 110.659° (or its 69.34° mirror)",
    min(abs(im - PUB.i), abs(im - (180 - PUB.i))) < 6.0)
chk("a0 within 15% of 27.07 mas",                           abs(a0m - PUB.a0) / PUB.a0 < 0.15)
chk("companion is a BLACK HOLE (> 20 M_sun)",               Mm > 20.0)
@printf("\n  → M_sec = %.1f M_⊙ from DR4 abscissae alone (Gaia-4 b: 0.0103, HD 114762 B: 0.139)\n", Mm)
println(ok ? "\n✅ GAIA BH3 — 33 M_SUN BLACK HOLE RECOVERED FROM DR4 EPOCH ASTROMETRY ALONE" :
             "\n❌ Gaia BH3 recovery outside tolerance — investigate")
