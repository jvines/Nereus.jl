# Recover the Gaia-4 b astrometric orbit from Gaia DR4 (pre-release) epoch
# astrometry ALONE — no RV. This certifies Nereus's epoch-level astrometry
# path end-to-end: the DR4 per-CCD along-scan abscissae (read by
# read_gaia_epoch_votable → IADData) drive iad_log_likelihood, and a fixed-dim
# PT fit must land on the published Gaia+RV solution.
#
# Published (Stefansson+ 2024, arXiv:2410.05654, Gaia+RV adopted):
#   P = 571.3 d,  e = 0.338,  i = 116.9°,  ω = 180.3°,  Ω = 158.6°
#   M_c = 11.8 M_J,  a0 = 0.279 mas,  π = 13.628 mas,  M_* = 0.644 M_sun
#
# Note i > 90° (retrograde node); inc uses a SinePrior over [0, π] so both
# senses are reachable. Distance/mass scale is pinned by an informative Gaia
# parallax prior (the abscissae alone give a0∝M_c·π, degenerate without π).
#
# Usage:  julia --project=. test/astrometry/fit_Gaia4_dr4_epoch.jl
#   env NEREUS_GAIA_DR4_XML=<path>  to reuse a local VOTable (else it fetches)
#   env GAIA4_ROUNDS=<n>             PT rounds (default 12; use 3 for a smoke)

using Nereus
using MCMCChains
using Statistics: median, quantile
using Printf

const GAIA4_SID = 1457486023639239296
const M_PRI     = 0.644          # M_sun (Stefansson+ 2024)
const PLX_GAIA  = 13.628         # mas
const PLX_ERR   = 0.021

# --- data --------------------------------------------------------------------
xml = get(ENV, "NEREUS_GAIA_DR4_XML", "")
if isempty(xml) || !isfile(xml)
    xml = fetch_gaia_dr4_prerelease()
end
src = read_gaia_epoch_votable(xml, GAIA4_SID)
@printf("Gaia-4: %d along-scan abscissae, G=%.2f, ref (α0,δ0)=(%.4f, %.4f)\n",
        length(src.iad.t), src.g_mag, src.ra0, src.dec0)

# --- model: single astrometric companion, DR4 epoch astrometry only ----------
target = build_target(
    M_pri = M_PRI,
    planets = (b = (
        a      = LogUniformPrior(0.3, 4.0),      # AU  (truth a_rel ≈ 1.16 → P≈571 d)
        M_sec  = LogUniformPrior(0.001, 0.05),   # M_sun (truth 11.8 M_J = 0.0113)
        sesinw = UniformPrior(-1.0, 1.0),
        secosw = UniformPrior(-1.0, 1.0),
        Mo     = UniformPrior(0.0, 2π),
        inc    = SinePrior(),
        Omega  = UniformPrior(0.0, 2π),
    ),),
    iad = src.iad,
    plx = NormalPrior(PLX_GAIA, PLX_ERR),        # informative Gaia parallax
    M_s = M_PRI,
)
println("Free params: ", n_unfrozen(target.params))
println("Unfrozen: ", join(target.params.layout.unfrozen_names, ", "))

# --- fixed-dim PT fit --------------------------------------------------------
nrounds = parse(Int, get(ENV, "GAIA4_ROUNDS", "12"))
@printf("\nPigeons PT: %d rounds × 8 chains\n", nrounds)
t0 = time()
chains, log_Z = sample_pt(target; n_rounds = nrounds, n_chains = 8, seed = 42,
                          show_report = false)
@printf("Done in %.1f min, log Z = %.2f\n\n", (time() - t0) / 60, log_Z)

# Persist the posterior so plots_Gaia4_dr4.jl can render without refitting.
let out = joinpath(@__DIR__, "plots_Gaia4_dr4")
    mkpath(out)
    save_chains(joinpath(out, "chains.nc"), chains, target.params;
                data = target.data, log_evidence = log_Z)
    println("chains → ", joinpath(out, "chains.nc"))
end

# --- posterior → physical ----------------------------------------------------
a_v   = vec(Array(chains[:, :a_k1, :]))
M_sec = vec(Array(chains[:, :M_sec_k1, :]))
ses   = vec(Array(chains[:, :sesinw_k1, :]))
sec   = vec(Array(chains[:, :secosw_k1, :]))
inc_v = vec(Array(chains[:, :inc_k1, :]))
e_v   = ses .^ 2 .+ sec .^ 2
M_J   = M_sec .* 1047.57
P_d   = [365.25 * sqrt(a_v[j]^3 / (M_PRI + M_sec[j])) for j in eachindex(a_v)]

q(x, p) = quantile(x, p)
band(x) = (q(x, 0.5), q(x, 0.84) - q(x, 0.5), q(x, 0.5) - q(x, 0.16))
println("Posterior medians [+1σ, −1σ]      (published):")
let (m, hi, lo) = band(P_d);   @printf("  P_days = %8.1f [+%.1f, −%.1f]   (571.3)\n", m, hi, lo) end
let (m, hi, lo) = band(M_J);   @printf("  M_Jup  = %8.2f [+%.2f, −%.2f]   (11.8)\n", m, hi, lo) end
let (m, hi, lo) = band(e_v);   @printf("  e      = %8.3f [+%.3f, −%.3f]   (0.338)\n", m, hi, lo) end
let (m, hi, lo) = band(rad2deg.(inc_v)); @printf("  i_deg  = %8.1f [+%.1f, −%.1f]   (116.9)\n", m, hi, lo) end

# --- recovery gate -----------------------------------------------------------
Pm, Mm, em, im = q(P_d, 0.5), q(M_J, 0.5), q(e_v, 0.5), rad2deg(q(inc_v, 0.5))
ok = true
chk(nm, c) = (global ok; ok &= c; @printf("  [%s] %s\n", c ? "PASS" : "FAIL", nm))
println("\n=== recovery vs published Gaia-4 b ===")
chk("P within 3% of 571.3 d",          abs(Pm - 571.3) / 571.3 < 0.03)
chk("M within 2 M_J of 11.8",          abs(Mm - 11.8) < 2.0)
chk("e within 0.10 of 0.338",          abs(em - 0.338) < 0.10)
chk("i within 12° of 116.9° (or its 63.1° mirror)",
    min(abs(im - 116.9), abs(im - 63.1)) < 12.0)
println(ok ? "\n✅ GAIA DR4 EPOCH-ASTROMETRY ORBIT RECOVERED" :
             "\n❌ Gaia-4 recovery outside tolerance — investigate")
