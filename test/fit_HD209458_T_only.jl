# HD 209458 b — TESS sector 56 transit-only fit.
#
# Reference values (Knutson+ 2007 / Stassun+ 2017):
#   P    = 3.52474859 d
#   Tc   = 2452826.628521 BJD-TDB (1986 transits before sector 56)
#   b    = 0.501 ± 0.013
#   Rp/Rs = 0.12086 (depth ~ 1.46%)
#   M_s  = 1.148 M_sun, R_s = 1.203 R_sun
#
# Pipeline: load TESS s56 PDCSAP CSV → mask transits → Savitzky-Golay
# detrend the out-of-transit baseline → divide out → fit Mandel-Agol
# transit with Kipping (q1, q2) limb darkening via NUTS.

using Nereus
using LogDensityProblems
using Statistics: median, std, quantile
using Printf
using MCMCChains
using Random: MersenneTwister

println("=" ^ 70)
println("HD 209458 b — TESS s56 transit-only fit")
println("=" ^ 70)

# ---- 1. Load LC ----------------------------------------------------
DATADIR = joinpath(@__DIR__, "..", "..", "data", "HD209458")
lc = load_tess_lc(joinpath(DATADIR, "HD209458_tess_s56_lc.csv"))
println("Loaded $(length(lc.t)) PDCSAP points, σ ≈ $(round(std(lc.flux), digits=5))")

# ---- 2. Bin to 10-min cadence (no detrending — PDCSAP is already
#       cotrended by the SPOC pipeline). 5× fewer points than 2-min
#       TESS cadence; HD 209458 b's ~3 hr transit gives 18 in-transit
#       bins, well-resolved.
const P_LIT = 3.52474859
const TC_LIT_BASE_BJD = 2452826.628521   # Knutson+ 2007 BJD-TDB
const TC_LIT_BASE = TC_LIT_BASE_BJD - 2_457_000.0   # convert to TJD
n_transits = round(Int, (mean(lc.t) - TC_LIT_BASE) / P_LIT)
TC_LIT = TC_LIT_BASE + n_transits * P_LIT
@printf("Sector mid-time → expected T0 ≈ %.4f TJD (transit #%d)\n",
        TC_LIT, n_transits)

flat_t, flat_flux, flat_flux_err = let bin_d = 10.0 / (60 * 24)
    bin_id = floor.(Int, (lc.t .- minimum(lc.t)) ./ bin_d)
    unique_bins = sort(unique(bin_id))
    tb  = Vector{Float64}(undef, length(unique_bins))
    fb  = Vector{Float64}(undef, length(unique_bins))
    ferb = Vector{Float64}(undef, length(unique_bins))
    @inbounds for (k, b) in enumerate(unique_bins)
        idxs = findall(==(b), bin_id)
        tb[k]  = mean(lc.t[idxs])
        fb[k]  = mean(lc.flux[idxs])
        ferb[k] = mean(lc.flux_err[idxs]) / sqrt(length(idxs))
    end
    tb, fb, ferb
end
println("Binned PDCSAP to 10-min cadence: $(length(flat_t)) pts " *
        "(σ ≈ $(round(std(flat_flux), digits=6)))")

# ---- 3. Build target -----------------------------------------------
target = build_target(
    M_s = 1.148, R_s = 1.203,
    planets = (b = (
        P  = NormalPrior(P_LIT, 0.005, 3.40, 3.65),     # tight period
        Tc = NormalPrior(TC_LIT, 0.05, TC_LIT - 0.5, TC_LIT + 0.5),
        sesinw = UniformPrior(-0.3, 0.3),                # near-circular
        secosw = UniformPrior(-0.3, 0.3),
        b  = UniformPrior(0.0, 1.0),
        rr = UniformPrior(0.05, 0.20),
    ),),
    phot = (TESS = (
        data    = (t = flat_t, flux = flat_flux, flux_err = flat_flux_err),
        jitter  = LogUniformPrior(1e-5, 1e-2),     # photometry-scale
        offset  = NormalPrior(0.0, 1e-3, -0.01, 0.01),
        q1      = UniformPrior(0.0, 1.0),
        q2      = UniformPrior(0.0, 1.0),
    ),),
)
println("\nFree parameters ($(n_unfrozen(target.params))): ",
        join(target.params.layout.unfrozen_names, ", "))

# ---- 4. Sampler — multi-chain NUTS (parallel via Julia threads) ----
# Run with `julia --threads=auto test/fit_HD209458_T_only.jl` so the
# 4 chains parallelize across cores (~4× wall-clock speedup) and we
# get R-hat convergence diagnostics for free.
println("\nRunning NUTS — 4 chains × (500 warmup + 1000 samples) ...")
println("Threads: $(Threads.nthreads())")
t0 = time()
chains = sample_nuts(target;
                     n_chains  = 4,
                     n_samples = 1000,
                     n_warmup  = 500,
                     seed      = 42,
                     show_report = false)
elapsed = time() - t0
@printf("Done in %.1f min\n\n", elapsed/60)

# ---- 5. Summary ----------------------------------------------------
function summ(name)
    v = vec(Array(chains[:, Symbol(name), :]))
    return quantile(v, 0.16), quantile(v, 0.50), quantile(v, 0.84)
end

println("=" ^ 70)
println("Posterior summary (16/50/84):")
println("=" ^ 70)
for nm in target.params.layout.unfrozen_names
    q16, q50, q84 = summ(nm)
    @printf("  %-15s = %12.6f  [+%.6f, -%.6f]\n",
            nm, q50, q84 - q50, q50 - q16)
end

# Derived
ses_v = vec(Array(chains[:, :sesinw_k1, :]))
sec_v = vec(Array(chains[:, :secosw_k1, :]))
e_v   = ses_v .^ 2 .+ sec_v .^ 2
b_v   = vec(Array(chains[:, :b_k1, :]))
rr_v  = vec(Array(chains[:, :rr_k1, :]))
P_v   = vec(Array(chains[:, :P_k1, :]))

println()
println("Derived (vs Knutson+ 2007):")
@printf("  P        = %.7f d   (lit: 3.5247486 d)\n", quantile(P_v, 0.5))
@printf("  e        = %.4f    (lit: ≈ 0)\n",          quantile(e_v, 0.5))
@printf("  b        = %.3f    (lit: 0.501 ± 0.013)\n", quantile(b_v, 0.5))
@printf("  Rp/Rs    = %.5f   (lit: 0.12086)\n",       quantile(rr_v, 0.5))
@printf("  depth    = %.4f%%  (lit: 1.46%%)\n",        100 * quantile(rr_v, 0.5)^2)
