# HD 209458 b — joint TESS s56 transit + ELODIE RV fit.
#
# Reference (Knutson+ 2007 / Naef+ 2004 / Stassun+ 2017):
#   P    = 3.52474859 d
#   K    = 84.27 ± 0.70 m/s   (Knutson+ 2007)
#   M_p  = 0.682 ± 0.015 M_J
#   b    = 0.501 ± 0.013
#   Rp/Rs = 0.12086
#
# Pipeline: load TESS s56 LC + Vizier ELODIE RVs, detrend LC, joint
# Mandel-Agol transit + Keplerian RV fit via Pigeons PT.

using Nereus
using Statistics: median, std, quantile
using Printf
using MCMCChains

println("=" ^ 70)
println("HD 209458 b — joint T+RV fit (TESS s56 + ELODIE)")
println("=" ^ 70)

DATADIR = joinpath(@__DIR__, "..", "..", "data", "HD209458")
lc  = load_tess_lc(joinpath(DATADIR, "HD209458_tess_s56_lc.csv"))
rvs = load_vizier_rv(joinpath(DATADIR, "HD209458_rvs.csv"))
println("LC: $(length(lc.t)) pts, σ ≈ $(round(std(lc.flux), digits=5))")
println("RV: $(length(rvs.t)) epochs across $(rvs.instruments)")

# Lit ephemeris in TJD (lightkurve's TESS time scale: BJD − 2,457,000)
const P_LIT = 3.52474859
const TC_LIT = (2452826.628521 - 2_457_000.0) +
               round(Int, (mean(lc.t) -
                           (2452826.628521 - 2_457_000.0)) / P_LIT) * P_LIT
@printf("Sector mid-time → expected T0 ≈ %.4f TJD\n", TC_LIT)

# Bin LC to 10-min cadence (no detrending — PDCSAP is cotrended).
flat = let bd = 10.0 / (60 * 24)
    bin_id = floor.(Int, (lc.t .- minimum(lc.t)) ./ bd)
    ub = sort(unique(bin_id))
    tt = Float64[]; ff = Float64[]; ee = Float64[]
    for b in ub
        idxs = findall(==(b), bin_id)
        push!(tt, mean(lc.t[idxs]))
        push!(ff, mean(lc.flux[idxs]))
        push!(ee, mean(lc.flux_err[idxs]) / sqrt(length(idxs)))
    end
    (t = tt, flux = ff, flux_err = ee)
end
println("Binned LC to $(length(flat.t)) 10-min cadences")

# RV: split per-instrument NamedTuples for build_target's `rv` kwarg
function rv_subset(rvs, inst_label::String, inst_idx::Int)
    mask = rvs.rv_inst .== inst_idx
    return (t = rvs.t[mask], rv = rvs.rv[mask], rv_err = rvs.rv_err[mask])
end
rv_blocks = let
    bl = NamedTuple()
    for (i, name) in enumerate(rvs.instruments)
        sym = Symbol(replace(name, "-" => "_"))
        sub = rv_subset(rvs, name, i)
        length(sub.t) >= 5 || continue
        bl = merge(bl, NamedTuple{(sym,)}(((data = sub,
                                            sigma = LogUniformPrior(0.5, 50.0)),)))
    end
    bl
end

# Build joint target
target = build_target(
    M_s = 1.148, R_s = 1.203,
    planets = (b = (
        P  = NormalPrior(P_LIT, 0.005, 3.40, 3.65),
        K  = LogUniformPrior(10.0, 300.0),
        Tc = NormalPrior(TC_LIT, 0.05, TC_LIT - 0.5, TC_LIT + 0.5),
        sesinw = UniformPrior(-0.15, 0.15),    # tight: HD 209458 known circular
        secosw = UniformPrior(-0.15, 0.15),
        b  = UniformPrior(0.0, 1.0),
        rr = UniformPrior(0.05, 0.20),
    ),),
    rv  = rv_blocks,
    phot = (TESS = (
        data    = flat,
        jitter  = LogUniformPrior(1e-5, 1e-2),
        offset  = NormalPrior(0.0, 1e-3, -0.01, 0.01),
        q1      = UniformPrior(0.0, 1.0),
        q2      = UniformPrior(0.0, 1.0),
    ),),
    # ELODIE baseline spans 7 yr; long-term drift competes with the
    # planetary signal. trend_order=1 lets the fit absorb it.
    trend_order = 1,
)
println("\nFree parameters ($(n_unfrozen(target.params))): ",
        join(target.params.layout.unfrozen_names, ", "))

println("\nRunning NUTS — 4 chains × (500 warmup + 1000 samples) on $(Threads.nthreads()) threads ...")
t0 = time()
chains = sample_nuts(target;
                     n_chains = 4, n_samples = 1000, n_warmup = 500,
                     seed     = 42, show_report = false)
elapsed = time() - t0
@printf("Done in %.1f min\n\n", elapsed/60)
log_Z = NaN  # sample_nuts doesn't return log Z; placeholder for printf below

println("=" ^ 70)
println("Posterior summary (16/50/84):")
println("=" ^ 70)
for nm in target.params.layout.unfrozen_names
    v = vec(Array(chains[:, Symbol(nm), :]))
    q16, q50, q84 = quantile(v, 0.16), quantile(v, 0.50), quantile(v, 0.84)
    @printf("  %-15s = %12.5f  [+%.5f, -%.5f]\n", nm, q50, q84 - q50, q50 - q16)
end

# Derived
P_v   = vec(Array(chains[:, :P_k1, :]))
K_v   = vec(Array(chains[:, :K_k1, :]))
ses   = vec(Array(chains[:, :sesinw_k1, :]))
sec   = vec(Array(chains[:, :secosw_k1, :]))
e_v   = ses .^ 2 .+ sec .^ 2
b_v   = vec(Array(chains[:, :b_k1, :]))
rr_v  = vec(Array(chains[:, :rr_k1, :]))
# Mp: M_p sin i = K · M_s^(2/3) · (P/(2π))^(1/3) / sqrt(1-e²)  (in solar units)
M_J_per_Msun = 1047.57
Mp_MJ = [K_v[j]^(1/3) for j in 1:length(K_v)]   # placeholder; see msec_from_K
# Use the actual mass-function inversion utility from Nereus:
M_PRI = 1.148
sin_i_v = [sin(acos(b_v[j] / (Mp_MJ[j]+1e-9))) for j in 1:length(K_v)]   # rough
# Cleaner: direct mass function call
Mp = Float64[]
for j in 1:length(K_v)
    sini = 1.0  # close to edge-on for a transit
    mp = msec_from_K(K_v[j], P_v[j], e_v[j], sini, M_PRI)
    push!(Mp, mp * M_J_per_Msun)
end

println()
println("Derived (vs Knutson+ 2007 / Naef+ 2004):")
@printf("  P          = %.7f d   (lit: 3.5247486)\n", quantile(P_v, 0.5))
@printf("  K          = %.2f m/s    (lit: 84.27 ± 0.70)\n", quantile(K_v, 0.5))
@printf("  e          = %.4f       (lit: ≈ 0)\n",        quantile(e_v, 0.5))
@printf("  b          = %.3f       (lit: 0.501 ± 0.013)\n", quantile(b_v, 0.5))
@printf("  Rp/Rs      = %.5f      (lit: 0.12086)\n",     quantile(rr_v, 0.5))
@printf("  M_p sin i  = %.3f M_J    (lit: 0.682 ± 0.015)\n", quantile(Mp, 0.5))
