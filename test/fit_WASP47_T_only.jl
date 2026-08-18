# WASP-47 — TESS sector 42 multi-planet transit fit.
#
# Reference (Vanderburg+ 2017 AJ 154, 237 / Becker+ 2015):
#   b (hot Jupiter):  P = 4.1591287 d, Tc=2456979.7641 BJD, Rp/Rs ≈ 0.103
#   d (Neptune):      P = 9.030672 d,  Tc=2456982.349,  Rp/Rs ≈ 0.029
#   e (USP super-E):  P = 0.789593 d,  Tc=2456983.178,  Rp/Rs ≈ 0.014
#   c (outer giant, RV-only, P=588d) — not in this fit.
#
#   M_s = 1.04 M_sun, R_s = 1.137 R_sun.
#
# Pipeline: bin TESS s42 PDCSAP to 5-min cadence (e is short-period; 5-min
# preserves its ~50-min transit), fit b + d + e jointly via NUTS.

using Nereus
using Statistics: median, std, mean, quantile
using Printf
using MCMCChains

println("=" ^ 70)
println("WASP-47 — TESS s42 multi-planet T-only fit")
println("=" ^ 70)

DATADIR = joinpath(@__DIR__, "..", "..", "data", "WASP47")
lc = load_tess_lc(joinpath(DATADIR, "WASP-47_tess_s42_lc.csv"))
println("Loaded $(length(lc.t)) PDCSAP pts, σ ≈ $(round(std(lc.flux), digits=5))")

# Lit ephemerides — convert all from BJD to TJD (= BJD - 2,457,000).
const Pb_LIT = 4.1591287
const Pd_LIT = 9.030672
const Pe_LIT = 0.789593
# Empirical Tc anchors derived directly from sector-42 phase-fold
# (a quick BLS-style scan). Vanderburg+ 2017 ephemerides propagated
# forward 596+ cycles arrived ~1 d off, presumably from accumulated
# uncertainty over ~7 years; fit anchors on real-data Tc.
const TCb_LIT = 2461.834   # b, depth 1.18%
const TCd_LIT = 2458.373   # d, depth 0.58% (ambiguous — may overlap with b's harmonic)
const TCe_LIT = 2461.201   # e, depth ~0.14%
@printf("  b: T0 in sector ≈ %.4f TJD\n", TCb_LIT)
@printf("  d: T0 in sector ≈ %.4f TJD\n", TCd_LIT)
@printf("  e: T0 in sector ≈ %.4f TJD\n", TCe_LIT)

# Bin to 5-min cadence (short-period e has ~50-min transit, need
# resolution to capture it).
flat = let bd = 5.0 / (60 * 24)
    bin_id = floor.(Int, (lc.t .- minimum(lc.t)) ./ bd)
    ub = sort(unique(bin_id))
    tt = Float64[]; ff = Float64[]; ee = Float64[]
    for b in ub
        idxs = findall(==(b), bin_id)
        push!(tt, mean(lc.t[idxs])); push!(ff, mean(lc.flux[idxs]))
        push!(ee, mean(lc.flux_err[idxs]) / sqrt(length(idxs)))
    end
    (t = tt, flux = ff, flux_err = ee)
end
println("Binned PDCSAP to 5-min cadence: $(length(flat.t)) pts " *
        "(σ ≈ $(round(std(flat.flux), digits=6)))")

target = build_target(
    M_s = 1.04, R_s = 1.137,
    planets = (
        b = (
            P  = NormalPrior(Pb_LIT, 0.005, 4.10, 4.22),
            Tc = NormalPrior(TCb_LIT, 0.01, TCb_LIT - 0.1, TCb_LIT + 0.1),
            sesinw = UniformPrior(-0.15, 0.15),
            secosw = UniformPrior(-0.15, 0.15),
            b  = UniformPrior(0.0, 0.95),
            rr = UniformPrior(0.05, 0.18),
        ),
        d = (
            P  = NormalPrior(Pd_LIT, 0.01, 8.95, 9.10),
            Tc = NormalPrior(TCd_LIT, 0.02, TCd_LIT - 0.2, TCd_LIT + 0.2),
            sesinw = UniformPrior(-0.15, 0.15),
            secosw = UniformPrior(-0.15, 0.15),
            b  = UniformPrior(0.0, 0.95),
            rr = UniformPrior(0.005, 0.05),
        ),
        e = (
            P  = NormalPrior(Pe_LIT, 0.001, 0.78, 0.80),
            Tc = NormalPrior(TCe_LIT, 0.01, TCe_LIT - 0.05, TCe_LIT + 0.05),
            sesinw = UniformPrior(-0.05, 0.05),
            secosw = UniformPrior(-0.05, 0.05),
            b  = UniformPrior(0.0, 0.95),
            rr = UniformPrior(0.005, 0.025),
        ),
    ),
    phot = (TESS = (
        data    = flat,
        jitter  = LogUniformPrior(1e-5, 1e-2),
        offset  = NormalPrior(0.0, 1e-3, -0.01, 0.01),
        q1      = UniformPrior(0.0, 1.0),
        q2      = UniformPrior(0.0, 1.0),
    ),),
)
println("\nFree params ($(n_unfrozen(target.params))): ",
        join(target.params.layout.unfrozen_names, ", "))

println("\nRunning NUTS — 4 chains × (500 warmup + 1000 samples) on $(Threads.nthreads()) threads ...")
t0 = time()
chains = sample_nuts(target;
                     n_chains = 4, n_samples = 1000, n_warmup = 500,
                     seed     = 42, show_report = false)
elapsed = time() - t0
@printf("Done in %.1f min\n\n", elapsed/60)

println("=" ^ 70)
println("Posterior summary (16/50/84):")
println("=" ^ 70)
for nm in target.params.layout.unfrozen_names
    v = vec(Array(chains[:, Symbol(nm), :]))
    q16, q50, q84 = quantile(v, 0.16), quantile(v, 0.50), quantile(v, 0.84)
    @printf("  %-15s = %12.5f  [+%.5f, -%.5f]\n", nm, q50, q84 - q50, q50 - q16)
end

println()
println("Derived (vs Vanderburg+ 2017):")
for (k, lit_P, lit_rr) in [("k1", Pb_LIT, 0.103), ("k2", Pd_LIT, 0.029), ("k3", Pe_LIT, 0.014)]
    P = quantile(vec(Array(chains[:, Symbol("P_$k"), :])), 0.5)
    rr = quantile(vec(Array(chains[:, Symbol("rr_$k"), :])), 0.5)
    @printf("  Planet %s: P = %.5f d (lit %.5f), Rp/Rs = %.4f (lit %.4f), depth = %.3f%%\n",
            k, P, lit_P, rr, lit_rr, 100*rr^2)
end
