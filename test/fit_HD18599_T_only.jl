#!/usr/bin/env julia
# HD 18599 b — joint Bayesian transit + GP rotation fit on the RAW
# active-star LC. Pipeline (sister to fit_HD18599_search.jl, which is
# the blind-search step):
#
#   1. Notch + TLS on the raw LC ⇒ informative priors on (P, T0, depth)
#      ONLY. The detrended LC is not used for the Bayesian fit.
#   2. Build target with raw LC + Mandel-Agol transit + CeleriteRotationFM17
#      noise model on the photometry channel.
#   3. NUTS — 4 chains, 1000 samples each.
#
# This exercises the GP-on-photometry support added to Nereus in this
# session: `CeleriteRotationFM17(channel=:phot)` plumbs through
# `noise_param_names`, `_default_noise_priors!`, `gp_log_likelihood`
# (with `_phot` suffixed param names), and `transit_log_likelihood`
# (which now hands off to the GP path when a phot-channel
# CovarianceNoise model is active).
#
# Sector 29 only (densest single sector — 95k 20-sec cadences ⇒ bin to
# 10-min for 3500 points; one 25-d span carries 6 transits + ~3 rotation
# cycles, plenty for a transit + GP joint fit).

using Nereus
using TransitLeastSquares
using Statistics: median, mean, std, quantile
using Printf
using MCMCChains

println("=" ^ 70)
println("HD 18599 b — raw LC transit + GP rotation NUTS fit (sector 29)")
println("=" ^ 70)

# ---- 1. Load LC and run notch + TLS for informative priors --------------
const DATADIR = joinpath(@__DIR__, "..", "..", "data", "HD18599")
lc = load_tess_lc(joinpath(DATADIR, "HD18599_tess_s29_SPOC_lc.csv"))
println("Loaded $(length(lc.t)) PDCSAP points (S29, 20s)")

print("Notch + TLS for prior search... ")
t0 = time()
notch = detrend_notch(lc.t, lc.flux, lc.flux_err;
                      window = 1.0,
                      durations = [1.0, 2.0, 3.0, 4.0, 6.0] ./ 24,
                      delta_bic = -1.0)
tls_res = tls(lc.t, notch.flux_detrended;
              flux_err = lc.flux_err,
              period_min = 1.0, period_max = 10.0,
              R_star = 0.798, M_star = 0.807,
              u = [0.5, 0.2],
              oversampling_factor = 3,
              verbose = false)
@printf("done in %.1f s\n", time() - t0)
@printf("TLS prior anchor: P=%.6f d, T0=%.4f BJD, depth=%.4e, SDE=%.1f\n",
        tls_res.period, tls_res.T0 + lc.t[1], tls_res.depth, tls_res.SDE)

# ---- 2. Bin the RAW LC to 10-min cadence --------------------------------
# We feed RAW flux into the transit + GP fit; binning is just for speed.
flat_t, flat_flux, flat_flux_err = let bin_d = 10.0 / (60 * 24)
    bin_id = floor.(Int, (lc.t .- minimum(lc.t)) ./ bin_d)
    unique_bins = sort(unique(bin_id))
    tb  = Vector{Float64}(undef, length(unique_bins))
    fb  = Vector{Float64}(undef, length(unique_bins))
    feb = Vector{Float64}(undef, length(unique_bins))
    @inbounds for (k, b) in enumerate(unique_bins)
        idxs = findall(==(b), bin_id)
        tb[k]  = mean(lc.t[idxs])
        fb[k]  = mean(lc.flux[idxs])           # RAW flux (not detrended)
        feb[k] = mean(lc.flux_err[idxs]) / sqrt(length(idxs))
    end
    tb, fb, feb
end
@printf("Binned RAW LC to 10-min: %d points, σ ≈ %.4e\n",
        length(flat_t), std(flat_flux))

# ---- 3. Build target — transit + GP rotation on photometry --------------
T0_bjd = tls_res.T0 + lc.t[1]
const M_S, R_S = 0.807, 0.798

phot_gp = CeleriteRotationFM17(channel = :phot)

# Tighten GP rotation hyper-priors to break the harmonic multimodality
# (auto-priors are LogUniform[1, 365] d which lets the chain wander to
# 17.5 d, 4.4 d harmonics). HD 18599 has known P_rot ≈ 8.74 d
# (Vines+ 2023), rotation amplitude ~1-2%.
gp_priors = Dict{String, PriorSpec}(
    "gp_log_period_phot"    => NormalPrior(log(8.74), 0.10,
                                            log(4.0), log(20.0)),
    "gp_log_amp_phot"       => UniformPrior(log(1e-4), log(0.05)),
    "gp_log_timescale_phot" => UniformPrior(log(2.0), log(200.0)),
)

target = build_target(
    M_s = M_S, R_s = R_S,
    planets = (b = (
        P  = NormalPrior(tls_res.period, 1e-3,
                          tls_res.period - 0.05, tls_res.period + 0.05),
        Tc = NormalPrior(T0_bjd, 0.05, T0_bjd - 0.5, T0_bjd + 0.5),
        sesinw = UniformPrior(-0.3, 0.3),
        secosw = UniformPrior(-0.3, 0.3),
        b  = UniformPrior(0.0, 1.0),
        rr = UniformPrior(0.005, 0.10),
    ),),
    phot = (TESS = (
        data    = (t = flat_t, flux = flat_flux, flux_err = flat_flux_err),
        jitter  = LogUniformPrior(1e-5, 1e-2),
        offset  = NormalPrior(0.0, 1e-3, -0.01, 0.01),
        q1      = UniformPrior(0.0, 1.0),
        q2      = UniformPrior(0.0, 1.0),
    ),),
    noise_models = [phot_gp],
    priors = gp_priors,
)

println("\nFree parameters ($(n_unfrozen(target.params))):")
println("  ", join(target.params.layout.unfrozen_names, ", "))

# ---- 4. Run affine-invariant ensemble MCMC ------------------------------
# Gradient-free → avoids the ForwardDiff cost through celerite that
# made NUTS take >1 hour. 50 walkers × 3000 steps (1000 burn-in) gives
# ~100k post-burn-in samples per parameter, plenty for the marginals
# we care about (P, b, Rp/Rs, GP P_rot).
println("\nEnsemble MCMC — 50 walkers × 3000 steps (1000 burn-in)")
println("Threads: $(Threads.nthreads())")
t0 = time()
chains = sample_ensemble(target;
                          n_walkers = 50,
                          n_steps   = 3000,
                          n_burnin  = 1000,
                          seed      = 42)
@printf("Done in %.1f min\n", (time() - t0) / 60)

# ---- 5. Posterior summary ----------------------------------------------
function summ(name)
    v = vec(Array(chains[:, Symbol(name), :]))
    return quantile(v, 0.16), quantile(v, 0.50), quantile(v, 0.84)
end

println()
println("=" ^ 70)
println("Posterior summary (16/50/84):")
println("=" ^ 70)
for nm in target.params.layout.unfrozen_names
    q16, q50, q84 = summ(nm)
    @printf("  %-25s = %12.6f  [+%.6f, -%.6f]\n",
            nm, q50, q84 - q50, q50 - q16)
end

ses = vec(Array(chains[:, :sesinw_k1, :]))
sec = vec(Array(chains[:, :secosw_k1, :]))
e_v = ses .^ 2 .+ sec .^ 2
b_v = vec(Array(chains[:, :b_k1, :]))
rr_v = vec(Array(chains[:, :rr_k1, :]))
P_v = vec(Array(chains[:, :P_k1, :]))
gp_log_P_v = vec(Array(chains[:, :gp_log_period_phot, :]))

println()
println("Derived (vs Vines+ 2023, MNRAS 518, 2627):")
@printf("  P            = %.7f d   (lit: 4.1375 d)\n", quantile(P_v, 0.5))
@printf("  e            = %.4f         (lit: ≈ 0)\n", quantile(e_v, 0.5))
@printf("  b            = %.3f         (lit: ~0.55)\n", quantile(b_v, 0.5))
@printf("  Rp/Rs        = %.4f         (lit: 0.0311)\n", quantile(rr_v, 0.5))
@printf("  depth        = %.4e        (lit: 9.7e-4)\n", quantile(rr_v, 0.5)^2)
@printf("  GP P_rot     = %.3f d       (lit: 8.74 d)\n",
        exp(quantile(gp_log_P_v, 0.5)))
