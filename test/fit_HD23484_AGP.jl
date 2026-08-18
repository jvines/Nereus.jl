# HD 23484 — Rajpaul-style multi-indicator ActivityGP on FEROS data.
#
# Reproduces the *method* of Rajpaul+ 2015 (the multi-output GP that couples RV
# with activity indicators through a shared latent G(t) and its derivative) on a
# clean DB target with the SAME indicator trio they used — BIS + FWHM + log R'HK
# — since the actual alpha Cen B data (their target) is fragmented in our DB.
#
# HD 23484: K2.5V, M_s ~= 0.79 Msun, known planet b (P ~ 456 d, e ~ 0.4).
# FEROS RVs are ABSOLUTE (systemic ~= 29.64 km/s); a handful of nights are bad
# reductions (FWHM 13-73 vs normal ~10.1, or wild RV) — cleaned transparently.
#
# Run:  julia --project=. -t 12 Nereus.jl/test/fit_HD23484_AGP.jl

using Nereus, DelimitedFiles, Statistics, Printf, Dates

const FILE = joinpath(@__DIR__, "..", "..", "data", "HD23484", "HD23484_feros.csv")
const M_S  = 0.79

raw = readdlm(FILE, ',', Float64; skipstart = 1)
bjd = raw[:,1]; rv = raw[:,2]; rverr = raw[:,3]
bis = raw[:,4]; biserr = raw[:,5]; fwhm = raw[:,6]; lrhk = raw[:,7]; ha = raw[:,8]
@printf("Loaded %d FEROS epochs\n", length(bjd))

# ---- Clean: drop catastrophic spectra + bad-night RVs --------------------
mad(x) = median(abs.(x .- median(x)))
# 1. bad spectra: FWHM well outside the normal CCF range (normal ~10.0-10.5).
good = (fwhm .> 9.5) .& (fwhm .< 11.0)
# 2. bad-night RVs: robust sigma-clip on RV (systemic-subtracted).
rmed = median(rv[good]); rsig = 1.4826 * mad(rv[good])
good .&= abs.(rv .- rmed) .< 6 * rsig
@printf("Clean: %d / %d kept (robust RV sigma = %.1f m/s; cut FWHM>11 and |dRV|>%.0f m/s)\n",
        sum(good), length(bjd), rsig, 6*rsig)
bjd, rv, rverr, bis, biserr, fwhm, lrhk, ha =
    (v[good] for v in (bjd, rv, rverr, bis, biserr, fwhm, lrhk, ha))

# ---- Center: systemic RV out, indicators to zero-median ------------------
rv   .-= median(rv)            # ~zero-mean residual frame (gamma stays ~0)
bisc  = bis  .- median(bis)
fwhmc = fwhm .- median(fwhm)
lrhkc = lrhk .- median(lrhk)
# Indicator errors: BIS has formal errors; FWHM / logR'HK do not in the DB, so
# assign a measurement floor well below each channel's activity scatter so the
# GP models the (activity) signal, not noise.
σ_fwhm = fill(max(0.02, 0.15 * 1.4826 * mad(fwhmc)), length(fwhmc))
σ_lrhk = fill(max(0.008, 0.15 * 1.4826 * mad(lrhkc)), length(lrhkc))
@printf("RV rms (clean) = %.2f m/s | BIS rms = %.1f | FWHM rms = %.3f | logRhk rms = %.4f\n",
        std(rv), std(bisc), std(fwhmc), std(lrhkc))

# ---- Data + Params (joint Rajpaul: RV + BIS + FWHM + logR'HK) -------------
data = Data(; t_rv = bjd, rv = rv, rv_err = rverr, rv_inst = ones(Int, length(bjd)),
            indicators     = Dict("bis"=>bisc, "fwhm"=>fwhmc, "logrhk"=>lrhkc),
            indicator_errs = Dict("bis"=>biserr, "fwhm"=>σ_fwhm, "logrhk"=>σ_lrhk))
ic = InstrumentConfig(rv = ["FEROS"])
priors = Dict{String, PriorSpec}(
    "P_k1"          => UniformPrior(50.0, 1000.0),    # find HD 23484 b (~456 d)
    "K_k1"          => UniformPrior(0.0, 60.0),
    "gamma_FEROS"   => UniformPrior(-200.0, 200.0),   # RV centered → ~0
    "gp_act_period" => UniformPrior(10.0, 60.0))      # K-dwarf rotation
params = Params(; max_kplanet = 1, planet_modes = [RV_ONLY],
                 instruments = ic, data = data, M_s = M_S,
                 parametrization = ParametrizationConfig(time = :Mo),
                 priors = priors,
                 noise_models = [ActivityGP(channels = [:bis, :fwhm, :logrhk])])
@printf("Params: %d unfrozen\n", n_unfrozen(params))
target = NereusTarget(params, data; unconstrained = false)

OUT = joinpath(@__DIR__, "..", "results",
               "HD23484_AGP_" * Dates.format(Dates.now(), "yyyymmdd_HHMMSS"))
mkpath(OUT)
@printf("\nFitting (joint Rajpaul GP) — 4 temps × 48 walkers × 4000+1000 ...\n")
t0 = time()
res = sample_ptemcee(target, data; n_temps = 4, n_walkers = 48,
                     n_steps = 4000, n_burnin = 1000, seed = 42, show_progress = false)
@printf("Sampler done in %.1f min\n", (time()-t0)/60)
ch = res.chains
q(p) = (v = vec(Array(ch[Symbol(p)]));
        (round(quantile(v,0.16),digits=3), round(median(v),digits=3), round(quantile(v,0.84),digits=3)))
@printf("\n=== HD 23484 b + activity GP ===\n")
@printf("P_k1            = %s d   (HD 23484 b lit ~456 d)\n", q("P_k1"))
@printf("K_k1            = %s m/s\n", q("K_k1"))
@printf("gp_act_period   = %s d   (rotation)\n", q("gp_act_period"))
for p in ["gp_act_amp","Vc","Vr","Bc","Br","Fc","Fr","Lc"]
    p in String.(names(ch,:parameters)) && @printf("%-15s = %s\n", p, q(p))
end
save_chains(joinpath(OUT, "chains.nc"), ch, params; data = data)
try; plot_rv_timeseries(ch, params, data; output = OUT); catch e; @warn "rv_timeseries" exception=e; end
try; plot_activity_gp_latent(ch, params, data; filename = joinpath(OUT,"agp_latent.png")); catch e; @warn "agp_latent" exception=e; end
@printf("\nArtifacts: %s\n", OUT)
