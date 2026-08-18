# HD 18599 — AGP-only fixed-noise fit. Trans-dim noise selection
# (fit_HD18599_activity_model_select.jl) fails to converge on the AGP branch
# because PT hot chains drift through unconstrained inactive-slot
# parameter space; this script runs AGP as the SOLE active noise
# model so every walker is constrained by the AGP likelihood from
# step zero. Output is the K_k1 posterior recovered under the full
# multi-indicator Rajpaul model — directly comparable to a separate
# AD-only run (and to Vines+ 2023 / Desidera+ 2023).

using Nereus
using DelimitedFiles
using Statistics: median, mean, std, quantile
using Printf
using MCMCChains
using Random
import Dates

const RV_FILE = joinpath(@__DIR__, "data", "hd18599.csv")
const P_REF   = 4.1374685534602405
const T0_REF  = 2.4583545857470357e6
const M_S, R_S = 0.807, 0.798
const P_ROT   = 8.74
const PAPER_INSTRUMENTS = Set(["HARPS_PRE", "HARPS_POST", "FEROS"])

println("="^70)
println("HD 18599 — AGP-only fixed-noise (multi-indicator Rajpaul)")
println("="^70)

# ---- RV + indicators load (shared with the model-select script) ---------------
raw = readdlm(RV_FILE, ',', Any, '\n'; header = true)
dm = raw[1]
inst_raw = String.(dm[:, 16]); prov_raw = String.(dm[:, 17])
keep = Int[]; inst_str = String[]
for i in 1:size(dm, 1)
    ins, prov = strip(inst_raw[i]), strip(prov_raw[i])
    if ins == "HARPS_POST" && prov == "ESO_PHASE3"; continue; end
    ins in PAPER_INSTRUMENTS || continue
    push!(keep, i); push!(inst_str, ins)
end
dm = dm[keep, :]
bjd = Float64.(dm[:, 1]); rv = Float64.(dm[:, 2]); rv_err = Float64.(dm[:, 3])
inst_names = sort!(unique(inst_str))
inst_map = Dict(name => i for (i, name) in enumerate(inst_names))
rv_inst = [inst_map[s] for s in inst_str]
@printf("Loaded %d RVs across %d instruments\n", length(bjd), length(inst_names))

function _load_indicator(col_v, col_e)
    raw_v = Float64[
        let v = dm[i, col_v]
            v === "" || (v isa AbstractString && strip(v) == "") ? NaN : Float64(v)
        end for i in 1:size(dm, 1)]
    raw_e = Float64[
        let v = dm[i, col_e]
            v === "" || (v isa AbstractString && strip(v) == "") ? NaN : Float64(v)
        end for i in 1:size(dm, 1)]
    v = copy(raw_v); e = copy(raw_e)
    for ins_id in unique(rv_inst)
        idx = findall(==(ins_id), rv_inst)
        fv = filter(k -> isfinite(v[k]), idx)
        fe = filter(k -> isfinite(e[k]) && e[k] > 0, idx)
        if !isempty(fv)
            μ = mean(v[fv])
            for k in idx; v[k] = isfinite(v[k]) ? v[k] - μ : 0.0; end
        end
        if !isempty(fe)
            med = median(e[fe])
            for k in idx; (isfinite(e[k]) && e[k] > 0) || (e[k] = med); end
        else
            for k in idx; e[k] = 1.0; end
        end
    end
    return (centered = v, err = e)
end

ind_bis    = _load_indicator(4, 5)
ind_fwhm   = _load_indicator(6, 7)
ind_halpha = _load_indicator(10, 11)
ind_logrhk = _load_indicator(12, 13)

data = Data(;
    t_rv = bjd, rv = rv, rv_err = rv_err, rv_inst = rv_inst,
    indicators = Dict("bis" => ind_bis.centered,
                       "fwhm" => ind_fwhm.centered,
                       "halpha" => ind_halpha.centered,
                       "logrhk" => ind_logrhk.centered),
    indicator_errs = Dict("bis" => ind_bis.err,
                           "fwhm" => ind_fwhm.err,
                           "halpha" => ind_halpha.err,
                           "logrhk" => ind_logrhk.err),
)
ic = InstrumentConfig(rv = inst_names)

# AGP only — joint LL is faster (1 Cholesky instead of 2) and we're
# not doing AD-vs-AGP head-to-head here; just recovering K under AGP.
agp = ActivityGP(channels = [:bis, :fwhm, :halpha, :logrhk],
                  marginalize_indicators = false)

priors = Dict{String, PriorSpec}()
rv_max = maximum(abs, rv)
priors["P_k1"]      = NormalPrior(P_REF, 1e-3, P_REF - 0.01, P_REF + 0.01)
priors["K_k1"]      = UniformPrior(0.0, 50.0)
priors["sesinw_k1"] = UniformPrior(-1.0, 1.0)
priors["secosw_k1"] = UniformPrior(-1.0, 1.0)
priors["Mo_k1"]     = UniformPrior(0.0, 2π)
for name in inst_names
    priors["gamma_$name"] = UniformPrior(-3 * rv_max, 3 * rv_max)
end
# Tight Rajpaul kernel priors anchored on Vines+ 2023 P_rot.
priors["gp_act_period"]   = NormalPrior(P_ROT, 1.0, 4.0, 16.0)
priors["gp_act_lambda_e"] = LogUniformPrior(5.0, 200.0)
priors["gp_act_lambda_p"] = LogUniformPrior(0.1, 5.0)

params = Params(;
    max_kplanet     = 1, planet_modes = [RV_ONLY],
    instruments     = ic, data = data,
    M_s             = M_S, R_s = R_S,
    parametrization = ParametrizationConfig(time = :Mo),
    priors          = priors,
    noise_models    = [agp],
    transdim_noise  = false,
)
@printf("\nParams built: %d unfrozen params (AGP-only fixed)\n",
         n_unfrozen(params))

const _STAMP = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
OUT_DIR = joinpath(@__DIR__, "..", "results", "HD18599_AGP_only_" * _STAMP)
mkpath(OUT_DIR)
@printf("Output dir: %s\n", OUT_DIR)

target = NereusTarget(params, data; unconstrained = false)
@printf("\nStarting sample_ptemcee (3 temps × 32 walkers × 1500 steps + 500 burnin)...\n")
t_start = time()
res = sample_ptemcee(target, data;
                      n_temps   = 3,
                      n_walkers = 32,
                      n_steps   = 1500,
                      n_burnin  = 500,
                      seed      = 42,
                      show_progress = true)
chains = res.chains
@printf("Sampler done in %.1f min\n", (time() - t_start) / 60)

save_chains(joinpath(OUT_DIR, "chains.nc"), chains, params; data = data)

@printf("\nGenerating plots...\n")
try
    plot_activity_gp_latent(chains, params, data;
                              filename = joinpath(OUT_DIR, "activity_gp_latent.png"),
                              n_draws = 200)
    @printf("  ✓ activity_gp_latent.png\n")
catch err; @warn "latent failed" exception = err; end
try
    plot_activity_gp_decomposition(chains, params, data;
                                     filename = joinpath(OUT_DIR, "activity_gp_decomposition.png"),
                                     n_draws = 200)
    @printf("  ✓ activity_gp_decomposition.png\n")
catch err; @warn "decomp failed" exception = err; end
try
    plot_rv_timeseries(chains, params, data; output = OUT_DIR)
    @printf("  ✓ rv_timeseries\n")
catch err; @warn "rv_ts failed" exception = err; end
try
    ppc = posterior_predictive_check(chains, params, data; n_draws = 300)
    plot_ppc(ppc; filename = joinpath(OUT_DIR, "ppc.png"))
    @printf("  ✓ ppc.png — RV χ²/dof (best) = %.3f\n",
             get(ppc.summary, "rv_red_chi2_total", NaN))
catch err; @warn "ppc failed" exception = err; end

K = vec(Array(chains[:K_k1]))
@printf("\nRecovered K_k1 = %.2f [%.2f, %.2f] m/s (median, 16/84)\n",
         median(K), quantile(K, 0.16), quantile(K, 0.84))
for nm in ["gp_act_period", "gp_act_lambda_e", "gp_act_lambda_p"]
    v = vec(Array(chains[Symbol(nm)]))
    @printf("  %s = %.3g [%.3g, %.3g]\n", nm,
             median(v), quantile(v, 0.16), quantile(v, 0.84))
end

@printf("\nAll artifacts in: %s\n", OUT_DIR)
