#!/usr/bin/env julia
# Trans-dimensional fit of HD 18599b: planet search + noise model comparison.
#
# HD 18599b / TOI-179b (Vines+ 2023, MNRAS 518, 2627):
#   P = 4.1375 d, K = 11 ± 3 m/s, Rp/Rs = 0.0311, Mp = 25.5 ± 4.6 Me
#   Active K2 dwarf, P_rot = 8.74 d, log R'HK = -4.41
#   M_s = 0.807 M_sun, R_s = 0.798 R_sun (ARIADNE)
#
# Data: 137 RVs across HARPS_PRE, HARPS_POST (SERVAL + DRS), FEROS,
#       ESPRESSO, CORALIE_14. Activity indicators: BIS, FWHM.
#
# Noise models (all independently toggleable):
#   AR(1) — autoregressive correlated noise
#   MA(1) — moving average correlated noise
#   ActivityDecorrelation(BIS) — linear decorrelation with BIS
#
# This gives 2^3 = 8 noise configurations explored jointly with N_p.
# Replicates the manual 6-run model comparison from Vines+ 2023 Table 8
# in a single trans-dim run.

using Nereus
using DelimitedFiles
using Statistics: median, mean, std
using Printf
using MCMCChains

# ---- Load data -------------------------------------------------------
datafile = joinpath(@__DIR__, "data", "hd18599.csv")
raw = readdlm(datafile, ',', Any, '\n'; header=true)
data_mat = raw[1]

# Filter out HARPS_POST DRS (bad errors ~6600 m/s, ingestion bug)
inst_raw = String.(data_mat[:, 16])
prov_raw = String.(data_mat[:, 17])
keep = Int[]
inst_str = String[]
for i in 1:size(data_mat, 1)
    ins = strip(inst_raw[i])
    prov = strip(prov_raw[i])
    if ins == "HARPS_POST" && prov == "ESO_PHASE3"
        continue
    elseif ins == "HARPS_POST" && prov == "PRIVATE"
        push!(keep, i)
        push!(inst_str, "HARPS_POST")
    else
        push!(keep, i)
        push!(inst_str, ins)
    end
end
data_mat = data_mat[keep, :]
bjd      = Float64.(data_mat[:, 1])
rv       = Float64.(data_mat[:, 2])
rv_err   = Float64.(data_mat[:, 3])

inst_names = sort!(unique(inst_str))
inst_map = Dict(name => i for (i, name) in enumerate(inst_names))
rv_inst = [inst_map[s] for s in inst_str]

println("HD 18599 trans-dim: $(length(bjd)) observations")
for (i, name) in enumerate(inst_names)
    mask = rv_inst .== i
    n = sum(mask)
    println("  [$i] $name: n=$n, " *
            "rms=$(@sprintf("%.1f", std(rv[mask]))) m/s, " *
            "mean_err=$(@sprintf("%.2f", mean(rv_err[mask]))) m/s")
end

# ---- Load and standardize activity indicators ------------------------
function load_indicator(col_idx)
    vals = Float64[]
    for i in 1:size(data_mat, 1)
        v = data_mat[i, col_idx]
        if v === "" || v === nothing || (v isa AbstractString && strip(v) == "")
            push!(vals, NaN)
        else
            push!(vals, Float64(v))
        end
    end
    return vals
end

# Mean-subtract and normalize to RMS (matching Vines+ 2023 Table 7 footnote)
function standardize!(v)
    finite = filter(isfinite, v)
    isempty(finite) && return v
    μ = mean(finite)
    σ = std(finite)
    σ == 0 && return v
    for i in eachindex(v)
        isfinite(v[i]) && (v[i] = (v[i] - μ) / σ)
    end
    return v
end

ind_bis = load_indicator(4)

# EMPEROR-style normalization: mean-subtract, divide by max(abs),
# then scale by max(abs(RV)). This makes C coefficients dimensionless
# and naturally O(1), bounded U(-1, 1).
function emperor_normalize!(v, rv)
    finite = filter(isfinite, v)
    isempty(finite) && return v
    μ = mean(finite)
    for i in eachindex(v)
        isfinite(v[i]) && (v[i] -= μ)
    end
    mx = maximum(abs, filter(isfinite, v))
    mx == 0 && return v
    rv_scale = maximum(abs, rv)
    for i in eachindex(v)
        isfinite(v[i]) && (v[i] = v[i] / mx * rv_scale)
    end
    return v
end

emperor_normalize!(ind_bis, rv)

indicators = Dict{String, Vector{Float64}}(
    "bisector_span" => ind_bis,
)

# ---- Build Data + Params ---------------------------------------------
data = Data(; t_rv=bjd, rv=rv, rv_err=rv_err, rv_inst=rv_inst,
              indicators=indicators)
ic = InstrumentConfig(rv=inst_names)

# Noise models: AR(1), MA(1), ActivityDecorrelation(BIS) — all toggleable
ar = ARModel(order=1)
ma = MAModel(order=1)
act = ActivityDecorrelation(indicators=["bisector_span"])

# Priors matching Vines+ 2023 Table 7
rv_max = maximum(abs, rv)
priors = Dict{String, PriorSpec}(
    # Orbital — generous, let trans-dim find it
    # Planet b: period constrained by TESS transit (P = 4.1375 ± 0.0004 d)
    "P_k1"         => UniformPrior(4.13, 4.15),
    "K_k1"         => UniformPrior(0.0, 50.0),
    "sesinw_k1"    => UniformPrior(-1.0, 1.0),
    "secosw_k1"    => UniformPrior(-1.0, 1.0),
    "Mo_k1"        => UniformPrior(-2π, 2π),
    # Slot for a potential second planet
    "P_k2"         => LogUniformPrior(1.0, 6.0),
    "K_k2"         => UniformPrior(0.0, 50.0),
    "sesinw_k2"    => UniformPrior(-1.0, 1.0),
    "secosw_k2"    => UniformPrior(-1.0, 1.0),
    "Mo_k2"        => UniformPrior(-2π, 2π),
)

# Per-instrument systematics (matching Table 7)
for name in inst_names
    priors["gamma_$name"] = UniformPrior(0.0, 3 * rv_max)
    priors["sigma_$name"] = LogUniformPrior(0.1, 20.0)
end

params = Params(;
    max_kplanet = 2,
    planet_modes = [RV_ONLY, RV_ONLY],
    instruments = ic,
    data = data,
    M_s = 0.807,
    priors = priors,
    noise_models = [ar, ma, act],
    transdim_noise = true,
    trend_order = 1,    # linear acceleration (Table 7: γ̇)
    external_priors = [ExternalPrior(:ecc, NormalPrior(0.0, 0.3), true)],
)

target = NereusTarget(params, data; unconstrained=false)

println("\nModel: max_kplanet=2, $(n_unfrozen(params)) total params")
println("Unfrozen: ", join(params.layout.unfrozen_names, ", "))

# ---- Trans-dim config ------------------------------------------------
# AR(1), MA(1), ActivityDecorrelation(BIS) all independently toggleable
td = TransDimConfig(
    max_kplanet = 2,
    birth_strategies = [PriorBirth(), InformedBirth()],
    birth_weights = [0.3, 0.7],
    transdim_fraction = 0.3,
    noise = true,
    toggleable = [ar, ma, act],
)

# ---- Run PT trans-dim ------------------------------------------------
println("\nStarting PT trans-dim: 15 rounds, 20 chains")
t0 = time()
chains, log_ev, pt_evals = sample_pt(target;
    td = td,
    n_rounds = 15,
    n_chains = 20,
    seed = 777,
    show_report = true,
)
dt = time() - t0
@printf("PT done in %.1f seconds (%d likelihood evals)\n", dt, pt_evals)

# ---- Save chains first -----------------------------------------------
save_chains(joinpath("/private/tmp", "hd18599_chains.nc"), chains, params; data=data)
println("Chains saved to /private/tmp/hd18599_chains.nc")

# ---- Results ---------------------------------------------------------
println("\n" * "="^60)
println("  HD 18599 — Trans-dimensional PT results")
println("="^60)

print_transdim_summary(chains, params; td=td, M_s=0.807)

# ---- Save results ----------------------------------------------------
outdir = joinpath(@__DIR__, "output")
mkpath(outdir)
save_transdim_summary(chains, params, outdir;
                       starname="HD18599", td=td, log_evidence=log_ev, M_s=0.807)

# ---- Plots -----------------------------------------------------------
plotdir = "/private/tmp/nereus_hd18599"

println("\nGenerating plots...")
plot_rv_timeseries(chains, params, data; output=plotdir)
for k in 1:params.config.max_kplanet
    plot_rv_phasefold(chains, params, data; planet=k, output=plotdir)
end
plot_histograms(chains, params; output=plotdir)

println("Plots saved to $plotdir/")
println("\n" * "="^60)
