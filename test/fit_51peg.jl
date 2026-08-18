#!/usr/bin/env julia
# First real-data fit: 51 Peg b with Nereus.jl
#
# 51 Peg b (Mayor & Queloz 1995): P ≈ 4.2308 d, K ≈ 55.9 m/s, e ≈ 0.
# Data: 780 HAMILTON + 330 ELODIE + 283 APF + 180 ESPRESSO + 118 HIRES_POST
# (1691 observations, absolute RVs).
#
# This is a 1-planet fixed-N fit using NUTS (HMC). We expect to
# recover the known parameters within uncertainties.

using Nereus
using DelimitedFiles
using Random: randperm
using Statistics: median, mean, std, quantile
using Printf

# ---- Load data -------------------------------------------------------
datafile = joinpath(@__DIR__, "data", "51peg.csv")
raw = readdlm(datafile, ',', Any, '\n'; header=true)
data_mat = raw[1]
header = raw[2]

bjd     = Float64.(data_mat[:, 1])
rv      = Float64.(data_mat[:, 2])
rv_err  = Float64.(data_mat[:, 3])
inst_str = String.(data_mat[:, 4])

# Map instrument strings to 1-based indices.
inst_names = sort!(unique(inst_str))
inst_map = Dict(name => i for (i, name) in enumerate(inst_names))
rv_inst = [inst_map[s] for s in inst_str]

# Subsample for speed (ForwardDiff through 600+ Kepler solves is slow;
# 100 per instrument is enough for a validation fit).
max_per_inst = 30
keep = Int[]
for (i, name) in enumerate(inst_names)
    idxs = findall(rv_inst .== i)
    if length(idxs) > max_per_inst
        idxs = sort(idxs[randperm(length(idxs))[1:max_per_inst]])
    end
    append!(keep, idxs)
end
sort!(keep)
bjd      = bjd[keep]
rv       = rv[keep]
rv_err   = rv_err[keep]
rv_inst  = rv_inst[keep]

println("51 Peg: $(length(bjd)) observations (subsampled to $max_per_inst/inst)")
for (i, name) in enumerate(inst_names)
    mask = rv_inst .== i
    n = sum(mask)
    println("  [$i] $name: n=$n, mean_rv=$(@sprintf("%.1f", mean(rv[mask]))) m/s, " *
            "rms=$(@sprintf("%.1f", std(rv[mask]))) m/s, " *
            "mean_err=$(@sprintf("%.2f", mean(rv_err[mask]))) m/s")
end

# ---- Build Data struct -----------------------------------------------
data = Data(; t_rv=bjd, rv=rv, rv_err=rv_err, rv_inst=rv_inst)
println("\nt_ref = $(@sprintf("%.2f", data.t_ref)) (BJD)")

# ---- Model config ----------------------------------------------------
# 1 planet, RV-only, sesinw parametrisation (default).
# Priors based on known 51 Peg b parameters ± generous ranges.
ic = InstrumentConfig(rv=inst_names)

priors = Dict{String, PriorSpec}(
    "n_p"          => FixedPrior(1.0),
    # Planet b
    "P_k1"         => LogUniformPrior(1.0, 20.0),
    "K_k1"         => LogUniformPrior(1.0, 200.0),
    "sesinw_k1"    => UniformPrior(-1.0, 1.0),
    "secosw_k1"    => UniformPrior(-1.0, 1.0),
    "Mo_k1"        => UniformPrior(-2π, 2π),
)
# Per-instrument systematics (mixed zero-points: some differential, some absolute)
for name in inst_names
    priors["gamma_$name"] = UniformPrior(-40000.0, 1000.0)
    priors["sigma_$name"] = LogUniformPrior(0.01, 50.0)
end

params = Params(;
    priors = priors,
    max_kplanet = 1,
    planet_modes = [RV_ONLY],
    instruments = ic,
    M_s = 1.11,     # Santos+ 2004
)

println("\nModel: $(n_unfrozen(params)) free parameters, $(n_frozen(params)) fixed")
println("Unfrozen: ", join(params.layout.unfrozen_names, ", "))

# ---- Build target and run NUTS ---------------------------------------
target = NereusTarget(params, data)

# For characterisation with NUTS, initialise near the known solution.
# NUTS is a local sampler — it explores one mode, not the full
# multimodal posterior. Blind period search needs parallel tempering
# (spec step 7). For characterisation, the user knows P from a
# periodogram or a prior detection.
# Init near known solution: 5 planet + 10 systemic (gamma, sigma per instrument)
# Instruments sorted alphabetically: APF, ELODIE, ESPRESSO, HAMILTON, HIRES_POST
# APF/HAMILTON/HIRES_POST are differential (~0), ELODIE/ESPRESSO are absolute (~-33 km/s)
x0 = [
    4.23,       # P_k1
    55.0,       # K_k1
    0.01,       # sesinw_k1
    0.01,       # secosw_k1
    0.0,        # Mo_k1
    0.0,        # gamma_APF (differential)
    3.0,        # sigma_APF
    -33250.0,   # gamma_ELODIE (absolute)
    8.0,        # sigma_ELODIE
    -33140.0,   # gamma_ESPRESSO (absolute)
    0.5,        # sigma_ESPRESSO
    0.0,        # gamma_HAMILTON (differential)
    5.0,        # sigma_HAMILTON
    -10.0,      # gamma_HIRES_POST (differential)
    1.5,        # sigma_HIRES_POST
]

println("\nStarting NUTS: 500 warmup + 1000 samples")
println("  AD=ForwardDiff, packed transforms + packed priors")
t0 = time()
chains = sample_nuts(target;
    n_samples = 1000,
    n_warmup  = 500,
    n_chains  = 1,
    init = x0,
    ad_backend = :ForwardDiff,
    target_accept = 0.85,
    progress = true,
)
dt = time() - t0
println(@sprintf("\nDone in %.1f seconds.", dt))

# ---- Results ---------------------------------------------------------
println("\n" * "="^60)
println("  51 Peg b — Nereus.jl NUTS posterior")
println("="^60)

fitted_stats = summarize_fitted(chains, params)
derived_stats = summarize_derived(chains, params)
print_results(fitted_stats, derived_stats)
print_ess_rhat(chains)

println("\n" * "="^60)
