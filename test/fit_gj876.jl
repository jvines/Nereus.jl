#!/usr/bin/env julia
# Multi-planet validation: GJ 876 b+c with Nereus.jl
#
# GJ 876 (Marcy+ 1998, 2001; Rivera+ 2010; Correia+ 2010):
#   b: P ≈ 61.117 d, K ≈ 214 m/s, e ≈ 0.032
#   c: P ≈ 30.088 d, K ≈ 88 m/s,  e ≈ 0.256
# Data: 257 HARPS_PRE + 13 HARPS_POST observations (PRIVATE provenance),
# both differential RVs (mean ≈ 0).
#
# This is a 2-planet fixed-N fit using NUTS (HMC). We fit only b+c
# (the two dominant signals). Planets d and e are too weak for a
# quick validation and would need GP/activity modeling.

using Nereus
using DelimitedFiles
using Random: randperm
using Statistics: median, mean, std, quantile
using Printf

# ---- Load data -------------------------------------------------------
datafile = joinpath(@__DIR__, "data", "gj876.csv")
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

# Subsample for speed (270 obs through 2 Kepler solves per obs is
# manageable, but let's cap at 80/inst for a quick validation).
max_per_inst = 80
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

println("GJ 876: $(length(bjd)) observations (subsampled to $max_per_inst/inst)")
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
# 2 planets, RV-only, sesinw parametrisation.
ic = InstrumentConfig(rv=inst_names)

priors = Dict{String, PriorSpec}(
    "n_p"          => FixedPrior(2.0),
    # Planet b (outer, dominant)
    "P_k1"         => LogUniformPrior(30.0, 100.0),
    "K_k1"         => LogUniformPrior(50.0, 500.0),
    "sesinw_k1"    => UniformPrior(-1.0, 1.0),
    "secosw_k1"    => UniformPrior(-1.0, 1.0),
    "Mo_k1"        => UniformPrior(-2π, 2π),
    # Planet c (inner, 2:1 resonance with b)
    "P_k2"         => LogUniformPrior(15.0, 60.0),
    "K_k2"         => LogUniformPrior(20.0, 300.0),
    "sesinw_k2"    => UniformPrior(-1.0, 1.0),
    "secosw_k2"    => UniformPrior(-1.0, 1.0),
    "Mo_k2"        => UniformPrior(-2π, 2π),
    # Instrument systematics
    "gamma_$(inst_names[1])" => UniformPrior(-50.0, 50.0),
    "sigma_$(inst_names[1])" => LogUniformPrior(0.01, 20.0),
    "gamma_$(inst_names[2])" => UniformPrior(-50.0, 50.0),
    "sigma_$(inst_names[2])" => LogUniformPrior(0.01, 20.0),
)

params = Params(;
    priors = priors,
    max_kplanet = 2,
    planet_modes = [RV_ONLY, RV_ONLY],
    instruments = ic,
)

println("\nModel: $(n_unfrozen(params)) free parameters, $(n_frozen(params)) fixed")
println("Unfrozen: ", join(params.layout.unfrozen_names, ", "))

# ---- Build target and run NUTS ---------------------------------------
target = NereusTarget(params, data)

# Init near known solution.
# GJ 876 b: P=61.117, K=214, e=0.032, w~14deg
# GJ 876 c: P=30.088, K=88, e=0.256, w~48deg
e_b = 0.032; w_b = deg2rad(14.0)
e_c = 0.256; w_c = deg2rad(48.0)
x0 = [
    61.1,     # P_k1 (planet b)
    210.0,    # K_k1
    sqrt(e_b)*sin(w_b),  # sesinw_k1
    sqrt(e_b)*cos(w_b),  # secosw_k1
    0.0,      # Mo_k1
    30.1,     # P_k2 (planet c)
    85.0,     # K_k2
    sqrt(e_c)*sin(w_c),  # sesinw_k2
    sqrt(e_c)*cos(w_c),  # secosw_k2
    0.0,      # Mo_k2
    0.0,      # gamma_HARPS_POST
    3.0,      # sigma_HARPS_POST
    0.0,      # gamma_HARPS_PRE
    3.0,      # sigma_HARPS_PRE
]

println("\nStarting NUTS: 500 warmup + 1000 samples")
println("  AD=ForwardDiff, 2-planet model, packed transforms + packed priors")
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
println("  GJ 876 b+c — Nereus.jl NUTS posterior")
println("="^60)

fitted_stats = summarize_fitted(chains, params)
derived_stats = summarize_derived(chains, params)
print_results(fitted_stats, derived_stats)
print_ess_rhat(chains)

println("\n" * "="^60)
