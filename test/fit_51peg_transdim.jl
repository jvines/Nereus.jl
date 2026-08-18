#!/usr/bin/env julia
# Trans-dimensional fit of 51 Peg: does RJMCMC find N_p = 1?
#
# 51 Peg b (Mayor & Queloz 1995): P ≈ 4.2308 d, K ≈ 55.9 m/s, e ≈ 0.
# Data: subsampled from 1691 observations across 5 instruments.
#
# Runs RJMCMC with max_kplanet=2, informed birth, 50k samples.
# Expected: posterior on N_p strongly favors 1.

using Nereus
using DelimitedFiles
using Random: randperm
using Statistics: median, mean, std, quantile
using Printf
using MCMCChains

# ---- Load data -------------------------------------------------------
datafile = joinpath(@__DIR__, "data", "51peg.csv")
raw = readdlm(datafile, ',', Any, '\n'; header=true)
data_mat = raw[1]

bjd      = Float64.(data_mat[:, 1])
rv       = Float64.(data_mat[:, 2])
rv_err   = Float64.(data_mat[:, 3])
inst_str = String.(data_mat[:, 4])

inst_names = sort!(unique(inst_str))
inst_map = Dict(name => i for (i, name) in enumerate(inst_names))
rv_inst = [inst_map[s] for s in inst_str]

# Subsample for speed
max_per_inst = 40
keep = Int[]
for (i, name) in enumerate(inst_names)
    idxs = findall(rv_inst .== i)
    if length(idxs) > max_per_inst
        idxs = sort(idxs[randperm(length(idxs))[1:max_per_inst]])
    end
    append!(keep, idxs)
end
sort!(keep)
bjd     = bjd[keep]
rv      = rv[keep]
rv_err  = rv_err[keep]
rv_inst = rv_inst[keep]

println("51 Peg trans-dim: $(length(bjd)) observations")
for (i, name) in enumerate(inst_names)
    mask = rv_inst .== i
    println("  [$i] $name: n=$(sum(mask)), rms=$(@sprintf("%.1f", std(rv[mask]))) m/s")
end

# ---- Build model -----------------------------------------------------
data = Data(; t_rv=bjd, rv=rv, rv_err=rv_err, rv_inst=rv_inst)
ic = InstrumentConfig(rv=inst_names)

params = Params(;
    max_kplanet = 2,
    planet_modes = [RV_ONLY, RV_ONLY],
    instruments = ic,
    data = data,
    M_s = 1.11,     # Santos+ 2004
)

target = NereusTarget(params, data; unconstrained=false)

println("\nModel: max_kplanet=2, $(n_unfrozen(params)) total params")
println("Unfrozen: ", join(params.layout.unfrozen_names, ", "))

# ---- Trans-dim config ------------------------------------------------
td = TransDimConfig(
    max_kplanet = 2,
    birth_strategies = [PriorBirth(), InformedBirth()],
    birth_weights = [0.3, 0.7],
    transdim_fraction = 0.3,
)

# ---- Run RJMCMC ------------------------------------------------------
n_samples = 50_000
n_warmup = 10_000

println("\nStarting RJMCMC: $(n_warmup) warmup + $(n_samples) samples")
t0 = time()
chains, rjmcmc_evals = sample_rjmcmc(target, data;
    td = td,
    n_samples = n_samples,
    n_warmup = n_warmup,
    seed = 42,
    initial_scale = 0.005,
)
dt = time() - t0
@printf("Done in %.1f seconds (%.0f samples/sec, %d likelihood evals)\n", dt, n_samples / dt, rjmcmc_evals)

# ---- RJMCMC results --------------------------------------------------
println("\n" * "="^60)
println("  51 Peg — Trans-dimensional RJMCMC results")
println("="^60)

print_transdim_summary(chains, params; td=td)

# ---- Also run PT trans-dim for comparison ----------------------------
println("\n\nStarting PT trans-dim: 14 rounds, 15 chains")
t0_pt = time()
chains_pt, log_ev, pt_evals = sample_pt(target;
    td = td,
    n_rounds = 14,
    n_chains = 15,
    seed = 123,
    show_report = true,
)
dt_pt = time() - t0_pt
@printf("PT done in %.1f seconds (%d likelihood evals)\n", dt_pt, pt_evals)

println("\n" * "="^60)
println("  51 Peg — Trans-dimensional PT results")
println("="^60)

print_transdim_summary(chains_pt, params; td=td)
outdir = joinpath(@__DIR__, "output")
mkpath(outdir)
save_transdim_summary(chains_pt, params, outdir;
                       starname="51Peg", td=td, log_evidence=log_ev)

println("\n" * "="^60)
