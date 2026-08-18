#!/usr/bin/env julia
# 51 Peg: joint trans-dim on planets + noise model (MA on/off)

using Nereus
using DelimitedFiles
using Statistics: median, mean, std
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

println("51 Peg noise+planet trans-dim: $(length(bjd)) obs (full dataset)")
for (i, name) in enumerate(inst_names)
    mask = rv_inst .== i
    println("  [$i] $name: n=$(sum(mask))")
end

data = Data(; t_rv=bjd, rv=rv, rv_err=rv_err, rv_inst=rv_inst)
ic = InstrumentConfig(rv=inst_names)

# ---- Model: 2 planets + MA noise (all toggleable) -------------------
ma = MAModel(order=1)
params = Params(
    max_kplanet = 2,
    planet_modes = [RV_ONLY, RV_ONLY],
    instruments = ic,
    data = data,
    noise_models = [ma],
    transdim_noise = false,  # only MA, no mutual exclusion issue
)

target = NereusTarget(params, data; unconstrained=false)

println("Model: max_kplanet=2, MA(1) toggleable, $(n_unfrozen(params)) params")
println("Unfrozen: ", join(params.layout.unfrozen_names, ", "))

# ---- Trans-dim config: planets + noise --------------------------------
td = TransDimConfig(
    max_kplanet = 2,
    noise = true,
    toggleable = [ma],
    birth_strategies = [PriorBirth(), InformedBirth()],
    birth_weights = [0.3, 0.7],
    transdim_fraction = 0.3,
)

# ---- Run PT -----------------------------------------------------------
println("\nStarting PT: 16 rounds, 20 chains")
t0 = time()
chains, log_ev, n_evals = sample_pt(target;
    td = td,
    n_rounds = 16,
    n_chains = 20,
    seed = 42,
    show_report = true,
)
dt = time() - t0
@printf("Done in %.1f seconds (%d evals)\n\n", dt, n_evals)

# ---- Results ----------------------------------------------------------
print_transdim_summary(chains, params; td=td)
outdir = joinpath(@__DIR__, "output")
mkpath(outdir)
save_transdim_summary(chains, params, outdir;
                       starname="51Peg_noise", td=td, log_evidence=log_ev)
