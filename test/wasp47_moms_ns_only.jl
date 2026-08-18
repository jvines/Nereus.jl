#!/usr/bin/env julia
# WASP-47 — MoMS-NS only, n_live = 1000.
#
# Loads the same multi-instrument target as compare_WASP47_joint_search.jl,
# but skips RJMCMC/PT/MoMS so we can iterate fast on MoMS-NS settings.
# Run after the v4d shootout to see if 1000 live points (vs 400) lets
# the chain find c (P ≈ 588 d, K ≈ 32 m/s) — the GLS of b-subtracted
# residuals shows c at clear FAP < 10⁻³ but no sampler at 400-live
# managed to land in that basin.

using Nereus
using DelimitedFiles: readdlm
using Statistics: median, std, mean, quantile
using Printf
using MCMCChains
using Random

const REPO_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const DATADIR   = joinpath(REPO_ROOT, "data", "WASP47")
const OUT_DIR   = joinpath(REPO_ROOT, "Nereus.jl", "results",
                            "WASP47_joint_search")
mkpath(OUT_DIR)

# Literature (Bryant & Bayliss 2022) — only used for the post-hoc
# comparison print, NOT plugged into priors.
const Pb_LIT, Kb_LIT = 4.1591287, 140.84
const Pc_LIT, Kc_LIT = 588.4,      31.6
const Pd_LIT, Kd_LIT = 9.030672,    4.96
const Pe_LIT, Ke_LIT = 0.789593,    4.55

println("="^70)
println("WASP-47 — MoMS-NS only (n_live = 1000)")
println("="^70)

# Photometry: K2 EVEREST (C03) as primary, TESS sectors as secondary.
# K2 has 20× better per-point precision (60 ppm vs 1280 ppm/bin), so
# the small planets d (440 ppm) and e (240 ppm) are recoverable from
# K2 photometry alone. TESS sectors add baseline coverage for ephemeris
# refinement on b. Each survey gets its own instrument index (1=K2,
# 2=TESS) so Nereus fits separate limb-darkening, jitter, and offset
# parameters per bandpass.
function _bin5min(t::Vector{Float64}, f::Vector{Float64}, e::Vector{Float64})
    bd = 5.0 / (60 * 24)
    bin_id = floor.(Int, (t .- minimum(t)) ./ bd)
    tb = Float64[]; fb = Float64[]; eb = Float64[]
    for b in sort(unique(bin_id))
        idxs = findall(==(b), bin_id)
        push!(tb, mean(t[idxs]))
        push!(fb, mean(f[idxs]))
        push!(eb, mean(e[idxs]) / sqrt(length(idxs)))
    end
    return tb, fb, eb
end

# K2 (single campaign C03 — only K2 dataset for WASP-47)
k2_path = joinpath(DATADIR, "WASP-47_k2_c03_everest_lc.csv")
k2_raw = load_tess_lc(k2_path)   # same 3-column CSV format
k2_t, k2_flux, k2_err = _bin5min(k2_raw.t, k2_raw.flux, k2_raw.flux_err)
@printf("K2 C03 EVEREST binned: %d pts, σ ≈ %.2e\n", length(k2_t), std(k2_flux))

# TESS multi-sector
sector_files = sort(filter(f -> occursin(r"WASP-47_tess_s\d+_lc\.csv$", f),
                            readdir(DATADIR; join = true)))
tess_t = Float64[]; tess_flux = Float64[]; tess_err = Float64[]
for fp in sector_files
    lc_raw = load_tess_lc(fp)
    tb, fb, eb = _bin5min(lc_raw.t, lc_raw.flux, lc_raw.flux_err)
    append!(tess_t, tb); append!(tess_flux, fb); append!(tess_err, eb)
    @printf("  %s: %d raw → %d binned\n",
            basename(fp), length(lc_raw.t), length(tb))
end
@printf("TESS combined binned: %d pts across %d sectors, σ ≈ %.2e\n",
        length(tess_t), length(sector_files), std(tess_flux))

# Concatenate: K2 first (instrument 1), TESS second (instrument 2)
phot_t   = vcat(k2_t,   tess_t)
phot_f   = vcat(k2_flux, tess_flux)
phot_e   = vcat(k2_err,  tess_err)
phot_inst = vcat(fill(1, length(k2_t)), fill(2, length(tess_t)))
perm = sortperm(phot_t)
phot_t = phot_t[perm]; phot_f = phot_f[perm]; phot_e = phot_e[perm]
phot_inst = phot_inst[perm]
flat = (t = phot_t, flux = phot_f, flux_err = phot_e)
@printf("Combined photometry: %d pts (K2: %d, TESS: %d)\n",
        length(phot_t), count(==(1), phot_inst), count(==(2), phot_inst))

# Load combined RVs
rv_raw_full = readdlm(joinpath(DATADIR, "WASP47_RVs_combined.csv"),
                       ',', Any, '\n'; header = true)
rv_raw, hdr = rv_raw_full[1], vec(rv_raw_full[2])
col(name) = findfirst(==(name), hdr)

flag_col = col("flag")
keep = trues(size(rv_raw, 1))
for i in 1:size(rv_raw, 1)
    f = rv_raw[i, flag_col]
    f === missing && continue
    s = String(strip(string(f)))
    if s in ("transit", "transit_night", "anomalous")
        keep[i] = false
    end
end
rv_raw = rv_raw[keep, :]

bjd_rv = Float64.(rv_raw[:, col("bjd")])
rv     = Float64.(rv_raw[:, col("rv")])
rv_err = Float64.(rv_raw[:, col("rv_err")])
rv_inst_str = String.(rv_raw[:, col("instrument")])
inst_names = sort(unique(rv_inst_str))
inst_to_idx = Dict(name => i for (i, name) in enumerate(inst_names))
rv_inst = [inst_to_idx[s] for s in rv_inst_str]
@printf("Loaded %d RVs across %d instruments\n",
        length(bjd_rv), length(inst_names))

data = Data(;
    t_rv = bjd_rv, rv = rv, rv_err = rv_err, rv_inst = rv_inst,
    t_phot = flat.t, flux = flat.flux, flux_err = flat.flux_err,
    phot_inst = phot_inst,
)
ic = InstrumentConfig(rv = inst_names, pm = ["K2", "TESS"])

priors = Dict{String, PriorSpec}()
for k in 1:3
    priors["P_k$k"]      = LogUniformPrior(0.5, 30.0)
    priors["K_k$k"]      = UniformPrior(0.0, 200.0)
    priors["sesinw_k$k"] = UniformPrior(-0.7, 0.7)
    priors["secosw_k$k"] = UniformPrior(-0.7, 0.7)
    priors["Tc_k$k"]     = UniformPrior(flat.t[1], flat.t[1] + 30.0)
    priors["b_k$k"]      = UniformPrior(0.0, 1.0)
    priors["rr_k$k"]     = UniformPrior(0.005, 0.20)
end
priors["P_k4"]      = LogUniformPrior(100.0, 1500.0)
priors["K_k4"]      = UniformPrior(0.0, 200.0)
priors["sesinw_k4"] = UniformPrior(-0.7, 0.7)
priors["secosw_k4"] = UniformPrior(-0.7, 0.7)

for name in inst_names
    rv_i = rv[rv_inst .== inst_to_idx[name]]
    isempty(rv_i) && continue
    centre = median(rv_i)
    span   = max(maximum(rv_i) - minimum(rv_i), 1.0)
    priors["gamma_$name"] = UniformPrior(centre - 3 * span, centre + 3 * span)
    priors["sigma_$name"] = ModJeffreysPrior(0.1, 50.0)
end
priors["offset_K2"]   = NormalPrior(0.0, 1e-3, -5e-3, 5e-3)
priors["jitter_K2"]   = LogUniformPrior(1e-5, 5e-3)
priors["q1_K2"]       = UniformPrior(0.0, 1.0)
priors["q2_K2"]       = UniformPrior(0.0, 1.0)
priors["offset_TESS"] = NormalPrior(0.0, 1e-3, -5e-3, 5e-3)
priors["jitter_TESS"] = LogUniformPrior(1e-5, 5e-3)
priors["q1_TESS"]     = UniformPrior(0.0, 1.0)
priors["q2_TESS"]     = UniformPrior(0.0, 1.0)
priors["rho_s"]         = NormalPrior(1.00, 0.30, 0.1, 5.0)

parametrization = ParametrizationConfig(time = :Tc, geom = :b_rr,
                                          use_rho_s = true)
params = Params(;
    max_kplanet     = 4,
    planet_modes    = [RVPM, RVPM, RVPM, RV_ONLY],
    instruments     = ic,
    data            = data,
    M_s             = 1.04,
    R_s             = 1.137,
    parametrization = parametrization,
    priors          = priors,
    trend_order     = 0,
    stability       = :amd,
    external_priors = [ExternalPrior(:ecc, NormalPrior(0.0, 0.3), true)],
)
@printf("Free params: %d\n", n_unfrozen(params))

target = NereusTarget(params, data; unconstrained = true)

td = TransDimConfig(
    max_kplanet       = 4,
    planets           = true,
    noise             = false,
    birth_strategies  = [PriorBirth(), JointInformedBirth()],
    birth_weights     = [0.3, 0.7],
    transdim_fraction = 0.3,
)

const SEED = parse(Int, get(ENV, "WASP47_NS_SEED",  "42"))
const N_LIVE = parse(Int, get(ENV, "WASP47_NS_LIVE",  "1000"))
const USE_PATHFINDER = parse(Bool, get(ENV, "WASP47_NS_PATHFINDER", "true"))

# Pathfinder warmup: bypass the early-locking trap that hits MoMS-NS
# on multi-planet datasets. With ALL slots forced active in the
# warm-start, NS shrinks onto the high-density region of the joint
# 4-planet posterior rather than discovering it from random γ init.
warm_start = if USE_PATHFINDER
    println("\n" * "="^70)
    println("Pathfinder warmup: $(N_LIVE) live points across 16 L-BFGS basins")
    println("="^70)
    t_pf = time()
    ws = pathfinder_warmstart_moms_ns(target, target.data;
                                        td = td,
                                        n_live = N_LIVE,
                                        n_pathfinder_runs = 16,
                                        seed = SEED + 100,
                                        quiet = false)
    @printf("Pathfinder warmup done in %.1f min\n", (time() - t_pf) / 60)
    ws
else
    nothing
end

println("\n" * "="^70)
println("MoMS-NS production: n_live = $(N_LIVE), batch_size = 16, dlogz = 0.5")
println("="^70)

t0 = time()
chains, log_Z, _ = sample_moms_ns(target, target.data;
    td             = td,
    n_live         = N_LIVE,
    dlogz          = parse(Float64, get(ENV, "WASP47_NS_DLOGZ", "0.5")),
    n_mcmc         = parse(Int, get(ENV, "WASP47_NS_MCMC",  "30")),
    seed           = SEED,
    informed_birth_fraction = 0.5,
    batch_size     = parse(Int, get(ENV, "WASP47_NS_BATCH", "16")),
    warm_start_points = warm_start,
)
dt = time() - t0
@printf("\nMoMS-NS done in %.1f min, log Z = %+.3f\n", dt / 60, log_Z)

probs = model_probabilities(chains; max_kplanet = 4)
@printf("\nN_p posterior:\n")
for k in 0:4
    @printf("  P(N_p=%d) = %.4f\n", k, get(probs, k, 0.0))
end

# Sort planets by period at modal N_p, report (P, K).
modal_np = argmax([get(probs, k, 0.0) for k in 0:4]) - 1
np_vec   = vec(Array(chains[:, :n_planets, :]))
mask     = np_vec .== modal_np
n_kept   = count(mask)
@printf("\nModal N_p = %d  (n_samples = %d)\n", modal_np, n_kept)
if modal_np >= 1 && n_kept > 0
    P_cols = [vec(Array(chains[:, Symbol("P_k$k"), :]))[mask] for k in 1:4]
    K_cols = [vec(Array(chains[:, Symbol("K_k$k"), :]))[mask] for k in 1:4]
    P_sorted = [Float64[] for _ in 1:modal_np]
    K_sorted = [Float64[] for _ in 1:modal_np]
    for s in 1:n_kept
        Ps = [P_cols[k][s] for k in 1:4]
        Ks = [K_cols[k][s] for k in 1:4]
        ord = sortperm(Ps)
        for j in 1:modal_np
            push!(P_sorted[j], Ps[ord[j]])
            push!(K_sorted[j], Ks[ord[j]])
        end
    end
    println("\nRecovered planets (sorted by period):")
    println("  Literature: e=0.79/4.6, b=4.16/141, d=9.03/5, c=588/32")
    for j in 1:modal_np
        @printf("    rank %d: P = %9.4f ± %7.4f d   K = %6.2f ± %5.2f m/s\n",
                j, median(P_sorted[j]), std(P_sorted[j]),
                median(K_sorted[j]), std(K_sorted[j]))
    end
end

save_chains(joinpath(OUT_DIR, "moms_ns_n_live_1000_chains.nc"),
             chains, target.params; data = target.data,
             log_evidence = log_Z)
@printf("\nChains saved to %s/moms_ns_n_live_1000_chains.nc\n", OUT_DIR)
