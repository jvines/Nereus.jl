#!/usr/bin/env julia
# WASP-47 — sample_pt only.
#
# Same K2+TESS+RV multi-instrument setup as wasp47_moms_ns_only.jl, but
# uses sample_pt_warm (Pathfinder-warmed parallel-tempered RJMCMC)
# instead of MoMS-NS. PT does iterative residual cleaning naturally
# via within-model MCMC at each temperature, and mixes between trans-
# dim modes via temperature ladder swaps — both avoid the L_thr-ratchet
# locking that hit MoMS-NS on this dataset.

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

# Literature values (Bryant & Bayliss 2022; Sinukoff 2017 for d, e). Used
# only for the post-hoc comparison print, NOT plugged into priors.
const Pb_LIT, Kb_LIT = 4.1591287, 140.84
const Pc_LIT, Kc_LIT = 588.4,      31.6
const Pd_LIT, Kd_LIT = 9.030672,    4.96
const Pe_LIT, Ke_LIT = 0.789593,    4.55

println("="^70)
println("WASP-47 — sample_pt (PT/RJMCMC, Pathfinder warmup)")
println("="^70)

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

# TESS sector 42 only — UNBINNED. v16 fast-track: dropping s92 cuts
# phot data 152k → 58k for ~2.5× speedup. d (9.03d) gets ~3 transits in
# a 27d sector, e (0.79d) gets ~34 — still detectable in principle, just
# lower SNR. Re-add s92 in a follow-up if needed.
sector_files = sort(filter(f -> occursin(r"WASP-47_tess_s42_lc\.csv$", f),
                            readdir(DATADIR; join = true)))
tess_t = Float64[]; tess_flux = Float64[]; tess_err = Float64[]
tess_inst = Int[]
for (i, fp) in enumerate(sector_files)
    lc_raw = load_tess_lc(fp)
    append!(tess_t,    lc_raw.t)
    append!(tess_flux, lc_raw.flux)
    append!(tess_err,  lc_raw.flux_err)
    append!(tess_inst, fill(i, length(lc_raw.t)))   # one channel per sector
    @printf("  %s: %d raw points (unbinned)\n",
            basename(fp), length(lc_raw.t))
end
@printf("TESS combined: %d points across %d sectors, σ_med = %.2e\n",
        length(tess_t), length(sector_files), median(tess_err))

# Sort by time
phot_t   = tess_t
phot_f   = tess_flux
phot_e   = tess_err
phot_inst = tess_inst
perm = sortperm(phot_t)
phot_t = phot_t[perm]; phot_f = phot_f[perm]; phot_e = phot_e[perm]
phot_inst = phot_inst[perm]
flat = (t = phot_t, flux = phot_f, flux_err = phot_e)
@printf("Photometry: %d unbinned TESS points (%d per-sector channels)\n",
        length(phot_t), length(unique(phot_inst)))

# Combined RVs (filtered)
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
# Per-sector channel names (e.g., "TESS_s42", "TESS_s92") so each sector
# gets independent offset/jitter/LD — sectors observed years apart and
# may have slightly different systematics.
function _sector_name(fp)
    m = match(r"_s(\d+)_", basename(fp))
    return m === nothing ? "TESS" : "TESS_s$(m.captures[1])"
end
pm_names = String[_sector_name(fp) for fp in sector_files]
ic = InstrumentConfig(rv = inst_names, pm = pm_names)

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
for name in pm_names
    priors["offset_$name"] = NormalPrior(0.0, 1e-3, -5e-3, 5e-3)
    priors["jitter_$name"] = LogUniformPrior(1e-5, 5e-3)
    priors["q1_$name"]     = UniformPrior(0.0, 1.0)
    priors["q2_$name"]     = UniformPrior(0.0, 1.0)
end
priors["rho_s"]       = NormalPrior(1.00, 0.30, 0.1, 5.0)

# CeleriteSHO GP on photometry — single damped oscillator. ~2× cheaper
# Cholesky than CeleriteRotation (2 effective terms vs 4) which makes
# 152k-point unbinned TESS tractable for an overnight run. Captures
# the dominant rotation/activity envelope (we're not trying to
# characterize rotation, just absorb it so the BLS sees transits).
# Init: ω0 = 2π/25d = 0.251 rad/d for WASP-47's ~25d rotation.
priors["gp_log_S0_phot"]     = UniformPrior(-25.0, -5.0)   # broad PSD
priors["gp_log_Q_phot"]      = UniformPrior(-1.0, 4.0)     # Q ∈ [0.4, 55]
priors["gp_log_omega0_phot"] = UniformPrior(-2.5, 0.5)     # P ∈ [2d, 50d]

parametrization = ParametrizationConfig(time = :Tc, geom = :b_rr,
                                          use_rho_s = true)
phot_gp = CeleriteSHO(channel = :phot, instruments = String[])
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
    noise_models    = [phot_gp],
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

const SEED      = parse(Int, get(ENV, "WASP47_PT_SEED",   "42"))
const PT_ROUNDS = parse(Int, get(ENV, "WASP47_PT_ROUNDS", "16"))
const PT_CHAINS = parse(Int, get(ENV, "WASP47_PT_CHAINS", "16"))

# Diagnostic-friendly bail-out: when WASP47_PT_DIAG_ONLY=1, exit after
# target/td construction (just above) so a downstream script can `include`
# this file to rebuild the same `target` + `data` + `params` + `td`
# context without re-running the sampler.
if get(ENV, "WASP47_PT_DIAG_ONLY", "0") == "1"
    @info "WASP47_PT_DIAG_ONLY=1 — target built, skipping sampling."
else

println("\n" * "="^70)
@printf("sample_pt_warm: %d rounds × %d chains  (seed=%d)\n",
        PT_ROUNDS, PT_CHAINS, SEED)
println("="^70)

t0 = time()
# sample_pt (NO Pathfinder warmup) — Pathfinder uses ForwardDiff Duals
# through the celerite GP forward, which causes massive Dual-array
# allocations × 152k phot points × 16 paths × hundreds of L-BFGS iters,
# triggering GC thrashing that stalls the run for hours. Trans-dim PT
# itself uses RWM/slice (no gradients), so skipping warmup is fine —
# just costs a few extra rounds for the chain to find typical set.
chains, _, n_evals, _ = sample_pt(target;
    td                   = td,
    n_rounds             = PT_ROUNDS,
    n_chains             = PT_CHAINS,
    seed                 = SEED,
    show_report          = true,
    within_model         = :rwm,
    early_stop_thresh    = 0.001,
    early_stop_min_rounds = 12,
)
dt = time() - t0
@printf("\nsample_pt done in %.1f min, %d log-density evaluations\n",
        dt / 60, n_evals)

# Modal-N_p posterior + planet table
np_vec = vec(Array(chains[:, :n_planets, :]))
n_total = length(np_vec)
@printf("\nN_p posterior:\n")
for k in 0:4
    p = count(==(k), np_vec) / n_total
    @printf("  P(N_p=%d) = %.4f\n", k, p)
end
modal_np = argmax([count(==(k), np_vec) / n_total for k in 0:4]) - 1
mask = np_vec .== modal_np
n_kept = count(mask)
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

save_chains(joinpath(OUT_DIR, "pt_warm_chains.nc"),
             chains, target.params; data = target.data)
@printf("\nChains saved to %s/pt_warm_chains.nc\n", OUT_DIR)

end  # if WASP47_PT_DIAG_ONLY != "1"
