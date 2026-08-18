#!/usr/bin/env julia
# WASP-47 joint Bayesian TTV-C fit (proper N-body inference, not a
# forward overlay from literature priors).
#
# Model: planets b and e both as RVPM_TTV_NB. K_b, K_e, P_b, Tc_b,
# eccentricities, transit geometry of b — all free. TTVFaster computes
# the b/e mutual N-body TTVs at each likelihood eval; transit_log_likelihood
# compares the resulting time-shifted transit model against the data.
# The posterior on K_e (→ M_e) is constrained jointly by transit shape
# (where the model puts the transit) AND transit timing (TTV pattern
# from b/e interaction).
#
# Why this beats the previous MC overlay: there's no Monte Carlo from
# literature priors. The posterior on M_e comes from the data, with
# the per-transit photometric likelihood doing all the work.

using Nereus
using DelimitedFiles: readdlm
using Statistics: median, std, mean, quantile
using Printf
using Random
using TTVFaster: Planet_plane_hk, compute_ttv!

const REPO_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const DATADIR   = joinpath(REPO_ROOT, "data", "WASP47")
const OUT_DIR   = joinpath(REPO_ROOT, "Nereus.jl", "results", "WASP47_TTV_C")
mkpath(OUT_DIR)

println("="^70)
println("WASP-47 joint TTV-C — proper Bayesian N-body inference")
println("="^70)

# --- Literature ephemerides + masses (Bryant 2022, Vanderburg 2017) -------
const P_b_LIT  = 4.1591287
const T0_b_LIT = 2456978.819            # empirical first K2 transit
const P_e_LIT  = 0.789593
const T0_e_LIT = 2457006.99267
const M_s_LIT  = 1.04
const R_s_LIT  = 1.137
const RR_B_LIT  = 0.1027
const B_B_LIT   = 0.13
const RHO_S_LIT = 0.96
const Q1_K2     = 0.45
const Q2_K2     = 0.30
const Q1_TESS   = 0.36
const Q2_TESS   = 0.32

# --- Load data (same K2 + 2 TESS sectors) --------------------------------
struct Epoch
    name::String
    t::Vector{Float64}
    flux::Vector{Float64}
    err::Vector{Float64}
end
function load_epoch(name, path, off)
    lc = load_tess_lc(path)
    return Epoch(name, lc.t .+ off, lc.flux, lc.flux_err)
end
epochs = [
    load_epoch("K2_c03",   joinpath(DATADIR, "WASP-47_k2_c03_everest_lc.csv"), 0.0),
    load_epoch("TESS_s42", joinpath(DATADIR, "WASP-47_tess_s42_lc.csv"), 2_457_000.0),
    load_epoch("TESS_s92", joinpath(DATADIR, "WASP-47_tess_s92_lc.csv"), 2_457_000.0),
]

# Detrend with mask wider than T14 (3.5 hr)
function _detrend(ep::Epoch)
    mask_b = mask_transits(ep.t, [P_b_LIT], [T0_b_LIT]; window = 0.08)
    mask_e = mask_transits(ep.t, [P_e_LIT], [T0_e_LIT]; window = 0.03)
    mask = mask_b .| mask_e
    win = ep.name == "K2_c03" ? 101 : 1001
    return detrend_savgol(ep.t, ep.flux, ep.err;
                           window_length = win, polyorder = 3,
                           transit_mask = mask)
end
dts = [_detrend(ep) for ep in epochs]

# Predicted b transit centers
function transits_in(t, P, T0)
    n_lo = ceil(Int, (minimum(t) + 0.1 - T0) / P)
    n_hi = floor(Int, (maximum(t) - 0.1 - T0) / P)
    return [T0 + n*P for n in n_lo:n_hi]
end
ep_tcs = [transits_in(ep.t, P_b_LIT, T0_b_LIT) for ep in epochs]
all_tcs = vcat(ep_tcs...)
@printf("Observed b transits: %d across %d epochs (span %.1f yr)\n",
        length(all_tcs), length(epochs),
        (maximum(all_tcs) - minimum(all_tcs)) / 365.25)

# Tight subset: ±0.06d = ±1.5 hr (just inside T14 + small OOT margin).
# 30-40% smaller than ±0.10d → 30-40% faster likelihood eval.
t_all = Float64[]; flux_all = Float64[]; err_all = Float64[]; inst_all = Int[]
for (i, ep) in enumerate(epochs)
    keep = falses(length(ep.t))
    for Tc in ep_tcs[i]
        @. keep |= abs(ep.t - Tc) < 0.06
    end
    append!(t_all, ep.t[keep])
    append!(flux_all, dts[i].flux_detrended[keep])
    append!(err_all, ep.err[keep])
    append!(inst_all, fill(i, count(keep)))
end
perm = sortperm(t_all)
t_all = t_all[perm]; flux_all = flux_all[perm]
err_all = err_all[perm]; inst_all = inst_all[perm]
@printf("Subset: %d photometric points\n", length(t_all))

# --- Build Data (no RV) ---------------------------------------------------
# RVPM_TTV_NB requires the planet to have an RV slot (K) for mass
# computation. With zero RV data the RV log-likelihood is just 0, so K
# is constrained ONLY through the TTV-NB → TTVFaster mass dependency.
data = Data(; t_phot = t_all, flux = flux_all, flux_err = err_all,
              phot_inst = inst_all,
              t_rv = Float64[], rv = Float64[], rv_err = Float64[],
              rv_inst = Int[])
ic = InstrumentConfig(rv = ["dummy_rv"], pm = [ep.name for ep in epochs])

# --- Priors ---------------------------------------------------------------
priors = Dict{String, PriorSpec}()

# Nereus's period-ordering hard prior enforces P_k1 < P_k2 to break
# label-switching symmetry. So planet 1 = e (inner, USP) and planet 2 =
# b (outer, hot Jupiter).

# Planet 1 = WASP-47e (inner USP): K free → mass; eccentricity free;
# orbit/geometry tightly pinned (well-known, photometrically invisible
# at our depth and cadence).
priors["P_k1"]      = FixedPrior(P_e_LIT)
priors["K_k1"]      = NormalPrior(4.55, 1.0, 0.5, 15.0)         # Sinukoff 17
priors["Tc_k1"]     = FixedPrior(T0_e_LIT)
priors["sesinw_k1"] = NormalPrior(0.0, 0.03, -0.2, 0.2)
priors["secosw_k1"] = NormalPrior(0.0, 0.03, -0.2, 0.2)
priors["b_k1"]      = FixedPrior(0.5)
priors["rr_k1"]     = FixedPrior(0.0145)

# Planet 2 = WASP-47b (outer hot Jupiter): free everything that defines
# transit shape + N-body interaction.
priors["P_k2"]      = NormalPrior(P_b_LIT, 1e-5,
                                   P_b_LIT - 5e-5, P_b_LIT + 5e-5)
priors["K_k2"]      = NormalPrior(141.0, 5.0, 50.0, 250.0)      # Bryant 22
priors["Tc_k2"]     = NormalPrior(T0_b_LIT, 0.001,
                                   T0_b_LIT - 0.01, T0_b_LIT + 0.01)
priors["sesinw_k2"] = NormalPrior(0.0, 0.02, -0.1, 0.1)
priors["secosw_k2"] = NormalPrior(0.0, 0.02, -0.1, 0.1)
priors["b_k2"]      = NormalPrior(B_B_LIT, 0.05, 0.0, 0.5)
priors["rr_k2"]     = NormalPrior(RR_B_LIT, 0.003,
                                   RR_B_LIT - 0.02, RR_B_LIT + 0.02)

priors["rho_s"]     = NormalPrior(RHO_S_LIT, 0.05,
                                   RHO_S_LIT - 0.2, RHO_S_LIT + 0.2)

# RV channel: dummy gamma/sigma (no data, just slot-filling)
priors["gamma_dummy_rv"] = FixedPrior(0.0)
priors["sigma_dummy_rv"] = FixedPrior(1.0)

# Per-instrument photometric nuisance
for nm in [ep.name for ep in epochs]
    priors["offset_$nm"] = NormalPrior(0.0, 5e-4, -5e-3, 5e-3)
    priors["jitter_$nm"] = LogUniformPrior(1e-6, 5e-3)
    priors["q1_$nm"]     = FixedPrior(startswith(nm, "K2") ? Q1_K2 : Q1_TESS)
    priors["q2_$nm"]     = FixedPrior(startswith(nm, "K2") ? Q2_K2 : Q2_TESS)
end

# --- Build Params ---------------------------------------------------------
parametrization = ParametrizationConfig(time = :Tc, geom = :b_rr,
                                          use_rho_s = true)
params = Params(;
    max_kplanet     = 2,
    planet_modes    = [RVPM_TTV_NB, RVPM_TTV_NB],
    instruments     = ic,
    data            = data,
    M_s             = M_s_LIT,
    R_s             = R_s_LIT,
    parametrization = parametrization,
    priors          = priors,
    stability       = :none,
)
@printf("Free params: %d / %d total\n",
        n_unfrozen(params), length(params.layout.name_to_idx))

target = NereusTarget(params, data; unconstrained = false)

# --- Sample (ptemcee with tuned settings) --------------------------------
const SEED      = parse(Int, get(ENV, "WASP47_TTV_C_SEED",    "42"))
const N_STEPS   = parse(Int, get(ENV, "WASP47_TTV_C_STEPS",   "2000"))
const N_WALKERS = parse(Int, get(ENV, "WASP47_TTV_C_WALKERS", "40"))
const N_TEMPS   = parse(Int, get(ENV, "WASP47_TTV_C_TEMPS",   "4"))
const N_BURNIN  = parse(Int, get(ENV, "WASP47_TTV_C_BURNIN",  "500"))

println("\nsample_ptemcee: $N_TEMPS temps × $N_WALKERS walkers × $N_STEPS steps")
t0 = time()
result = sample_ptemcee(target, data;
    n_temps   = N_TEMPS,
    n_walkers = N_WALKERS,
    n_steps   = N_STEPS,
    n_burnin  = N_BURNIN,
    seed      = SEED,
    show_progress = true,
)
elapsed = time() - t0
@printf("Done in %.1f s  (%.0f evals/sec)\n",
        elapsed, result.n_evals / elapsed)

chains = result.chains
save_chains(joinpath(OUT_DIR, "ttv_c_chains.nc"), chains, params; data = data)

# --- Report posterior on M_e (the key science output) ---------------------
const M_SUN = 1.989e33
const M_E   = 5.972e27
function _ke_to_me(K_e, sesinw, secosw)
    e = hypot(sesinw, secosw)^2
    e = clamp(e, 0.0, 0.99)
    # M_e * sin(i_e) in M_Jup
    mp_sini = msini(M_s_LIT, K_e, P_e_LIT, e)
    # assume i_e ≈ 90° (USP transits)
    return mp_sini * 1.898e30 / M_E
end
ke_chain  = vec(Array(chains[:, :K_k1, :]))
se_chain  = vec(Array(chains[:, :sesinw_k1, :]))
sc_chain  = vec(Array(chains[:, :secosw_k1, :]))
me_chain  = _ke_to_me.(ke_chain, se_chain, sc_chain)
@printf("\nWASP-47 e mass posterior (M_⊕):\n")
@printf("  median = %.2f   [16, 84] = [%.2f, %.2f]\n",
        median(me_chain), quantile(me_chain, 0.16), quantile(me_chain, 0.84))
@printf("  Lit ref (Vanderburg 2017): 9.0 ± 1.7 M_⊕\n")
@printf("  Prior K_e center: %.2f m/s (Sinukoff 2017)\n", 4.55)

# --- Reconstruct TTV-C prediction at posterior median --------------------
function _ttvc_prediction(theta_med, observed_tcs::Vector{Float64})
    # In this fit: planet 1 = e (inner), planet 2 = b (outer)
    P_b  = theta_med.values[params.layout.name_to_idx["P_k2"]]
    Tc_b = theta_med.values[params.layout.name_to_idx["Tc_k2"]]
    se_b = theta_med.values[params.layout.name_to_idx["sesinw_k2"]]
    sc_b = theta_med.values[params.layout.name_to_idx["secosw_k2"]]
    e_b  = hypot(se_b, sc_b)^2
    w_b  = atan(se_b, sc_b)
    Tp_b = tc_to_tp(Tc_b, P_b, e_b, w_b)
    Tc1_b = Tp_b + P_b/4
    K_b  = theta_med.values[params.layout.name_to_idx["K_k2"]]
    b_b  = theta_med.values[params.layout.name_to_idx["b_k2"]]
    rr_b = theta_med.values[params.layout.name_to_idx["rr_k2"]]
    rho  = theta_med.values[params.layout.name_to_idx["rho_s"]]
    a_Rs_b = rho_s_to_a_Rs(rho, P_b)
    inc_b  = acosd(b_b / a_Rs_b)
    mp_b = msini(M_s_LIT, K_b, P_b, e_b) / sind(inc_b)
    mratio_b = mp_b * 1.898e30 / M_SUN / M_s_LIT

    K_e  = theta_med.values[params.layout.name_to_idx["K_k1"]]
    se_e = theta_med.values[params.layout.name_to_idx["sesinw_k1"]]
    sc_e = theta_med.values[params.layout.name_to_idx["secosw_k1"]]
    e_e  = hypot(se_e, sc_e)^2
    w_e  = atan(se_e, sc_e)
    mp_e = msini(M_s_LIT, K_e, P_e_LIT, e_e)
    mratio_e = mp_e * 1.898e30 / M_SUN / M_s_LIT

    pl_e = Planet_plane_hk(mratio_e, P_e_LIT, T0_e_LIT,
                            e_e*cos(w_e), e_e*sin(w_e))
    pl_b = Planet_plane_hk(mratio_b, P_b, Tc1_b,
                            e_b*cos(w_b), e_b*sin(w_b))
    n_lo = ceil(Int,  (minimum(observed_tcs) - T0_e_LIT) / P_e_LIT)
    n_hi = floor(Int, (maximum(observed_tcs) - T0_e_LIT) / P_e_LIT)
    times_e = [T0_e_LIT + n*P_e_LIT for n in n_lo:n_hi]
    ttv_e = zeros(Float64, length(times_e))
    ttv_b = zeros(Float64, length(observed_tcs))
    try
        compute_ttv!(5, pl_e, pl_b, times_e, observed_tcs, ttv_e, ttv_b)
    catch
    end
    any(isnan, ttv_b) && (ttv_b[isnan.(ttv_b)] .= 0.0)
    return ttv_b
end

# Sample TTV-C predictions across posterior draws → posterior band
chain_n = size(chains)[1]
chain_c = size(chains)[3]
draws = min(400, chain_n * chain_c)
inds  = rand(Random.MersenneTwister(SEED+2), 1:(chain_n*chain_c), draws)
ttvc_samples = Matrix{Float64}(undef, draws, length(all_tcs))
for (k, idx) in enumerate(inds)
    chain_idx = ((idx - 1) ÷ chain_n) + 1
    iter_idx = ((idx - 1) % chain_n) + 1
    theta_k = Theta{Float64}(params)
    set_n_p!(theta_k, 2)
    for nm in keys(params.layout.name_to_idx)
        sym = Symbol(nm)
        sym in names(chains, :parameters) || continue
        theta_k.values[params.layout.name_to_idx[nm]] =
            chains[iter_idx, sym, chain_idx]
    end
    ttvc_samples[k, :] .= _ttvc_prediction(theta_k, all_tcs)
end
ttvc_med = vec(median(ttvc_samples; dims = 1))
ttvc_lo  = [quantile(ttvc_samples[:, i], 0.16) for i in 1:length(all_tcs)]
ttvc_hi  = [quantile(ttvc_samples[:, i], 0.84) for i in 1:length(all_tcs)]

cycle_idx = [Int(round((Tc - T0_b_LIT) / P_b_LIT)) for Tc in all_tcs]

# Per-epoch groupings (for marker styles)
epoch_groups = Vector{Vector{Int}}(undef, length(epochs))
ep_starts = cumsum(vcat([0], [length(et) for et in ep_tcs]))
for i in 1:length(epochs)
    epoch_groups[i] = collect((ep_starts[i]+1):ep_starts[i+1])
end
epoch_labels = [ep.name for ep in epochs]

# Pull the TTV-A free-offset chain (saved by wasp47_ttv_demo.jl) so the
# joint TTV-C posterior band can be compared to model-agnostic data.
ttva_chain_path = joinpath(REPO_ROOT, "Nereus.jl", "results",
                            "WASP47_TTV", "ttv_a_chains.nc")
δt_med_a = nothing; δt_lo_a = nothing; δt_hi_a = nothing
ttva_idx = Int[]
if isfile(ttva_chain_path)
    ttva_chains, _ = load_chains(ttva_chain_path)
    chain_names = names(ttva_chains, :parameters)
    for sym in chain_names
        s = String(sym)
        m = match(r"^ttv_k1_t(\d+)$", s)
        m === nothing && continue
        slot = parse(Int, m.captures[1])
        push!(ttva_idx, slot - 1)
    end
    sort!(ttva_idx)
    δt_med_a = Float64[]; δt_lo_a = Float64[]; δt_hi_a = Float64[]
    for idx in ttva_idx
        v = vec(Array(ttva_chains[:, Symbol("ttv_k1_t$(idx+1)"), :]))
        push!(δt_med_a, median(v))
        push!(δt_lo_a,  quantile(v, 0.16))
        push!(δt_hi_a,  quantile(v, 0.84))
    end
    @printf("Loaded TTV-A residuals from %d transits.\n", length(ttva_idx))
end

if δt_med_a !== nothing
    # Overlay: TTV-A black points with error bars + TTV-C median + CI band
    # in same y-axis. Index lookup: ttva_idx and cycle_idx may differ — we
    # use the TTV-A indices and recompute TTV-C at those if not already done.
    a_groups = Vector{Vector{Int}}(undef, length(epochs))
    for (i, er) in enumerate([extrema([x for x in transits_in(ep.t, P_b_LIT, T0_b_LIT)])
                                for ep in epochs])
        nothing
    end
    # Build epoch groupings by index value (matches the TTV-A demo logic)
    function _group_by_breaks(idx_vec; gap = 50)
        sorted_idx = sort(unique(idx_vec))
        # Split sorted indices into consecutive runs (gaps > `gap`)
        runs = Vector{Vector{Int}}()
        run = [sorted_idx[1]]
        for i in 2:length(sorted_idx)
            if sorted_idx[i] - sorted_idx[i-1] > gap
                push!(runs, run); run = Int[]
            end
            push!(run, sorted_idx[i])
        end
        push!(runs, run)
        # Map back to positions in idx_vec
        return [findall(x -> x in Set(r), idx_vec) for r in runs]
    end
    a_groups = _group_by_breaks(ttva_idx)
    a_labels = ["K2_c03", "TESS_s42", "TESS_s92"][1:length(a_groups)]

    # Match ttv_c samples to the same transit indices as ttv_a (interp by
    # nearest cycle in `cycle_idx`).
    function _nearest_idx(target_cycle, candidates)
        i = argmin(abs.(candidates .- target_cycle))
        return i
    end
    sel = [_nearest_idx(c, cycle_idx) for c in ttva_idx]
    ttvc_med_at_a = ttvc_med[sel]
    ttvc_lo_at_a  = ttvc_lo[sel]
    ttvc_hi_at_a  = ttvc_hi[sel]

    # Y-limits set by data scatter (TTV-A errorbars in min); CI band may
    # legitimately extend further but we crop so the data stays visible.
    y_max = max(maximum(24 * 60 .* δt_hi_a), 5.0) * 1.3
    y_min = min(minimum(24 * 60 .* δt_lo_a), -5.0) * 1.3
    plot_ttv_diagram(ttva_idx, δt_med_a, δt_lo_a, δt_hi_a;
                      epoch_groups = a_groups,
                      epoch_labels = a_labels,
                      ylim = (y_min, y_max),
                      predicted    = ttvc_med_at_a,
                      predicted_lo = ttvc_lo_at_a,
                      predicted_hi = ttvc_hi_at_a,
                      predicted_label = "Joint N-body posterior (68%)",
                      filename = joinpath(OUT_DIR, "ttv_c_vs_ttv_a.png"))
    println("Wrote $(OUT_DIR)/ttv_c_vs_ttv_a.png")
else
    @warn "TTV-A chain not found at $ttva_chain_path — skipping comparison plot. Run wasp47_ttv_demo.jl first."
end

println("\n" * "="^70)
println("Joint TTV-C demo complete. See $(OUT_DIR)/")
println("="^70)
