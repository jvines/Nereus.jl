#!/usr/bin/env julia
# WASP-47 TTV demo — multi-epoch K2 c03 + TESS s42 + TESS s92.
#
# K2 c03 (Becker+ 2015): ~17 transits, σ_phot ≈ 6×10⁻⁵, anchors timing.
# TESS s42 (2021): 5 transits at σ_phot ≈ 5×10⁻³, lever arm to 2021.
# TESS s92 (2025): 6 transits, lever arm to 2025. ~3-epoch span ≈ 6 yr,
# ~925 b-orbital cycles. Only observed transits get free δt slots; the
# rest are FixedPrior(0) to keep the MCMC dimensionality reasonable.
#
# Outputs in results/WASP47_TTV/:
#   oc_diagram_ttv_a.png        — O−C across all 3 epochs
#   oc_diagram_ttv_compare.png  — same with TTV-C N-body overlay
#   transit_overlay.png         — per-transit data + model

using Nereus
using DelimitedFiles: readdlm
using Statistics: median, std, mean, quantile
using Printf
using Random
using TTVFaster: Planet_plane_hk, compute_ttv!

const REPO_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const DATADIR   = joinpath(REPO_ROOT, "data", "WASP47")
const OUT_DIR   = joinpath(REPO_ROOT, "Nereus.jl", "results", "WASP47_TTV")
mkpath(OUT_DIR)

println("="^70)
println("WASP-47 TTV demo — K2 c03 + TESS s42 + TESS s92")
println("="^70)

# --- Literature ephemeris (Bryant & Bayliss 2022, Becker 2015) -----------
const P_b_LIT  = 4.1591287
# T0 anchored empirically to the first K2 c03 transit (BJD 2456978.819
# from the EVEREST CSV). Becker+ 2015's published value 2456979.7613
# is offset by 0.94 d from this CSV — likely a Becker-to-EVEREST time-
# convention discrepancy. The shape of the O−C diagram is invariant
# under T0 choice (constant offsets fold into TTV_0).
const T0_b_LIT = 2456978.819
# WASP-47e T0 similarly anchored from the LC; e's signal is too weak to
# extract directly so we shift by the same 0.94 d.
const P_e_LIT  = 0.789593
const T0_e_LIT = 2457006.99267
const M_s_LIT  = 1.04
const R_s_LIT  = 1.137
# Bryant 2022 transit params (used as fixed priors so the fit isolates TTV)
const RR_B_LIT  = 0.1027
const B_B_LIT   = 0.13
const RHO_S_LIT = 0.96
const Q1_K2     = 0.45      # Claret quadratic LD, K2 bandpass
const Q2_K2     = 0.30
const Q1_TESS   = 0.36      # Claret quadratic LD, TESS bandpass
const Q2_TESS   = 0.32

# --- Load all 3 epochs ----------------------------------------------------
struct Epoch
    name::String
    t::Vector{Float64}
    flux::Vector{Float64}
    err::Vector{Float64}
    sigma_floor::Float64
end

function load_epoch(name::AbstractString, path::AbstractString, bjd_offset::Real)
    lc = load_tess_lc(path)
    t = lc.t .+ bjd_offset
    return Epoch(name, t, lc.flux, lc.flux_err, median(lc.flux_err))
end

ep_k2  = load_epoch("K2_c03",  joinpath(DATADIR, "WASP-47_k2_c03_slc.csv"), 0.0)
ep_s42 = load_epoch("TESS_s42", joinpath(DATADIR, "WASP-47_tess_s42_lc.csv"), 2_457_000.0)
ep_s92 = load_epoch("TESS_s92", joinpath(DATADIR, "WASP-47_tess_s92_lc.csv"), 2_457_000.0)
epochs = [ep_k2, ep_s42, ep_s92]
for ep in epochs
    @printf("  %-8s : %5d pts, BJD %.2f .. %.2f, σ_med = %.2e\n",
            ep.name, length(ep.t), ep.t[1], ep.t[end], ep.sigma_floor)
end

# --- Predicted transit centers for b in each epoch ------------------------
function transits_in(t::AbstractVector, P::Real, T0::Real)
    n_lo = ceil(Int,  (minimum(t) + 0.1 - T0) / P)
    n_hi = floor(Int, (maximum(t) - 0.1 - T0) / P)
    return [(n, T0 + n * P) for n in n_lo:n_hi]
end

ep_transits = [transits_in(ep.t, P_b_LIT, T0_b_LIT) for ep in epochs]
all_observed = vcat(ep_transits...)         # (transit_idx, Tc) pairs
sort!(all_observed; by = x -> x[1])
observed_idx = [x[1] for x in all_observed]
observed_tcs = [x[2] for x in all_observed]
n_obs_tr = length(observed_idx)
@printf("\nObserved b transits across all epochs: %d (idx %d..%d)\n",
        n_obs_tr, first(observed_idx), last(observed_idx))

# Total slots needed: covers transit `0` (anchored at T0_b_LIT) up to the
# highest observed index. Slot i+1 ↔ transit number i.
n_ttv_slots = maximum(observed_idx) + 1
@printf("TTV slot count: %d (only %d are free; rest fixed at 0)\n",
        n_ttv_slots, n_obs_tr)

# --- Detrend each epoch (savgol with transit mask) ----------------------
function _detrend_epoch(ep::Epoch)
    # Mask width must comfortably exceed T14 + ingress/egress (~3.5 hr
    # for WASP-47b → half-width 0.075 d) so savgol does not absorb the
    # transit dip into the trend. K2 short cadence has 1-min sampling
    # → many more points per window than TESS; scale savgol window
    # accordingly. K2 SC: 1001 pts × 1 min = ~17 hr smoothing window.
    mask_b = mask_transits(ep.t, [P_b_LIT], [T0_b_LIT]; window = 0.08)
    mask_e = mask_transits(ep.t, [P_e_LIT], [T0_e_LIT]; window = 0.03)
    mask = mask_b .| mask_e
    return detrend_savgol(ep.t, ep.flux, ep.err;
                           window_length = 1001, polyorder = 3,
                           transit_mask = mask)
end

dts = [_detrend_epoch(ep) for ep in epochs]

# --- Subset to ±0.10 d around predicted b transits ----------------------
function _subset(ep::Epoch, dt_result, tcs::Vector{Float64})
    keep = falses(length(ep.t))
    for Tc in tcs
        @. keep |= abs(ep.t - Tc) < 0.10
    end
    return ep.t[keep], dt_result.flux_detrended[keep], ep.err[keep], keep
end

t_all   = Float64[]
flux_all = Float64[]
err_all = Float64[]
inst_all = Int[]
for (i, ep) in enumerate(epochs)
    tcs = [x[2] for x in ep_transits[i]]
    t_sub, f_sub, e_sub, _ = _subset(ep, dts[i], tcs)
    append!(t_all, t_sub)
    append!(flux_all, f_sub)
    append!(err_all, e_sub)
    append!(inst_all, fill(i, length(t_sub)))
    @printf("  %s subset: %d points\n", ep.name, length(t_sub))
end
perm = sortperm(t_all)
t_all = t_all[perm]; flux_all = flux_all[perm]
err_all = err_all[perm]; inst_all = inst_all[perm]

# --- Build Params ---------------------------------------------------------
data = Data(; t_phot = t_all, flux = flux_all, flux_err = err_all,
              phot_inst = inst_all)
inst_names = [ep.name for ep in epochs]
ic = InstrumentConfig(rv = String[], pm = inst_names)

priors = Dict{String, PriorSpec}()
# Pin orbit + LD to literature so the chain only fits TTVs + per-epoch
# offset/jitter. The interesting question is: do the K2 anchor + TESS
# 2021/2025 epochs jointly recover a credible TTV pattern for b?
# Let P_b float in a narrow prior so it can absorb any small linear
# drift (~0.6 sec per period off literature would explain a +10 min
# accumulated offset across ~925 cycles). Without this, the chain would
# dump the drift into the TTV slots as a fake "increasing offset" pattern.
priors["P_k1"]      = NormalPrior(P_b_LIT, 1e-5,
                                   P_b_LIT - 5e-5, P_b_LIT + 5e-5)
priors["sesinw_k1"] = FixedPrior(0.0)
priors["secosw_k1"] = FixedPrior(0.0)
# Fix Tc_k1 to the linear ephemeris zero point. Letting it float is
# 100% degenerate with a constant offset in all per-transit TTV slots:
# (Tc_k1 + δ, ttv_kI − δ) is identical to (Tc_k1, ttv_kI) for every I.
# Fixing breaks the degeneracy; the chain can still float TTV_0 freely
# to absorb any small Tc bias.
priors["Tc_k1"]     = FixedPrior(T0_b_LIT)
priors["b_k1"]      = FixedPrior(B_B_LIT)
priors["rr_k1"]     = NormalPrior(RR_B_LIT, 0.005,
                                   RR_B_LIT - 0.02, RR_B_LIT + 0.02)
priors["rho_s"]     = NormalPrior(RHO_S_LIT, 0.05,
                                   RHO_S_LIT - 0.2, RHO_S_LIT + 0.2)
for nm in inst_names
    priors["offset_$nm"] = NormalPrior(0.0, 5e-4, -5e-3, 5e-3)
    priors["jitter_$nm"] = LogUniformPrior(1e-6, 5e-3)
    if startswith(nm, "K2")
        priors["q1_$nm"] = FixedPrior(Q1_K2)
        priors["q2_$nm"] = FixedPrior(Q2_K2)
    else
        priors["q1_$nm"] = FixedPrior(Q1_TESS)
        priors["q2_$nm"] = FixedPrior(Q2_TESS)
    end
end
# TTV priors: tight Normal(0, 0.002d = ~3 min) on observed transits,
# fixed at 0 elsewhere. The hard bound ±0.01 d (~14 min) keeps the
# search local.
observed_set = Set(observed_idx)
for i in 1:n_ttv_slots
    slot_transit = i - 1
    if slot_transit in observed_set
        priors["ttv_k1_t$i"] = NormalPrior(0.0, 0.002, -0.01, 0.01)
    else
        priors["ttv_k1_t$i"] = FixedPrior(0.0)
    end
end

parametrization = ParametrizationConfig(time = :Tc, geom = :b_rr,
                                          use_rho_s = true)
params = Params(;
    max_kplanet     = 1,
    planet_modes    = [PM_TTV],
    instruments     = ic,
    data            = data,
    M_s             = M_s_LIT,
    R_s             = R_s_LIT,
    parametrization = parametrization,
    priors          = priors,
    stability       = :none,
    ttv_n_transits  = Dict(1 => n_ttv_slots),
)
@printf("\nFree params: %d  (of %d total)\n",
        n_unfrozen(params), length(params.layout.name_to_idx))

target = NereusTarget(params, data; unconstrained = false)

# --- Sample ---------------------------------------------------------------
const SEED      = parse(Int, get(ENV, "WASP47_TTV_SEED",    "42"))
const N_STEPS   = parse(Int, get(ENV, "WASP47_TTV_STEPS",   "1500"))
const N_WALKERS = parse(Int, get(ENV, "WASP47_TTV_WALKERS", "80"))
const N_TEMPS   = parse(Int, get(ENV, "WASP47_TTV_TEMPS",   "5"))
const N_BURNIN  = parse(Int, get(ENV, "WASP47_TTV_BURNIN",  "500"))

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
@printf("Done in %.1f s\n", time() - t0)

chains = result.chains
save_chains(joinpath(OUT_DIR, "ttv_a_chains.nc"), chains, params; data = data)

# --- Extract TTV offsets posteriors --------------------------------------
function _stats(v::AbstractVector)
    return (med = median(v),
            lo  = quantile(v, 0.16),
            hi  = quantile(v, 0.84))
end

δt_med = Float64[]; δt_lo = Float64[]; δt_hi = Float64[]
for idx in observed_idx
    slot = idx + 1
    s = _stats(vec(Array(chains[:, Symbol("ttv_k1_t$slot"), :])))
    push!(δt_med, s.med); push!(δt_lo, s.lo); push!(δt_hi, s.hi)
end

println("\nRecovered WASP-47b TTV offsets:")
for (i, idx) in enumerate(observed_idx)
    epoch_tag = idx <= 20 ? "K2" : idx < 900 ? "TESSs42" : "TESSs92"
    @printf("  idx %3d  (%s):  δt = %+7.2f ± %5.2f min\n",
            idx, epoch_tag,
            24*60*δt_med[i], 24*60*(δt_hi[i] - δt_lo[i])/2)
end

# --- Plot 1: TTV-A O-C, one panel per epoch (no wasted x-axis) -----------
# Group observed transits by which epoch's index range they fall in.
epoch_ranges = [(name = ep.name,
                  lo = first([x[1] for x in ep_transits[i]]),
                  hi = last([x[1] for x in ep_transits[i]]))
                 for (i, ep) in enumerate(epochs)]
epoch_groups = Vector{Vector{Int}}(undef, length(epoch_ranges))
for (i, er) in enumerate(epoch_ranges)
    epoch_groups[i] = findall(idx -> er.lo <= idx <= er.hi, observed_idx)
end
epoch_labels = [ep.name for ep in epochs]
plot_ttv_diagram(observed_idx, δt_med, δt_lo, δt_hi;
                  epoch_groups = epoch_groups,
                  epoch_labels = epoch_labels,
                  filename = joinpath(OUT_DIR, "oc_diagram_ttv_a.png"))
println("\nWrote $(OUT_DIR)/oc_diagram_ttv_a.png")

# --- TTV-C: b ↔ e N-body prediction at observed transit times ------------
const M_E_OVER_M_SUN = 3.003e-6
mratio_b = 1.21e-3 / M_s_LIT
mratio_e = 9.0 * M_E_OVER_M_SUN / M_s_LIT
let
    # Monte Carlo over input uncertainties so the N-body prediction
    # gets a credibility band comparable to the data error bars.
    # Input priors (Vanderburg+ 2017, Bryant+ 2022):
    #   M_e ~ N(9.0, 1.7) M⊕            (Vanderburg 2017)
    #   M_b ~ N(363, 13) M⊕             (Bryant 2022)
    #   e_e ~ HalfNormal(σ=0.03)        (consistent with 0)
    #   e_b ~ HalfNormal(σ=0.005)
    #   ω_e, ω_b ~ Uniform(0, 2π)       (unconstrained)
    n_draws = 500
    times_b = collect(observed_tcs)
    times_e = let
        n_lo = ceil(Int,  (minimum(times_b) - T0_e_LIT) / P_e_LIT)
        n_hi = floor(Int, (maximum(times_b) - T0_e_LIT) / P_e_LIT)
        [T0_e_LIT + n * P_e_LIT for n in n_lo:n_hi]
    end
    samples = Matrix{Float64}(undef, n_draws, length(times_b))
    rng = Random.MersenneTwister(SEED + 1)
    n_ok = 0
    for s in 1:n_draws
        m_e_ratio = max(0.5, 9.0 + 1.7 * randn(rng)) *
                     M_E_OVER_M_SUN / M_s_LIT
        m_b_ratio = max(50.0, 363.0 + 13.0 * randn(rng)) *
                     M_E_OVER_M_SUN / M_s_LIT
        e_e = abs(0.03 * randn(rng))
        e_b = abs(0.005 * randn(rng))
        w_e = 2π * rand(rng)
        w_b = 2π * rand(rng)
        pl_e = Planet_plane_hk(m_e_ratio, P_e_LIT, T0_e_LIT,
                                e_e*cos(w_e), e_e*sin(w_e))
        pl_b = Planet_plane_hk(m_b_ratio, P_b_LIT, T0_b_LIT,
                                e_b*cos(w_b), e_b*sin(w_b))
        ttv_e_s = zeros(Float64, length(times_e))
        ttv_b_s = zeros(Float64, length(times_b))
        try
            compute_ttv!(5, pl_e, pl_b, times_e, times_b, ttv_e_s, ttv_b_s)
        catch err_
            err_ isa DomainError || err_ isa ArgumentError ||
                err_ isa AssertionError || rethrow()
            continue
        end
        any(isnan, ttv_b_s) && continue
        n_ok += 1
        samples[n_ok, :] .= ttv_b_s
    end
    samples = samples[1:n_ok, :]
    ttv_b_med = vec(median(samples; dims = 1))
    ttv_b_lo  = [quantile(samples[:, i], 0.16) for i in 1:length(times_b)]
    ttv_b_hi  = [quantile(samples[:, i], 0.84) for i in 1:length(times_b)]
    @printf("\nTTV-C: %d/%d valid TTVFaster draws.\n", n_ok, n_draws)
    @printf("Predicted b: median peak-to-peak %.2f s ;  68%% band width median %.2f s\n",
            60 * 24 * 60 * (maximum(ttv_b_med) - minimum(ttv_b_med)),
            60 * 24 * 60 * median(ttv_b_hi .- ttv_b_lo))
    plot_ttv_diagram(observed_idx, δt_med, δt_lo, δt_hi;
                      epoch_groups = epoch_groups,
                      epoch_labels = epoch_labels,
                      predicted    = ttv_b_med,
                      predicted_lo = ttv_b_lo,
                      predicted_hi = ttv_b_hi,
                      predicted_label = "N-body model",
                      filename = joinpath(OUT_DIR, "oc_diagram_ttv_compare.png"))
    println("Wrote $(OUT_DIR)/oc_diagram_ttv_compare.png")
end

# --- Per-transit overlay --------------------------------------------------
function _med_or_fixed(chains, params, name::AbstractString)
    sym = Symbol(name)
    if sym in names(chains, :parameters)
        return median(vec(Array(chains[:, sym, :])))
    end
    return Nereus.fixed_value(params.config.priors[name])
end

theta_med = Theta{Float64}(params)
set_n_p!(theta_med, 1)
for nm in keys(params.layout.name_to_idx)
    sym = Symbol(nm)
    if sym in names(chains, :parameters)
        theta_med.values[params.layout.name_to_idx[nm]] =
            median(vec(Array(chains[:, sym, :])))
    end
end
pred, _ = phot_predictions(theta_med, data)
@printf("Median-orbit prediction: %d points below 0.99 flux\n",
        count(<(0.99), pred))

# Pick up to 8 representative transits to overlay (mix of K2 + TESS)
sel_idx = vcat(observed_idx[1:min(3, length(observed_idx))],
                observed_idx[max(1, end-4):end])
unique!(sel_idx)
sel_idx = sort(sel_idx)[1:min(8, end)]

t_segs    = Vector{Vector{Float64}}()
flux_segs = Vector{Vector{Float64}}()
err_segs  = Vector{Vector{Float64}}()
model_segs = Vector{Vector{Float64}}()
tcs_used = Float64[]
idx_used = Int[]
for idx in sel_idx
    Tc_pred = T0_b_LIT + idx * P_b_LIT
    pts = findall(x -> abs(x - Tc_pred) < 0.08, t_all)
    isempty(pts) && continue
    push!(t_segs,    t_all[pts])
    push!(flux_segs, flux_all[pts])
    push!(err_segs,  err_all[pts])
    push!(model_segs, pred[pts])
    push!(tcs_used, Tc_pred)
    push!(idx_used, idx)
end

plot_transit_overlay(t_segs, flux_segs, err_segs, model_segs;
                      transit_idx = idx_used,
                      tcs = tcs_used,
                      filename = joinpath(OUT_DIR, "transit_overlay.png"))
println("Wrote $(OUT_DIR)/transit_overlay.png")

println("\n" * "="^70)
println("Demo complete. See $(OUT_DIR)/")
println("="^70)
