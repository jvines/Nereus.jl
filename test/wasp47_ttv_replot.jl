#!/usr/bin/env julia
# Replot TTV-C compare panel using saved chains + literature ecc values.
# Avoids the 6-minute MCMC rerun when only the prediction overlay changes.

using Nereus
using Statistics: median, quantile
using Printf
using TTVFaster: Planet_plane_hk, compute_ttv!

const REPO_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const OUT_DIR   = joinpath(REPO_ROOT, "Nereus.jl", "results", "WASP47_TTV")

const P_b_LIT  = 4.1591287
const T0_b_LIT = 2456979.7613
const P_e_LIT  = 0.789593
const T0_e_LIT = 2457007.93267
const M_s_LIT  = 1.04

chains, _ = load_chains(joinpath(OUT_DIR, "ttv_a_chains.nc"))
chain_names = names(chains, :parameters)

# Identify observed transit slots: any ttv_k1_tI column with non-trivial
# posterior std (free params have wider chains than fixed ones).
observed_slots = Int[]
δt_med = Float64[]; δt_lo = Float64[]; δt_hi = Float64[]
for sym in chain_names
    s = String(sym)
    m = match(r"^ttv_k1_t(\d+)$", s)
    m === nothing && continue
    slot = parse(Int, m.captures[1])
    v = vec(Array(chains[:, sym, :]))
    push!(observed_slots, slot)
    push!(δt_med, median(v))
    push!(δt_lo,  quantile(v, 0.16))
    push!(δt_hi,  quantile(v, 0.84))
end
perm = sortperm(observed_slots)
observed_slots = observed_slots[perm]
observed_idx = observed_slots .- 1
δt_med = δt_med[perm]; δt_lo = δt_lo[perm]; δt_hi = δt_hi[perm]
observed_tcs = T0_b_LIT .+ observed_idx .* P_b_LIT

@printf("Loaded %d free TTV slots from saved chains\n", length(observed_idx))

# TTV-C with Vanderburg 2017 eccentricities
const M_E_OVER_M_SUN = 3.003e-6
mratio_b = 1.21e-3 / M_s_LIT
mratio_e = 9.0 * M_E_OVER_M_SUN / M_s_LIT
e_e_lit, w_e_lit = 0.03, 0.4
e_b_lit, w_b_lit = 0.0028, 1.0
pl_e = Planet_plane_hk(mratio_e, P_e_LIT, T0_e_LIT,
                        e_e_lit*cos(w_e_lit), e_e_lit*sin(w_e_lit))
pl_b = Planet_plane_hk(mratio_b, P_b_LIT, T0_b_LIT,
                        e_b_lit*cos(w_b_lit), e_b_lit*sin(w_b_lit))

times_b = collect(observed_tcs)
n_lo = ceil(Int,  (minimum(times_b) - T0_e_LIT) / P_e_LIT)
n_hi = floor(Int, (maximum(times_b) - T0_e_LIT) / P_e_LIT)
times_e = [T0_e_LIT + n * P_e_LIT for n in n_lo:n_hi]
ttv_e = zeros(Float64, length(times_e))
ttv_b = zeros(Float64, length(times_b))
compute_ttv!(5, pl_e, pl_b, times_e, times_b, ttv_e, ttv_b)
any(isnan, ttv_b) && (ttv_b[isnan.(ttv_b)] .= 0.0)
amp_sec = round(60 * 24 * 60 * (maximum(ttv_b) - minimum(ttv_b)) / 2; digits=1)
@printf("TTV-C predicted b amplitude: ±%.2f s (e_e=%.3f, e_b=%.4f, M_e=9 M⊕)\n",
        amp_sec, e_e_lit, e_b_lit)

# Group by epoch: assume transit indices fall into 3 clusters
# (the WASP-47 demo: K2 c03 idx ≲ 20, TESS s42 ~594-599, TESS s92 ~920-925).
function _group_by_epoch(idx::AbstractVector{<:Integer})
    sorted_idx = sort(unique(idx))
    breaks = Int[]
    for i in 2:length(sorted_idx)
        sorted_idx[i] - sorted_idx[i-1] > 20 && push!(breaks, i)
    end
    bounds = [1; breaks; length(sorted_idx) + 1]
    return [findall(x -> sorted_idx[bounds[k]] <= x < (k == length(bounds)-1 ?
                                                        Inf :
                                                        sorted_idx[bounds[k+1]]),
                     idx) for k in 1:length(bounds)-1]
end
groups = _group_by_epoch(observed_idx)
labels = ["K2_c03", "TESS_s42", "TESS_s92"][1:length(groups)]

plot_ttv_diagram(observed_idx, δt_med, δt_lo, δt_hi;
                  epoch_groups = groups,
                  epoch_labels = labels,
                  predicted = ttv_b,
                  predicted_label =
                    "TTVFaster b↔e (M_e=9 M⊕, ±$(amp_sec) s)",
                  filename = joinpath(OUT_DIR, "oc_diagram_ttv_compare.png"))
println("Wrote $(OUT_DIR)/oc_diagram_ttv_compare.png")
