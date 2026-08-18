# Subtract chain MAP b model from K2 + TESS, then BLS to see if d and e
# emerge as the top peaks (vs the masked-but-unsubtracted version where
# b's harmonics dominated).
#
# This is the test of Jose's hypothesis: "planet b is getting a shit fit
# and injecting shit to the BLS".

ENV["WASP47_PT_DIAG_ONLY"] = "1"
include(joinpath(@__DIR__, "wasp47_pt_only.jl"))

using Nereus
using Printf
using Statistics: median, std, mean
using Random, MCMCChains
using BoxLeastSquares: BLS
using LombScargle: lombscargle, freqpower

const Pb = 4.1591287
const Pd = 9.030672
const Pe = 0.789593

chains, _ = load_chains(joinpath(OUT_DIR, "pt_warm_chains.nc"))
np = vec(Array(chains[:, :n_planets, :]))
modal_idx = findall(np .== 1)

function build_theta_at(target, chains, idx, n_active)
    layout = target.params.layout
    td = TransDimState(max_planets = target.params.config.max_kplanet)
    for k in 1:n_active
        Nereus.activate_planet!(td, k)
    end
    th = Theta(target.params; td=td)
    for name in keys(layout.name_to_idx)
        sym = Symbol(name)
        sym in names(chains) || continue
        v = vec(Array(chains[:, sym, :]))[idx]
        isfinite(v) && (th.values[layout.name_to_idx[name]] = v)
    end
    return th
end

# Find pseudo-MAP at N_p=1
sub = modal_idx[sortperm(rand(MersenneTwister(42), length(modal_idx)))[1:min(2000, length(modal_idx))]]
best_lp = -Inf; best_idx = sub[1]
for idx in sub
    th = build_theta_at(target, chains, idx, 1)
    lp = rv_log_likelihood(th, target.data) + transit_log_likelihood(th, target.data)
    if isfinite(lp) && lp > best_lp
        global best_lp = lp; global best_idx = idx
    end
end

th_map = build_theta_at(target, chains, best_idx, 1)
@printf("Chain MAP b: P=%.6f, Tc=%.4f, b_imp=%.3f, rr=%.4f\n",
        planet_P(th_map, 1),
        th_map.values[target.params.layout.planet_blocks[1].t],
        th_map.values[target.params.layout.planet_blocks[1].b],
        th_map.values[target.params.layout.planet_blocks[1].r])

# We need a phot-only forward model for b. transit_log_likelihood
# evaluates the full likelihood; we want PREDICTIONS. There's no
# `phot_predictions` exported, but we can construct it cheaply: take
# th_map, compute residuals = data - model by abusing
# _phot_ll_no_transit (which computes the likelihood with no transits)
# vs full transit_log_likelihood. Better: do it directly.
#
# Cleanest: implement a tiny inline transit-flux predictor here, using
# the same machinery as the likelihood.

phot_pred, _ = phot_predictions(th_map, target.data)
phot_resid = target.data.flux .- phot_pred .+ 1.0  # bring residuals back to "~1.0 + dips" scale
@printf("Phot residuals: median=%.6f, std=%.6f\n",
        median(phot_resid), std(phot_resid))

# Subset: K2 only
k2_mask = target.data.phot_inst .== 1
k2_t = target.data.t_phot[k2_mask]
k2_resid = phot_resid[k2_mask]
k2_err = target.data.flux_err[k2_mask]
@printf("K2 subset: %d points\n", length(k2_t))

# BLS the K2 residuals
periods_search = collect(range(0.5, 20.0; length=20_000))
durations = [0.5/24, 1/24, 1.5/24, 2/24, 3/24, 4/24]
bls = BLS(k2_t, k2_resid, k2_err; duration=durations, periods=periods_search)

sorted = sortperm(bls.power; rev=true)
top = Float64[]
println("\nBLS top 15 peaks (K2 — chain b model SUBTRACTED):")
for i in sorted
    P = bls.periods[i]
    if all(abs(log10(P) - log10(P0)) > 0.02 for P0 in top)
        push!(top, P)
        @printf("  P = %7.4f d   power = %10.1f   dur = %.4f d\n",
                P, bls.power[i], bls.duration[i])
        length(top) >= 15 && break
    end
end

println("\nLit period BLS power (post-subtract):")
for (name, P_lit) in (("e", Pe), ("b", Pb), ("d", Pd))
    j = argmin(abs.(bls.periods .- P_lit))
    @printf("  %-3s P=%7.4f → power=%10.1f, dur=%.4f d\n",
            name, P_lit, bls.power[j], bls.duration[j])
end
