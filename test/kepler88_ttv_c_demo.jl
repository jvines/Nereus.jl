#!/usr/bin/env julia
# Kepler-88 joint TTV-C N-body fit. Two planets: b (transits, P=10.95d)
# and c (non-transiting perturber near 2:1 MMR, P=22.34d). c's mass is
# constrained by the ~12-hour TTV signal it induces on b's transits.
#
# This is the right architecture for this kind of system: ~10 free
# orbital params instead of 134 per-transit nuisance Tc slots. Goodman-
# Weare stretch behaves well in low D — same sampler that crashed to
# acc=4% on the TTV-A fit hits acc=20-30% here.
#
# Reference: Nesvorný+ 2013 "KOI-142, The King of Transit Variations"
#            Hadden+ 2017, Yoffe+ 2021 (mass refinements)

using Nereus
using Statistics: median, std, mean, quantile
using Printf
using Random
using TTVFaster: Planet_plane_hk, compute_ttv!

const REPO_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const DATADIR   = joinpath(REPO_ROOT, "data", "Kepler88")
const OUT_DIR   = joinpath(REPO_ROOT, "Nereus.jl", "results", "Kepler88_TTV_C")
mkpath(OUT_DIR)

println("="^70)
println("Kepler-88 joint TTV-C — N-body fit, ~10 free params")
println("="^70)

# --- Literature ephemerides (Nesvorný+ 2013, Yoffe+ 2021) -------------
const P_b_LIT  = 10.9541651
const T0_b_LIT = 2454969.20559
const P_c_LIT  = 22.3395             # near 2:1 with b
const T0_c_LIT = 2454974.0           # rough first inferior-conjunction
const M_s_LIT  = 0.99
const R_s_LIT  = 0.880
const RR_B_LIT = 0.094
const B_B_LIT  = 0.595
const Q1_K     = 0.50
const Q2_K     = 0.30
const RHO_S_LIT = 2.05

# K conversion: K[m/s] ≈ 28.4 * M_p[M_Jup] * M_s^(-2/3) * P[yr]^(-1/3)
# M_b ≈ 9 M_Earth → K_b ≈ 2.3 m/s  (FIXED, doesn't affect own TTV)
# M_c ≈ 200 M_Earth → K_c ≈ 45 m/s (FREE, gives M_c via TTV)
const K_B_LIT = 2.3
const K_C_LIT = 45.0

# --- Load Kepler LC ---------------------------------------------------
lc = load_tess_lc(joinpath(DATADIR, "Kepler88_kplr_lc.csv"))
t_full, flux, err = lc.t, lc.flux, lc.flux_err
@printf("Kepler LC: %d points, span %.1f d\n",
        length(t_full), t_full[end] - t_full[1])

# Transit centers for b (linear ephemeris reference)
function transits_in(t, P, T0)
    n_lo = ceil(Int, (minimum(t) + 0.1 - T0) / P)
    n_hi = floor(Int, (maximum(t) - 0.1 - T0) / P)
    return [(n, T0 + n*P) for n in n_lo:n_hi]
end
tcs_b = transits_in(t_full, P_b_LIT, T0_b_LIT)
T0_b_FIT = tcs_b[1][2]
tcs_b = transits_in(t_full, P_b_LIT, T0_b_FIT)
@printf("Predicted b transits: %d\n", length(tcs_b))

# --- Detrend (mask covers T14 + TTV shifts) ---------------------------
# T14 ~5.5 hr (half-width 0.12 d) + ±6 hr TTV shift (0.25 d) → 0.37 d
# half-width mask. Narrower than the 0.7 d in v1 which butchered the
# polynomial baseline and left only 1424 fittable points.
mask = mask_transits(t_full, [P_b_LIT], [T0_b_FIT]; window = 0.38)
dt = detrend_savgol(t_full, flux, err;
                     window_length = 401, polyorder = 3,
                     transit_mask = mask)
flux_dt = dt.flux_detrended

# Subset to ±0.5 d around each predicted Tc — safely covers TTV shifts.
keep = falses(length(t_full))
for (_, Tc) in tcs_b
    @. keep |= abs(t_full - Tc) < 0.5
end
keep .&= isfinite.(flux_dt)
keep .&= isfinite.(err)
t_fit    = t_full[keep]
flux_fit = flux_dt[keep]
err_fit  = err[keep]
@printf("Subset: %d points\n", length(t_fit))

data = Data(; t_phot = t_fit, flux = flux_fit, flux_err = err_fit,
              phot_inst = ones(Int, length(t_fit)),
              t_rv = Float64[], rv = Float64[], rv_err = Float64[],
              rv_inst = Int[])
ic = InstrumentConfig(rv = ["dummy_rv"], pm = ["Kepler"])

# --- Priors -----------------------------------------------------------
priors = Dict{String, PriorSpec}()

# Planet 1 = Kepler-88 b (inner, transiting). Period order: P_k1 < P_k2.
priors["P_k1"]      = NormalPrior(P_b_LIT, 5e-5,
                                   P_b_LIT - 3e-4, P_b_LIT + 3e-4)
priors["K_k1"]      = FixedPrior(K_B_LIT)
priors["Tc_k1"]     = NormalPrior(T0_b_FIT, 0.002,
                                   T0_b_FIT - 0.02, T0_b_FIT + 0.02)
priors["sesinw_k1"] = NormalPrior(0.0, 0.05, -0.2, 0.2)
priors["secosw_k1"] = NormalPrior(0.0, 0.05, -0.2, 0.2)
priors["b_k1"]      = NormalPrior(B_B_LIT, 0.10, 0.0, 0.95)
priors["rr_k1"]     = NormalPrior(RR_B_LIT, 0.01, 0.05, 0.15)

# Planet 2 = Kepler-88 c (outer, NON-TRANSITING). Forced non-transit
# via b_k2=10 + tiny rr — keeps it in the N-body state but contributes
# no flux. K_k2 is the mass parameter (constrained by TTV on b).
priors["P_k2"]      = NormalPrior(P_c_LIT, 0.02, P_c_LIT - 0.5, P_c_LIT + 0.5)
priors["K_k2"]      = NormalPrior(K_C_LIT, 15.0, 5.0, 120.0)
priors["Tc_k2"]     = NormalPrior(T0_c_LIT, 0.5, T0_c_LIT - 5.0, T0_c_LIT + 5.0)
# Eccentricities tighter — TTVFaster 1st-order series breaks for e>0.1
# and high-e modes were absorbing mass excess in v1.
priors["sesinw_k2"] = NormalPrior(0.0, 0.05, -0.2, 0.2)
priors["secosw_k2"] = NormalPrior(0.0, 0.05, -0.2, 0.2)
priors["b_k2"]      = FixedPrior(1.9)      # >1+rr → transit gate off; max validated 2.0
priors["rr_k2"]     = FixedPrior(0.001)    # cosmetic; no flux contribution

priors["rho_s"]         = NormalPrior(RHO_S_LIT, 0.3, 0.5, 5.0)
priors["offset_Kepler"] = NormalPrior(0.0, 5e-4, -3e-3, 3e-3)
priors["jitter_Kepler"] = LogUniformPrior(1e-5, 5e-3)
priors["q1_Kepler"]     = NormalPrior(Q1_K, 0.15, 0.0, 1.0)
priors["q2_Kepler"]     = NormalPrior(Q2_K, 0.15, 0.0, 1.0)
priors["gamma_dummy_rv"] = FixedPrior(0.0)
priors["sigma_dummy_rv"] = FixedPrior(1.0)

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

# --- Sample (low-D so stretch ensemble behaves) ------------------------
const SEED      = parse(Int, get(ENV, "KEPLER88_TTVC_SEED",    "42"))
const N_STEPS   = parse(Int, get(ENV, "KEPLER88_TTVC_STEPS",   "2000"))
const N_WALKERS = parse(Int, get(ENV, "KEPLER88_TTVC_WALKERS", "60"))
const N_TEMPS   = parse(Int, get(ENV, "KEPLER88_TTVC_TEMPS",   "4"))
const N_BURNIN  = parse(Int, get(ENV, "KEPLER88_TTVC_BURNIN",  "500"))

println("\nsample_ptemcee: $N_TEMPS temps × $N_WALKERS walkers × $N_STEPS steps")
t0 = time()
result = sample_ptemcee(target, data;
    n_temps       = N_TEMPS,
    n_walkers     = N_WALKERS,
    n_steps       = N_STEPS,
    n_burnin      = N_BURNIN,
    seed          = SEED,
    show_progress = true,
)
elapsed = time() - t0
@printf("Done in %.1f s  (%.0f evals/sec)\n",
        elapsed, result.n_evals / elapsed)

chains = result.chains
save_chains(joinpath(OUT_DIR, "kepler88_ttvc_chains.nc"), chains, params; data = data)

# --- Report M_c posterior (the science output) ------------------------
const M_SUN = 1.989e33
const M_E   = 5.972e27
function _kc_to_mc(K_c, sesinw, secosw, b_c_dummy)
    e = hypot(sesinw, secosw)^2
    e = clamp(e, 0.0, 0.99)
    # For non-transiting c: assume inclination ≈ b's inclination (coplanar)
    mp_sini = msini(M_s_LIT, K_c, P_c_LIT, e)
    return mp_sini * 1.898e30 / M_E
end
kc_chain  = vec(Array(chains[:, :K_k2, :]))
se_chain  = vec(Array(chains[:, :sesinw_k2, :]))
sc_chain  = vec(Array(chains[:, :secosw_k2, :]))
mc_chain  = _kc_to_mc.(kc_chain, se_chain, sc_chain, 0.0)
@printf("\nKepler-88 c mass posterior (M_⊕):\n")
@printf("  median = %.1f   [16, 84] = [%.1f, %.1f]\n",
        median(mc_chain), quantile(mc_chain, 0.16), quantile(mc_chain, 0.84))
@printf("  Nesvorný+ 2013: ~190 M_⊕\n")
@printf("  Yoffe+ 2021:    214 ± 6 M_⊕\n")

# --- Publication-grade O−C plot via top-level API --------------------
# One call: extracts per-transit Tc (data) from phot LL profiles AND
# the TTVFaster N-body envelope from posterior draws. Nereus theme +
# colors applied automatically. No inline plotting code below.
fig = plot_ttv_oc(chains, data, params;
                    planet_a_k = 1, planet_b_k = 2,
                    filename = joinpath(OUT_DIR, "oc_diagram_kepler88_ttvc.png"),
                    half_window = 0.5,
                    n_envelope_draws = 200)
println("\nWrote $(OUT_DIR)/oc_diagram_kepler88_ttvc.png")

# Peak-to-peak from the median N-body envelope (via top-level API)
all_tcs_b = [tcs_b[i][2] for i in 1:length(tcs_b)]
env = ttvc_envelope(chains, params;
                      tcs_observed = all_tcs_b,
                      planet_a_k = 1, planet_b_k = 2,
                      n_draws = 200)
p2p_ttvc = (maximum(env.med) - minimum(env.med)) * 24 * 60
@printf("\nTTV-C envelope peak-to-peak: %.1f min\n", p2p_ttvc)
@printf("Nesvorný+ 2013 reports ~720 min (~12 hr).\n")

println("\n" * "="^70)
println("Demo complete.")
println("="^70)
