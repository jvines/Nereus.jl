#!/usr/bin/env julia
# Re-plot Kepler-88 TTV-C O−C diagram via the top-level `plot_ttv_oc`
# API. Loads the saved chain — no resampling.

using Nereus
using Printf

const REPO_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const DATADIR   = joinpath(REPO_ROOT, "data", "Kepler88")
const OUT_DIR   = joinpath(REPO_ROOT, "Nereus.jl", "results", "Kepler88_TTV_C")

# --- Rebuild the same Data + Params as the demo --------------------
const P_b_LIT  = 10.9541651
const T0_b_LIT = 2454969.20559
const P_c_LIT  = 22.3395
const T0_c_LIT = 2454974.0
const M_s_LIT  = 0.99
const R_s_LIT  = 0.880
const RR_B_LIT = 0.094
const B_B_LIT  = 0.595
const Q1_K     = 0.50
const Q2_K     = 0.30
const RHO_S_LIT = 2.05
const K_B_LIT = 2.3
const K_C_LIT = 45.0

lc = load_tess_lc(joinpath(DATADIR, "Kepler88_kplr_lc.csv"))
t_full, flux, err = lc.t, lc.flux, lc.flux_err

function transits_in(t, P, T0)
    n_lo = ceil(Int, (minimum(t) + 0.1 - T0) / P)
    n_hi = floor(Int, (maximum(t) - 0.1 - T0) / P)
    return [(n, T0 + n*P) for n in n_lo:n_hi]
end
tcs_b = transits_in(t_full, P_b_LIT, T0_b_LIT)
T0_b_FIT = tcs_b[1][2]
tcs_b = transits_in(t_full, P_b_LIT, T0_b_FIT)

mask = mask_transits(t_full, [P_b_LIT], [T0_b_FIT]; window = 0.38)
dt = detrend_savgol(t_full, flux, err;
                     window_length = 401, polyorder = 3,
                     transit_mask = mask)
flux_dt = dt.flux_detrended
keep = falses(length(t_full))
for (_, Tc) in tcs_b
    @. keep |= abs(t_full - Tc) < 0.5
end
keep .&= isfinite.(flux_dt)
keep .&= isfinite.(err)
data = Data(; t_phot = t_full[keep], flux = flux_dt[keep],
              flux_err = err[keep],
              phot_inst = ones(Int, count(keep)),
              t_rv = Float64[], rv = Float64[], rv_err = Float64[],
              rv_inst = Int[])
ic = InstrumentConfig(rv = ["dummy_rv"], pm = ["Kepler"])

priors = Dict{String, PriorSpec}()
priors["P_k1"]      = NormalPrior(P_b_LIT, 5e-5, P_b_LIT - 3e-4, P_b_LIT + 3e-4)
priors["K_k1"]      = FixedPrior(K_B_LIT)
priors["Tc_k1"]     = NormalPrior(T0_b_FIT, 0.002, T0_b_FIT - 0.02, T0_b_FIT + 0.02)
priors["sesinw_k1"] = NormalPrior(0.0, 0.05, -0.2, 0.2)
priors["secosw_k1"] = NormalPrior(0.0, 0.05, -0.2, 0.2)
priors["b_k1"]      = NormalPrior(B_B_LIT, 0.10, 0.0, 0.95)
priors["rr_k1"]     = NormalPrior(RR_B_LIT, 0.01, 0.05, 0.15)
priors["P_k2"]      = NormalPrior(P_c_LIT, 0.02, P_c_LIT - 0.5, P_c_LIT + 0.5)
priors["K_k2"]      = NormalPrior(K_C_LIT, 15.0, 5.0, 120.0)
priors["Tc_k2"]     = NormalPrior(T0_c_LIT, 0.5, T0_c_LIT - 5.0, T0_c_LIT + 5.0)
priors["sesinw_k2"] = NormalPrior(0.0, 0.05, -0.2, 0.2)
priors["secosw_k2"] = NormalPrior(0.0, 0.05, -0.2, 0.2)
priors["b_k2"]      = FixedPrior(1.9)
priors["rr_k2"]     = FixedPrior(0.001)
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

# --- Load chains and call the new top-level plot ------------------
chains, _ = load_chains(joinpath(OUT_DIR, "kepler88_ttvc_chains.nc"))
println("Loaded chains. Calling plot_ttv_oc(...)")
fig = plot_ttv_oc(chains, data, params;
                    planet_a_k = 1, planet_b_k = 2,
                    filename = joinpath(OUT_DIR, "oc_diagram_kepler88_v2.png"),
                    half_window = 0.5,
                    ngrid = 401,
                    n_envelope_draws = 200)
println("Wrote $(OUT_DIR)/oc_diagram_kepler88_v2.png")
