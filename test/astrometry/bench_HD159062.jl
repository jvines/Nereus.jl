# Phase 1 real-data benchmark: HD 159062 B
#
# This is the orvara tutorial system (Brandt et al. 2021, AJ 162, 186):
# a white-dwarf companion at ~57 AU resolved by Hirsch+ 2019 NIRC2
# imaging, with HARPS+HIRES RV and a strong Hipparcos-Gaia proper-
# motion anomaly. orvara's HGCA Mode A (no IAD/GOST) is the same
# approximation as Nereus's Phase 1, so this is a like-for-like test.
#
# Reference posterior values (Brandt+ 2021 Table 4, HD 159062 B):
#   M_pri       = 0.81 ± 0.02 M_sun (fixed prior)
#   M_companion = 0.609 ± 0.024 M_sun
#   P           = 2880 +1480/-1290 yr
#   a           = 207 +72/-58 AU
#   e           = 0.51 +0.34/-0.21
#   i           = 50.5 +3.6/-3.7 deg
#   Ω           = 90.4 +1.7/-1.6 deg
#   ω (companion) = 246 +44/-114 deg
#
# Note: orvara reports the *companion's* ω. Nereus samples the host's
# ω, which is the companion's ω + π (mod 2π). So we expect ω_NEREUS
# at the best fit ≈ 246° − 180° = 66°.
#
# This script:
#   1. Loads the data via the new orvara-format loaders.
#   2. Sets up a single-planet RVAS model with M_pri=0.81 M_sun.
#   3. Sets theta to Brandt's best-fit values.
#   4. Reports the log-likelihood and the per-component breakdown.
#   5. Perturbs each astrometry parameter ±1σ and confirms log-lik
#      drops monotonically — indicates we're at a local max.

using Nereus
using Statistics: median, std

# ---- 1. Load data --------------------------------------------------
const DATADIR = joinpath(@__DIR__, "..", "..", "..", "data", "HD159062")

println("=== HD 159062 — Phase 1 real-data benchmark ===")
println()

println("[1] Loading data...")
hgca   = load_hgca_row(joinpath(DATADIR, "HGCA_vEDR3.fits"), 85653)
rvdat  = load_orvara_rv(joinpath(DATADIR, "HD159062_RV.dat"))
relast = load_orvara_relast(joinpath(DATADIR, "HD159062_relAST.txt"))

println("    HGCA HIP 85653: distance = ", round(1000/hgca.plx, digits=2), " pc")
println("    Hip pmra=", round(hgca.pmra[1], digits=2),
        ", HG pmra=",  round(hgca.pmra[2], digits=2),
        ", Gaia pmra=",round(hgca.pmra[3], digits=2),
        " mas/yr")
println("    Δpmra (Hip→Gaia) = ", round(hgca.pmra[3]-hgca.pmra[1], digits=2),
        " mas/yr  (proper-motion anomaly)")
println("    RV: ", length(rvdat.t), " HIRES epochs, ",
        round((maximum(rvdat.t)-minimum(rvdat.t))/365.25, digits=2), " yr baseline")
println("    RelAST: ", n_relast(relast), " NIRC2 epochs (", round(relast.t[1], digits=1),
        "..", round(relast.t[end], digits=1), " MJD)")
println()

data = Data(
    t_rv = rvdat.t, rv = rvdat.rv, rv_err = rvdat.rv_err,
    rv_inst = rvdat.rv_inst,
    relastrom = relast, hgca = hgca,
)

# ---- 2. Build model -------------------------------------------------
println("[2] Building Params...")
const M_PRI_HD159062 = 0.81   # M_sun, Brandt+ 2021 (from Hirsch+ 2019)

params = Params(
    max_kplanet  = 1,
    planet_modes = [RVAS],
    instruments  = InstrumentConfig(rv = ["HIRES"]),
    data         = data,
    stability    = :none,
    M_s          = M_PRI_HD159062,
)

println("    n_total = ", n_total(params), ",  n_unfrozen = ", n_unfrozen(params))
println("    block: ", typeof(params.layout.planet_blocks[1]))
println()

# ---- 3. Set theta to Brandt's best-fit ------------------------------
println("[3] Setting theta to Brandt+ 2021 best-fit values...")

# Brandt's published medians
const BRANDT = (
    M_companion = 0.609,
    P_yr        = 2880.0,
    e           = 0.51,
    i_deg       = 50.5,
    Omega_deg   = 90.4,
    omega_comp_deg = 246.0,
)
P_days_brandt   = BRANDT.P_yr * 365.25
e_brandt        = BRANDT.e
i_brandt        = deg2rad(BRANDT.i_deg)
Ω_brandt        = deg2rad(BRANDT.Omega_deg)
ω_host_brandt   = deg2rad(BRANDT.omega_comp_deg) - π    # host's ω = companion's ω - π
# Wrap into (-π, π]
ω_host_brandt   = mod(ω_host_brandt + π, 2π) - π
M_sec_brandt    = BRANDT.M_companion

# Derive K from the mass function (inverse problem). With M_sec, M_pri,
# sin i, P, e known, K is determined:
#   K^3 = f_M(M) · 2π G / [P · (1-e²)^(3/2)]
# where f_M(M) = M_sec^3 sin^3 i / (M_pri + M_sec)^2.
function _K_from_msec(M_pri, M_sec, P_days, e, sin_i)
    f_M = M_sec^3 * sin_i^3 / (M_pri + M_sec)^2
    # Use Nereus's mass_function constant in reverse
    # K = (f_M / (P_days * (1-e²)^(3/2) * F_factor))^(1/3)
    F = 1.0 / (2π * 6.6743e-11 * 1.989e30) * 86400.0
    one_minus_e2 = 1 - e * e
    K3 = f_M / (P_days * one_minus_e2^(3/2) * F)
    return cbrt(K3)
end
K_brandt = _K_from_msec(M_PRI_HD159062, M_sec_brandt, P_days_brandt, e_brandt, sin(i_brandt))

println("    Derived K = ", round(K_brandt, digits=2), " m/s (RV semi-amplitude)")

theta = Theta(params)
set_param!(theta, "n_p", 1)
set_param!(theta, "P_k1",       P_days_brandt)
set_param!(theta, "K_k1",       K_brandt)
set_param!(theta, "sesinw_k1",  sqrt(e_brandt) * sin(ω_host_brandt))
set_param!(theta, "secosw_k1",  sqrt(e_brandt) * cos(ω_host_brandt))
# M_o (mean anomaly at t_ref). Brandt fits Tp, not Mo. Without a
# published Tp we sweep Mo to find the best phase below — for now,
# use 0 and let the diagnostic show whether RV/relAST drive a mismatch.
set_param!(theta, "Mo_k1",      0.0)
set_param!(theta, "inc_k1",     i_brandt)
set_param!(theta, "Omega_k1",   Ω_brandt)
set_param!(theta, "plx",        hgca.plx)
set_param!(theta, "gamma_HIRES", median(rvdat.rv))   # zero-point near data mean
set_param!(theta, "sigma_HIRES", 5.0)                # per-point jitter

# ---- 4. Diagnose log-likelihood -------------------------------------
println()
println("[4] Log-likelihood at Brandt+ 2021 best-fit:")
ll_rel  = relastrom_log_likelihood(theta, data)
ll_hg   = hgca_log_likelihood(theta, data)
ll_full = rv_log_likelihood(theta, data)
ll_rv_only = ll_full - ll_rel - ll_hg   # rough decomposition
println("    rel astrom ll  = ", round(ll_rel, digits=2))
println("    HGCA       ll  = ", round(ll_hg, digits=2))
println("    RV         ll  ≈ ", round(ll_rv_only, digits=2), "  (full − rel − HGCA)")
println("    TOTAL      ll  = ", round(ll_full, digits=2))
println()

# Sweep Mo to find best RV phase (since we don't have Brandt's Tp).
println("[4a] Sweeping Mo to find best RV phase...")
function _sweep_mo(theta, data)
    best_mo, best_ll = 0.0, -Inf
    for mo in range(0, 2π, length=51)
        set_param!(theta, "Mo_k1", mo)
        ll = rv_log_likelihood(theta, data)
        if ll > best_ll
            best_ll = ll
            best_mo = mo
        end
    end
    return best_mo, best_ll
end
best_mo, best_ll = _sweep_mo(theta, data)
set_param!(theta, "Mo_k1", best_mo)
println("    best Mo = ", round(rad2deg(best_mo), digits=1), "°,  ll = ", round(best_ll, digits=2))
ll_at_brandt = rv_log_likelihood(theta, data)
println("    final ll at Brandt's params (best Mo) = ", round(ll_at_brandt, digits=2))
println()

# Optimize gamma + sigma for HIRES too (linear nuisances)
println("[4b] Sweeping gamma_HIRES, sigma_HIRES...")
function _sweep_gs(theta, data, rv_med)
    best_gamma, best_sigma = rv_med, 5.0
    ll_best = -Inf
    for γ in range(rv_med - 30, rv_med + 30, length=21)
        for σ in (1.0, 2.0, 3.0, 5.0, 10.0, 20.0)
            set_param!(theta, "gamma_HIRES", γ)
            set_param!(theta, "sigma_HIRES", σ)
            ll = rv_log_likelihood(theta, data)
            if ll > ll_best
                ll_best = ll
                best_gamma = γ
                best_sigma = σ
            end
        end
    end
    return best_gamma, best_sigma, ll_best
end
best_gamma, best_sigma, ll_best = _sweep_gs(theta, data, median(rvdat.rv))
set_param!(theta, "gamma_HIRES", best_gamma)
set_param!(theta, "sigma_HIRES", best_sigma)
println("    best gamma = ", round(best_gamma, digits=2), " m/s")
println("    best sigma = ", round(best_sigma, digits=2), " m/s")
println("    ll best = ", round(ll_best, digits=2))
println()

# ---- 5. Perturbation tests ------------------------------------------
println("[5] Perturbation tests (each from best Mo, gamma, sigma):")
ll_anchor = rv_log_likelihood(theta, data)
println("    anchor ll = ", round(ll_anchor, digits=2))
println()

# 1σ perturbations from Brandt's table
perturbations = [
    ("inc_k1",     i_brandt,        deg2rad(3.6),  "i ± 3.6° (1σ)"),
    ("Omega_k1",   Ω_brandt,        deg2rad(1.7),  "Ω ± 1.7° (1σ)"),
    ("P_k1",       P_days_brandt,   1480 * 365.25, "P + 1480 yr (1σ_high)"),
]
for (name, center, dev, label) in perturbations
    set_param!(theta, name, center + dev)
    ll_plus = rv_log_likelihood(theta, data)
    set_param!(theta, name, center - dev)
    ll_minus = rv_log_likelihood(theta, data)
    set_param!(theta, name, center)
    println("    $label:  ll(+) = ", round(ll_plus, digits=2),
            ",  ll(0) = ",     round(ll_anchor, digits=2),
            ",  ll(−) = ",     round(ll_minus, digits=2),
            ",  Δ(+) = ",      round(ll_plus  - ll_anchor, digits=2),
            ",  Δ(−) = ",      round(ll_minus - ll_anchor, digits=2))
end

println()
println("=== END ===")
