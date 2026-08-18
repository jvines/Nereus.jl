# C1: Quick MAP fit of HD 159062 using Optim.LBFGS via Nereus.sample_map.
#
# Goal: get a single point estimate of M_companion. If MAP gives a value
# in [0.5, 0.7] M_sun (orvara's posterior 1σ range is 0.609 ± 0.024),
# Phase 1 is qualitatively validated.
#
# Long-period system → broad (P, e) priors will likely yield local optima.
# We initialize near orvara's published values so LBFGS finds the right
# basin without a global search.

using Nereus
using LogDensityProblems
using Statistics: median
using Printf

println("=" ^ 70)
println("C1: HD 159062 MAP fit (Phase 1 validation)")
println("=" ^ 70)
println()

# ---- 1. Load data --------------------------------------------------
DATADIR = joinpath(@__DIR__, "..", "..", "..", "data", "HD159062")
hgca   = load_hgca_row(joinpath(DATADIR, "HGCA_vEDR3.fits"), 85653)
rvdat  = load_orvara_rv(joinpath(DATADIR, "HD159062_RV.dat"))
relast = load_orvara_relast(joinpath(DATADIR, "HD159062_relAST.txt"))
data = Data(t_rv=rvdat.t, rv=rvdat.rv, rv_err=rvdat.rv_err,
            rv_inst=rvdat.rv_inst, relastrom=relast, hgca=hgca)

println("Loaded: $(length(rvdat.t)) RV epochs, $(n_relast(relast)) relAST epochs, HGCA HIP 85653")
println()

# ---- 2. Custom priors for HD 159062 --------------------------------
# Long-period white-dwarf companion: orvara reports
#   P ~ 2880 +1480/-1290 yr  → LogUniform(800, 8000) yr to bracket
#   K ~ 900 m/s for M_sec ~ 0.6 M_sun (derived) → LogUniform(50, 5000)
#   e ~ 0.51 ± 0.27         → Uniform(0, 0.95)
#   i ~ 50.5° ± 3.6°        → Uniform(0, π)
#   Ω ~ 90.4° ± 1.7°        → Uniform(0, 2π)
#   plx tight from Gaia     → handled by default_priors via hgca

# Build with defaults then override.
params = Params(
    max_kplanet = 1,
    planet_modes = [RVAS],
    instruments  = InstrumentConfig(rv = ["HIRES"]),
    data         = data,
    stability    = :none,
    M_s          = 0.81,                # Hirsch+ 2019 spectroscopic
    priors = Dict{String, PriorSpec}(
        # Period: 800-8000 yr in days
        "P_k1"        => LogUniformPrior(800.0 * 365.25, 8000.0 * 365.25),
        # K: planet's mass × sin(i) sets this; broad
        "K_k1"        => LogUniformPrior(10.0, 5000.0),
        "sesinw_k1"   => UniformPrior(-1.0, 1.0),
        "secosw_k1"   => UniformPrior(-1.0, 1.0),
        "Mo_k1"       => UniformPrior(0.0, 2π),
        "inc_k1"      => UniformPrior(0.01, π - 0.01),
        "Omega_k1"    => UniformPrior(0.0, 2π),
        "n_p"         => FixedPrior(1.0),
        # Tighten HIRES jitter to physical range (Hirsch+ 2019: σ_HIRES ~1.5 m/s nominal)
        "sigma_HIRES" => LogUniformPrior(0.1, 20.0),
        # plx and gamma defaults
    ),
)
println("Model: $(n_unfrozen(params)) free parameters")
println("Unfrozen: ", join(params.layout.unfrozen_names, ", "))
println()

# ---- 3. Initial guess near orvara published values -----------------
# Tp computed via inverse: at tp, M_anom = 0. Brandt+ 2021 doesn't
# publish Tp; we set Mo such that the orbit reaches periastron near
# the middle of the data baseline.
function init_x0_HD159062(params, data, hgca)
    M_pri = 0.81; M_sec_init = 0.609
    P_init = 2880.0 * 365.25
    e_init = 0.51
    ω_comp_init = deg2rad(246.0)   # companion convention (PlanetOrbits)
    i_init = deg2rad(50.5)
    Ω_init = deg2rad(90.4)

    # Derive K consistent with M_sec, M_pri, P, e, sin(i)
    # K = (f_M / (P_d (1-e²)^(3/2) F))^(1/3)
    F = 86400.0 / (2π * 6.6743e-11 * 1.989e30)
    f_M = M_sec_init^3 * sin(i_init)^3 / (M_pri + M_sec_init)^2
    K_init = cbrt(f_M / (P_init * (1-e_init^2)^(3/2) * F))

    # sesinw = sqrt(e) sin ω, secosw = sqrt(e) cos ω
    sqe = sqrt(e_init)
    sesinw_init = sqe * sin(ω_comp_init)
    secosw_init = sqe * cos(ω_comp_init)

    # Mo: reasonable guess (sweep would be needed for real fit; set 0)
    Mo_init = 0.0

    # plx near Gaia
    plx_init = hgca.plx
    # gamma + sigma at HIRES: median(rv) and 5 m/s
    γ_init = median(data.rv)
    σ_init = 5.0

    # Order MUST match params.layout.unfrozen_names
    name_to_value = Dict{String, Float64}(
        "P_k1"         => P_init,
        "K_k1"         => K_init,
        "sesinw_k1"    => sesinw_init,
        "secosw_k1"    => secosw_init,
        "Mo_k1"        => Mo_init,
        "inc_k1"       => i_init,
        "Omega_k1"     => Ω_init,
        "plx"          => plx_init,
        "gamma_HIRES"  => γ_init,
        "sigma_HIRES"  => σ_init,
    )

    x0 = Float64[]
    for nm in params.layout.unfrozen_names
        haskey(name_to_value, nm) || error("missing init for $nm")
        push!(x0, name_to_value[nm])
    end
    return x0, K_init
end

x0_bounded, K_init = init_x0_HD159062(params, data, hgca)
println("Init (bounded space):")
for (nm, v) in zip(params.layout.unfrozen_names, x0_bounded)
    println(@sprintf("  %-15s = %12.4f", nm, v))
end
println(@sprintf("\nDerived K from orvara M_sec=0.609 → K = %.2f m/s", K_init))
println()

# ---- 4. Build target and run MAP -----------------------------------
target = NereusTarget(params, data)

# Initial guess in unconstrained space (LBFGS works there)
x0_unc = Nereus.transform_forward(x0_bounded,
                                    target.transform.type_ids,
                                    target.transform.lowers,
                                    target.transform.uppers)
ll_init = LogDensityProblems.logdensity(target, x0_unc)
println(@sprintf("Initial log-posterior at orvara-init: %.2f", ll_init))
println()

println("Running LBFGS (max 5000 iters)...")
t_start = time()
result = sample_map(target;
                    init = x0_unc,
                    method = :LBFGS,
                    maxiter = 5000,
                    g_tol = 1e-6)
elapsed = time() - t_start

println(@sprintf("Done in %.1f s, log-posterior at MAP = %.2f, converged=%s",
                 elapsed, result.log_posterior, result.converged))
println()

# Map MAP from unconstrained → bounded
x_map_bounded = Nereus.transform_inverse(result.x_map,
                                            target.transform.type_ids,
                                            target.transform.lowers,
                                            target.transform.uppers)

# ---- 5. Decode MAP to physical parameters --------------------------
println("MAP estimate (bounded / physical space):")
for (nm, v) in zip(params.layout.unfrozen_names, x_map_bounded)
    println(@sprintf("  %-15s = %14.6f", nm, v))
end
println()

# Derive M_companion and a from MAP
P_map_d = x_map_bounded[findfirst(==("P_k1"), params.layout.unfrozen_names)]
K_map   = x_map_bounded[findfirst(==("K_k1"), params.layout.unfrozen_names)]
ses_map = x_map_bounded[findfirst(==("sesinw_k1"), params.layout.unfrozen_names)]
sec_map = x_map_bounded[findfirst(==("secosw_k1"), params.layout.unfrozen_names)]
inc_map = x_map_bounded[findfirst(==("inc_k1"), params.layout.unfrozen_names)]

e_map = ses_map^2 + sec_map^2
ω_map = atan(ses_map, sec_map)
sin_i_map = sin(inc_map)
M_sec_map = msec_from_K(K_map, P_map_d, e_map, sin_i_map, 0.81)
a_map = a_from_P(P_map_d, 0.81 + M_sec_map)

println("=" ^ 70)
println("Derived from MAP:")
println(@sprintf("  P            = %.1f yr", P_map_d / 365.25))
println(@sprintf("  e            = %.3f", e_map))
println(@sprintf("  ω (companion) = %.1f°", rad2deg(ω_map)))
println(@sprintf("  i            = %.1f°", rad2deg(inc_map)))
println(@sprintf("  K            = %.2f m/s", K_map))
println(@sprintf("  M_companion  = %.4f M_sun = %.2f M_J", M_sec_map, M_sec_map * 1047.57))
println(@sprintf("  a            = %.1f AU", a_map))
println()
println("Brandt+ 2021 (1σ marginal medians):")
println("  P            = 2880 +1480/-1290 yr")
println("  e            = 0.51 +0.34/-0.21")
println("  ω (companion) = 246 +44/-114°")
println("  i            = 50.5 +3.6/-3.7°")
println("  M_companion  = 0.609 ± 0.024 M_sun")
println("  a            = 207 +72/-58 AU")
println()
println("=" ^ 70)
m_in_range = 0.5 < M_sec_map < 0.7
if m_in_range
    println("✅ M_companion in [0.5, 0.7] M_sun — PHASE 1 VALIDATED")
else
    println(@sprintf("⚠ M_companion = %.3f outside [0.5, 0.7] — investigate", M_sec_map))
end
