# C2: NUTS MCMC fit of HD 4747 B (HIP 3850).
#
# Brand 4747 B is a brown dwarf at P ≈ 37.5 yr, M ≈ 65 M_J. The orbital
# period is comparable to the 22-yr RV baseline, so the joint
# RV+HGCA+relastrom likelihood is well-conditioned and the posterior
# is unimodal. This is the cleanest available real-data benchmark for
# Phase 1.
#
# Reference values (Brandt+ 2021, Table 4):
#   M_companion = 65.3 ± 3.3 M_J  =  0.0623 ± 0.0032 M_sun
#   P           = 37.5 ± 1.6 yr
#   a           = 11.3 ± 0.4 AU
#   e           = 0.74 ± 0.01
#   i           = 49.5 ± 1.4 deg

using Nereus
using LogDensityProblems
using Statistics: median, std, quantile
using Printf
using Random: MersenneTwister

println("=" ^ 70)
println("C2: HD 4747 NUTS fit (Phase 1 benchmark)")
println("=" ^ 70)
println()

# ---- 1. Load data --------------------------------------------------
DATADIR = joinpath(@__DIR__, "..", "..", "..", "data", "HD4747")
HGCAFILE = joinpath(@__DIR__, "..", "..", "..", "data", "HD159062", "HGCA_vEDR3.fits")

hgca   = load_hgca_row(HGCAFILE, 3850)
rvdat  = load_orvara_rv(joinpath(DATADIR, "HD4747_RV.dat"))
relast = load_orvara_relast(joinpath(DATADIR, "HD4747_relAST.txt"))
data = Data(t_rv=rvdat.t, rv=rvdat.rv, rv_err=rvdat.rv_err,
            rv_inst=rvdat.rv_inst, relastrom=relast, hgca=hgca)

println("Loaded HIP 3850: dist = $(round(1000/hgca.plx, digits=2)) pc")
println("$(length(rvdat.t)) RV epochs ($(round((maximum(rvdat.t)-minimum(rvdat.t))/365.25, digits=1)) yr baseline)")
println("$(n_relast(relast)) relAST epochs")
println("HGCA Δpmra (Hip→Gaia) = $(round(hgca.pmra[3]-hgca.pmra[1], digits=3)) mas/yr")
println("HGCA Δpmdec (Hip→Gaia) = $(round(hgca.pmdec[3]-hgca.pmdec[1], digits=3)) mas/yr")
println()

# ---- 2. Model ------------------------------------------------------
const M_PRI_HD4747 = 0.856   # Crepp+ 2016 (Brandt 2021 ref)

params = Params(
    max_kplanet  = 1,
    planet_modes = [RVAS],
    instruments  = InstrumentConfig(rv = ["HIRES"]),
    data         = data,
    stability    = :none,
    M_s          = M_PRI_HD4747,
    parametrization = ParametrizationConfig(mass = :a_driven),
    priors = Dict{String, PriorSpec}(
        "n_p"          => FixedPrior(1.0),
        # orvara-equivalent priors: log-flat on a (AU) and M_sec (M_sun),
        # isotropic on inc. Covers Brandt+ 2021 reference (a≈11.3 AU,
        # M_sec≈0.062 M_sun = 65 M_J) with broad bounds.
        "a_k1"         => LogUniformPrior(1.0, 100.0),
        "M_sec_k1"     => LogUniformPrior(0.001, 0.5),
        "sesinw_k1"    => UniformPrior(-1.0, 1.0),
        "secosw_k1"    => UniformPrior(-1.0, 1.0),
        "Mo_k1"        => UniformPrior(0.0, 2π),
        # SinePrior = isotropic (cos i uniform), orvara/EMPEROR-II default.
        "inc_k1"       => SinePrior(),
        "Omega_k1"     => UniformPrior(0.0, 2π),
        "sigma_HIRES"  => LogUniformPrior(0.1, 30.0),
        # Fix M_pri at Crepp+ 2016 spectroscopic value (0.856 ± 0.014).
        # Sampling M_pri here pushes the chain to ~1.19 M_sun (prior
        # boundary) to compensate for missing companion-RV data
        # (Crepp+ 2018) that orvara uses but we don't ingest in
        # Phase 1. With companion-RV the K↔M_pri degeneracy is
        # broken; without it, fixing M_pri is the honest choice.
        "M_pri"        => FixedPrior(0.856),
    ),
)
println("Model: $(n_unfrozen(params)) free parameters")
println("Unfrozen: ", join(params.layout.unfrozen_names, ", "))
println()

# ---- 3. Initial guess near Brandt+ 2021 best-fit -------------------
M_sec_init = 65.3 / 1047.57    # M_J → M_sun
a_init     = 11.3              # Brandt+ 2021
e_init     = 0.74
ω_comp_init = deg2rad(0.0)     # we'll let MCMC find it; init at 0
i_init     = deg2rad(49.5)
Ω_init     = deg2rad(0.0)      # unknown; MCMC finds

x0_b = Dict{String, Float64}(
    "a_k1"         => a_init,
    "M_sec_k1"     => M_sec_init,
    "sesinw_k1"    => sqrt(e_init) * sin(ω_comp_init),
    "secosw_k1"    => sqrt(e_init) * cos(ω_comp_init),
    "Mo_k1"        => 1.0,                          # phase guess
    "inc_k1"       => i_init,
    "Omega_k1"     => deg2rad(180.0),               # rough
    "plx"          => hgca.plx,
    "gamma_HIRES"  => median(rvdat.rv),
    "sigma_HIRES"  => 2.0,
    "M_pri"        => 0.82,
)
x0 = [x0_b[nm] for nm in params.layout.unfrozen_names]
target = NereusTarget(params, data)
x0_unc = Nereus.transform_forward(x0, target.transform.type_ids,
                                    target.transform.lowers, target.transform.uppers)
ll0 = LogDensityProblems.logdensity(target, x0_unc)
println("Initial log-posterior: $(round(ll0, digits=1))")
println()

# ---- 4. Run sampler ------------------------------------------------
# NUTS gets stuck in long-period multimodal posteriors without a good
# init. Use Pigeons (parallel tempering) which is robust to bad starts
# and handles multimodality automatically.
using MCMCChains
println("Running Pigeons PT: 12 rounds × 8 chains")
t0 = time()
chains, log_Z = sample_pt(target; n_rounds = 12, n_chains = 8, seed = 42, show_report = false)
elapsed = time() - t0
println(@sprintf("Done in %.1f min, log Z = %.2f", elapsed/60, log_Z))
println()

# ---- 5. Posterior summaries ---------------------------------------
println("=" ^ 70)
println("Posterior summary (16th / 50th / 84th percentiles):")
println("=" ^ 70)

# Extract chain values per parameter
function summarize(chains, name)
    vals = vec(Array(chains[:, Symbol(name), :]))
    q16 = quantile(vals, 0.16)
    q50 = quantile(vals, 0.50)
    q84 = quantile(vals, 0.84)
    return q16, q50, q84
end

for nm in params.layout.unfrozen_names
    q16, q50, q84 = summarize(chains, nm)
    @printf("  %-15s = %12.4f  [+%.4f, -%.4f]\n", nm, q50, q84 - q50, q50 - q16)
end
println()

# Derived: M_companion, a, P_yr, e, ω, i in physical units
function derived_chain(chains, params)
    a_vals    = vec(Array(chains[:, :a_k1,       :]))
    M_sec     = vec(Array(chains[:, :M_sec_k1,   :]))
    ses_vals  = vec(Array(chains[:, :sesinw_k1,  :]))
    sec_vals  = vec(Array(chains[:, :secosw_k1,  :]))
    inc_vals  = vec(Array(chains[:, :inc_k1,     :]))
    Ω_vals    = vec(Array(chains[:, :Omega_k1,   :]))
    # M_pri may be sampled (in chain) or frozen (use config.M_s)
    n = length(a_vals)
    Mp_vals = if :M_pri in keys(chains)
        vec(Array(chains[:, :M_pri, :]))
    else
        fill(params.config.M_s, n)
    end

    e     = Vector{Float64}(undef, n)
    ω     = Vector{Float64}(undef, n)
    P_yr  = Vector{Float64}(undef, n)
    @inbounds for j in 1:n
        e[j] = ses_vals[j]^2 + sec_vals[j]^2
        ω[j] = atan(ses_vals[j], sec_vals[j])
        # P [yr] from Kepler's third law: a^3 = M_total * P^2 (AU, M_sun, yr)
        P_yr[j] = sqrt(a_vals[j]^3 / (Mp_vals[j] + M_sec[j]))
    end
    return (P_yr=P_yr, e=e, ω_deg=rad2deg.(ω),
            i_deg=rad2deg.(inc_vals), Ω_deg=rad2deg.(Ω_vals),
            M_pri=Mp_vals, M_sec=M_sec, M_J=M_sec .* 1047.57, a_au=a_vals)
end

der = derived_chain(chains, params)
println("Derived posteriors:")
for (k, v) in pairs(der)
    q16, q50, q84 = quantile(v, 0.16), quantile(v, 0.50), quantile(v, 0.84)
    @printf("  %-10s = %14.4f  [+%.4f, -%.4f]\n", string(k), q50, q84-q50, q50-q16)
end
println()
println("Brandt+ 2021 reference:")
println("  M_J      = 65.3 ± 3.3")
println("  P_yr     = 37.5 ± 1.6")
println("  a_AU     = 11.3 ± 0.4")
println("  e        = 0.74 ± 0.01")
println("  i_deg    = 49.5 ± 1.4")
println()

# Validation check
M_med = quantile(der.M_J, 0.50)
M_sig = (quantile(der.M_J, 0.84) - quantile(der.M_J, 0.16)) / 2
M_diff = abs(M_med - 65.3) / sqrt(M_sig^2 + 3.3^2)

println("=" ^ 70)
@printf("M_companion: ours = %.1f ± %.1f M_J, Brandt = 65.3 ± 3.3 M_J\n", M_med, M_sig)
@printf("Tension: %.2f σ\n", M_diff)
println(M_diff < 3.0 ? "✅ PHASE 1 VALIDATED — within 3σ of orvara" :
                       "⚠ Tension > 3σ — investigate")
