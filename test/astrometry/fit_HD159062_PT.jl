# C3: Pigeons PT fit of HD 159062 (long-period white-dwarf companion).
#
# C1 (MAP) failed because of multimodal posterior + degenerate jitter.
# C2 (HD 4747 with Pigeons PT) gave M_companion within 0.04σ of orvara.
# C3 applies the same Pigeons PT machinery to the harder long-period
# system. Reference: M_companion = 0.609 ± 0.024 M_sun.

using Nereus
using LogDensityProblems
using MCMCChains
using Statistics: median, std, quantile
using Printf

println("=" ^ 70)
println("C3: HD 159062 Pigeons PT fit")
println("=" ^ 70)
println()

DATADIR = joinpath(@__DIR__, "..", "..", "..", "data", "HD159062")
hgca   = load_hgca_row(joinpath(DATADIR, "HGCA_vEDR3.fits"), 85653)
rvdat  = load_orvara_rv(joinpath(DATADIR, "HD159062_RV.dat"))
relast = load_orvara_relast(joinpath(DATADIR, "HD159062_relAST.txt"))
data = Data(t_rv=rvdat.t, rv=rvdat.rv, rv_err=rvdat.rv_err,
            rv_inst=rvdat.rv_inst, relastrom=relast, hgca=hgca)
println("Loaded: $(length(rvdat.t)) RV, $(n_relast(relast)) relAST, HGCA HIP 85653")
println()

const M_PRI = 0.81

params = Params(
    max_kplanet = 1, planet_modes = [RVAS],
    instruments = InstrumentConfig(rv = ["HIRES"]),
    data = data, stability = :none, M_s = M_PRI,
    parametrization = ParametrizationConfig(mass = :a_driven),
    priors = Dict{String, PriorSpec}(
        "n_p"          => FixedPrior(1.0),
        # orvara-equivalent priors: log-flat on a, log-flat on M_sec.
        # Brandt 2021 ref: a≈207 AU, M_sec≈0.609 M_sun. Bounds wide
        # enough to permit Brandt's mode + alternatives.
        "a_k1"         => LogUniformPrior(5.0, 1500.0),
        "M_sec_k1"     => LogUniformPrior(0.05, 1.5),
        "sesinw_k1"    => UniformPrior(-1.0, 1.0),
        "secosw_k1"    => UniformPrior(-1.0, 1.0),
        "Mo_k1"        => UniformPrior(0.0, 2π),
        # SinePrior = isotropic (cos i uniform), orvara/EMPEROR-II default.
        "inc_k1"       => SinePrior(),
        "Omega_k1"     => UniformPrior(0.0, 2π),
        "sigma_HIRES"  => LogUniformPrior(1e-3, 1e3),
        # Gaussian spectroscopic prior on M_pri (Hirsch+ 2019: 0.81 ± 0.04).
        "M_pri"        => NormalPrior(0.81, 0.04, 0.5, 1.2),
    ),
)
println("Model: $(n_unfrozen(params)) free params")
println()

target = NereusTarget(params, data)

println("Running Pigeons PT: 14 rounds × 10 chains (longer than C2 — harder problem)")
t0 = time()
chains, log_Z = sample_pt(target;
                          n_rounds = 14,
                          n_chains = 10,
                          seed = 42,
                          show_report = false)
elapsed = time() - t0
println(@sprintf("Done in %.1f min, log Z = %.2f", elapsed/60, log_Z))
println()

# ---- Posterior summary ---------------------------------------------
println("=" ^ 70)
println("Posterior summary:")
println("=" ^ 70)
function summarize(chains, name)
    vals = vec(Array(chains[:, Symbol(name), :]))
    return quantile(vals, 0.16), quantile(vals, 0.50), quantile(vals, 0.84)
end
for nm in params.layout.unfrozen_names
    q16, q50, q84 = summarize(chains, nm)
    @printf("  %-15s = %14.4f  [+%.4f, -%.4f]\n", nm, q50, q84-q50, q50-q16)
end
println()

# Derived
a_v   = vec(Array(chains[:, :a_k1, :]))
M_sec = vec(Array(chains[:, :M_sec_k1, :]))
Mp_v  = vec(Array(chains[:, :M_pri, :]))
ses_v = vec(Array(chains[:, :sesinw_k1, :]))
sec_v = vec(Array(chains[:, :secosw_k1, :]))
inc_v = vec(Array(chains[:, :inc_k1, :]))
Ω_v   = vec(Array(chains[:, :Omega_k1, :]))
n = length(a_v)
e_arr = Vector{Float64}(undef, n); ω_arr = Vector{Float64}(undef, n)
P_yr  = Vector{Float64}(undef, n)
for j in 1:n
    e_arr[j] = ses_v[j]^2 + sec_v[j]^2
    ω_arr[j] = atan(ses_v[j], sec_v[j])
    # Kepler's third law: a^3 = M_total · P^2 (AU, M_sun, yr)
    P_yr[j]  = sqrt(a_v[j]^3 / (Mp_v[j] + M_sec[j]))
end

println("Derived posteriors:")
for (k, v) in pairs((P_yr=P_yr, e=e_arr, ω_deg=rad2deg.(ω_arr),
                     i_deg=rad2deg.(inc_v), Ω_deg=rad2deg.(Ω_v),
                     M_pri=Mp_v, M_sec=M_sec, a_au=a_v))
    q16, q50, q84 = quantile(v, 0.16), quantile(v, 0.50), quantile(v, 0.84)
    @printf("  %-10s = %14.4f  [+%.4f, -%.4f]\n", string(k), q50, q84-q50, q50-q16)
end
println()
println("Brandt+ 2021 reference:")
println("  P            = 2880 +1480/-1290 yr")
println("  e            = 0.51 +0.34/-0.21")
println("  i            = 50.5 +3.6/-3.7°")
println("  M_companion  = 0.609 ± 0.024 M_sun")
println("  a            = 207 +72/-58 AU")
println()

M_med = quantile(M_sec, 0.50)
M_sig = (quantile(M_sec, 0.84) - quantile(M_sec, 0.16)) / 2
M_diff = abs(M_med - 0.609) / sqrt(M_sig^2 + 0.024^2)
println("=" ^ 70)
@printf("M_companion: ours = %.3f ± %.3f M_sun, Brandt = 0.609 ± 0.024 M_sun\n", M_med, M_sig)
@printf("Tension: %.2f σ\n", M_diff)
println(M_diff < 3.0 ? "✅ PHASE 1 VALIDATED on long-period system" :
                       "⚠ Tension > 3σ — IAD+GOST may be needed for this regime")
