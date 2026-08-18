# Diagnose HD 159062 B: does our likelihood actually have a high-log_post
# mode at Brandt's published P=2880 yr, or does the data prefer short P
# in our model?
#
# Strategy: pin all orbit parameters at Brandt+ 2021 medians except P
# and the linear nuisances (γ, σ_jit, Tp). Sweep log P, optimize the
# nuisances at each P, plot log_posterior(P). If there's a peak near
# P=2880 yr with comparable height to our found mode at P=300 yr, the
# disagreement is sampler-induced (mode trapping). If the long-P
# region is uniformly low log_post, our likelihood disagrees with
# orvara's at the level that would explain the discrepancy.

using Nereus, LogDensityProblems, Optim, Printf
using Statistics: median

DATADIR = joinpath(@__DIR__, "..", "..", "..", "data", "HD159062")
hgca   = load_hgca_row(joinpath(DATADIR, "HGCA_vEDR3.fits"), 85653)
rvdat  = load_orvara_rv(joinpath(DATADIR, "HD159062_RV.dat"))
relast = load_orvara_relast(joinpath(DATADIR, "HD159062_relAST.txt"))
data = Data(t_rv=rvdat.t, rv=rvdat.rv, rv_err=rvdat.rv_err,
            rv_inst=rvdat.rv_inst, relastrom=relast, hgca=hgca)

# Brandt+ 2021 medians (pinned)
const M_PRI    = 0.81
const M_SEC    = 0.6083
const E        = 0.51
const I_RAD    = deg2rad(50.5)
const Ω_RAD    = deg2rad(90.4)
const ω_C_RAD  = deg2rad(246.0)   # companion convention (PlanetOrbits)

params = Params(
    max_kplanet  = 1, planet_modes = [RVAS],
    instruments  = InstrumentConfig(rv = ["HIRES"]),
    data         = data, stability = :none, M_s = M_PRI,
    parametrization = ParametrizationConfig(mass = :M_sec_driven),
    priors = Dict{String, PriorSpec}(
        "n_p"          => FixedPrior(1.0),
        "P_k1"         => LogUniformPrior(50.0 * 365.25, 30000.0 * 365.25),
        "M_sec_k1"     => FixedPrior(M_SEC),     # pin
        "sesinw_k1"    => FixedPrior(sqrt(E) * sin(ω_C_RAD)),
        "secosw_k1"    => FixedPrior(sqrt(E) * cos(ω_C_RAD)),
        "Mo_k1"        => UniformPrior(0.0, 2π),    # free (phase)
        "inc_k1"       => FixedPrior(I_RAD),
        "Omega_k1"     => FixedPrior(Ω_RAD),
        "sigma_HIRES"  => LogUniformPrior(1e-3, 1e3),
        "M_pri"        => FixedPrior(M_PRI),
    ),
)
target = NereusTarget(params, data)
unfrozen = params.layout.unfrozen_names
println("Unfrozen at Brandt-mode: ", join(unfrozen, ", "))
n_dim = length(unfrozen)
P_idx = findfirst(==("P_k1"), unfrozen)

# Quick profile: at each P, optimize over (Mo, gamma, sigma) via a
# multistart NelderMead.
function neg_lp(x_unc)
    lp = LogDensityProblems.logdensity(target, x_unc)
    return isfinite(lp) ? -lp : 1e10
end

function profile_at_P(P_yr)
    P_d = P_yr * 365.25
    # Build x0 in bounded space
    Pmin = target.transform.lowers[P_idx]
    Pmax = target.transform.uppers[P_idx]
    z = (log(P_d) - log(Pmin)) / (log(Pmax) - log(Pmin))
    z = clamp(z, 1e-9, 1-1e-9)

    best_lp = -Inf
    for restart in 1:8
        # Random init for nuisances
        x_unc = randn(n_dim) .* 0.5
        # Pin P
        x_unc[P_idx] = log(z / (1 - z)) * 0.5  # approximate logit
        function obj(y)
            xx = copy(x_unc)
            for (k, i) in enumerate(setdiff(1:n_dim, [P_idx]))
                xx[i] = y[k]
            end
            return neg_lp(xx)
        end
        y0 = x_unc[setdiff(1:n_dim, [P_idx])]
        res = Optim.optimize(obj, y0, NelderMead(),
                              Optim.Options(iterations=1000, g_tol=1e-3))
        lp = -Optim.minimum(res)
        best_lp = max(best_lp, lp)
    end
    return best_lp
end

println()
println("Profile log_posterior(P) with all orbit params at Brandt+ 2021 medians:")
println(" P (yr)    log_post   ")
P_grid = [100, 150, 200, 300, 500, 800, 1200, 1800, 2500, 2880, 3500, 5000, 8000, 15000]
results = Tuple{Float64, Float64}[]
for P in P_grid
    lp = profile_at_P(Float64(P))
    push!(results, (Float64(P), lp))
    @printf("  %7d   %10.2f\n", P, lp)
end

println()
println("Brandt's P = 2880 yr — does our likelihood peak there?")
imax = argmax(last.(results))
@printf("Peak: P = %.0f yr,  log_post = %.2f\n", results[imax][1], results[imax][2])
@printf("At P = 2880 yr,         log_post = %.2f\n",
        results[findfirst(r -> r[1] == 2880, results)][2])
