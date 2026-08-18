# Diagnose: where are the modes of HD 159062's posterior?
#
# C3 converged to P=807 yr (right at LogUniform lower bound). orvara
# reports P=2880 yr. Are these two MODES of the same posterior, or did
# our chain just sit at the prior wall?
#
# Strategy: profile the log-likelihood along log P, marginalizing other
# orbital parameters via a quick local optimization at each P. If we
# see two peaks, there are two modes; if one peak (and our chain found
# it), Brandt's 2880-yr value is wrong/spurious.

using Nereus
using LogDensityProblems
using Optim
using Statistics: median
using Printf

DATADIR = joinpath(@__DIR__, "..", "..", "..", "data", "HD159062")
hgca   = load_hgca_row(joinpath(DATADIR, "HGCA_vEDR3.fits"), 85653)
rvdat  = load_orvara_rv(joinpath(DATADIR, "HD159062_RV.dat"))
relast = load_orvara_relast(joinpath(DATADIR, "HD159062_relAST.txt"))
data = Data(t_rv=rvdat.t, rv=rvdat.rv, rv_err=rvdat.rv_err,
            rv_inst=rvdat.rv_inst, relastrom=relast, hgca=hgca)

const M_PRI = 0.81

# Wide P prior so we can probe everywhere from 50 yr to 30000 yr.
params = Params(
    max_kplanet = 1, planet_modes = [RVAS],
    instruments = InstrumentConfig(rv = ["HIRES"]),
    data = data, stability = :none, M_s = M_PRI,
    priors = Dict{String, PriorSpec}(
        "n_p"          => FixedPrior(1.0),
        "P_k1"         => LogUniformPrior(50.0 * 365.25, 30000.0 * 365.25),
        "K_k1"         => LogUniformPrior(50.0, 5000.0),
        "sesinw_k1"    => UniformPrior(-1.0, 1.0),
        "secosw_k1"    => UniformPrior(-1.0, 1.0),
        "Mo_k1"        => UniformPrior(0.0, 2π),
        "inc_k1"       => UniformPrior(0.01, π - 0.01),
        "Omega_k1"     => UniformPrior(0.0, 2π),
        "sigma_HIRES"  => LogUniformPrior(0.1, 20.0),
    ),
)
target = NereusTarget(params, data)
unfrozen = params.layout.unfrozen_names

# Index mapping
idx = Dict(nm => findfirst(==(nm), unfrozen) for nm in unfrozen)
P_idx = idx["P_k1"]

# Profile likelihood: at each P, optimize over all OTHER unfrozen params.
function profile_at_P(P_days; init_x_unc=nothing)
    n = length(unfrozen)
    # Default x0 in bounded space
    if init_x_unc === nothing
        x0_b = Dict(
            "P_k1"        => P_days,
            "K_k1"        => 1000.0,
            "sesinw_k1"   => 0.3,
            "secosw_k1"   => 0.3,
            "Mo_k1"       => π,
            "inc_k1"      => π/3,
            "Omega_k1"    => π,
            "plx"         => hgca.plx,
            "gamma_HIRES" => median(rvdat.rv),
            "sigma_HIRES" => 2.0,
        )
        x0 = [x0_b[nm] for nm in unfrozen]
        x_unc = Nereus.transform_forward(x0, target.transform.type_ids,
                                           target.transform.lowers, target.transform.uppers)
    else
        x_unc = copy(init_x_unc)
        # Override P slot
        # Need P in unconstrained: logit on [Pmin, Pmax] with bounds
        Pmin = target.transform.lowers[P_idx]
        Pmax = target.transform.uppers[P_idx]
        z = (P_days - Pmin) / (Pmax - Pmin)
        z = clamp(z, 1e-9, 1-1e-9)
        x_unc[P_idx] = log(z / (1-z))
    end

    # Free indices = all except P_idx
    free_idx = [i for i in 1:n if i != P_idx]
    function neg_lp(y_free)
        x_full = copy(x_unc)
        @inbounds for (k, i) in enumerate(free_idx)
            x_full[i] = y_free[k]
        end
        lp = LogDensityProblems.logdensity(target, x_full)
        return isfinite(lp) ? -lp : 1e10
    end
    y0 = x_unc[free_idx]
    res = Optim.optimize(neg_lp, y0, Optim.NelderMead(),
                          Optim.Options(iterations=2000, g_tol=1e-4))
    return -Optim.minimum(res)   # log-posterior at profiled P
end

println("Profiling log-posterior over log P (NelderMead at each P)...")
println("  P (yr)    log_post")
P_grid_yr = 10 .^ range(log10(70), log10(10000), length=25)
log_post_vals = Float64[]
for P_yr in P_grid_yr
    P_d = P_yr * 365.25
    lp = profile_at_P(P_d)
    push!(log_post_vals, lp)
    @printf("  %8.1f    %.2f\n", P_yr, lp)
end
println()
imax = argmax(log_post_vals)
P_max = P_grid_yr[imax]
@printf("Best P from coarse scan: %.1f yr (log_post=%.2f)\n", P_max, log_post_vals[imax])
println()
println("Brandt+ 2021 P = 2880 yr   ;   our C3 found P = 807 yr")

# Scan finer near each peak we find
peaks = Int[]
for i in 2:(length(log_post_vals)-1)
    if log_post_vals[i] > log_post_vals[i-1] && log_post_vals[i] > log_post_vals[i+1]
        push!(peaks, i)
    end
end
if !isempty(peaks)
    println()
    println("Local peaks found in the scan:")
    for i in peaks
        @printf("  P ≈ %.1f yr,  log_post = %.2f\n", P_grid_yr[i], log_post_vals[i])
    end
else
    println("\nOnly a single monotonic peak detected — no alternate modes nearby.")
end
