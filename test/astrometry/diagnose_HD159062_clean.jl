# Cleaner diagnostic: take a known-good orbit (our found Pigeons mode
# at P=372 yr) and just walk P up while letting Mo + nuisances re-fit.
# If log_post stays high near Brandt's P=2880 yr, modes coexist.

using Nereus, LogDensityProblems, Optim, Printf
using Statistics: median

DATADIR = joinpath(@__DIR__, "..", "..", "..", "data", "HD159062")
hgca   = load_hgca_row(joinpath(DATADIR, "HGCA_vEDR3.fits"), 85653)
rvdat  = load_orvara_rv(joinpath(DATADIR, "HD159062_RV.dat"))
relast = load_orvara_relast(joinpath(DATADIR, "HD159062_relAST.txt"))
data = Data(t_rv=rvdat.t, rv=rvdat.rv, rv_err=rvdat.rv_err,
            rv_inst=rvdat.rv_inst, relastrom=relast, hgca=hgca)

const M_PRI = 0.81

# Reuse the fit_HD159062_Msec.jl model setup
params = Params(
    max_kplanet = 1, planet_modes = [RVAS],
    instruments = InstrumentConfig(rv = ["HIRES"]),
    data = data, stability = :none, M_s = M_PRI,
    parametrization = ParametrizationConfig(mass = :M_sec_driven),
    priors = Dict{String, PriorSpec}(
        "n_p"          => FixedPrior(1.0),
        "P_k1"         => LogUniformPrior(50.0 * 365.25, 30000.0 * 365.25),
        "M_sec_k1"     => LogUniformPrior(0.001, 2.0),
        "sesinw_k1"    => UniformPrior(-1.0, 1.0),
        "secosw_k1"    => UniformPrior(-1.0, 1.0),
        "Mo_k1"        => UniformPrior(0.0, 2π),
        "inc_k1"       => SinePrior(),
        "Omega_k1"     => UniformPrior(0.0, 2π),
        "sigma_HIRES"  => LogUniformPrior(1e-3, 1e3),
        "M_pri"        => NormalPrior(0.81, 0.04, 0.5, 1.2),
    ),
)
target = NereusTarget(params, data)
unfrozen = params.layout.unfrozen_names

# Set initial theta from our Pigeons posterior medians (at P=372 yr)
init_b = Dict{String, Float64}(
    "P_k1"        => 372.4 * 365.25,
    "M_sec_k1"    => 0.5419,
    "sesinw_k1"   => sqrt(0.7228) * sin(deg2rad(150)),    # rough ω
    "secosw_k1"   => sqrt(0.7228) * cos(deg2rad(150)),
    "Mo_k1"       => 1.5,
    "inc_k1"      => deg2rad(42.94),
    "Omega_k1"    => deg2rad(154.08),
    "plx"         => hgca.plx,
    "gamma_HIRES" => median(rvdat.rv),
    "sigma_HIRES" => 5.0,
    "M_pri"       => 0.7082,
)
x0_b = [init_b[nm] for nm in unfrozen]
x0_u = Nereus.transform_forward(x0_b, target.transform.type_ids,
                                  target.transform.lowers, target.transform.uppers)
ll0 = LogDensityProblems.logdensity(target, x0_u)
println(@sprintf("log_post at our found mode (P=372 yr): %.2f", ll0))
println()

# Now: vary P (in bounded space), keep all else fixed, scan.
P_idx_unc = findfirst(==("P_k1"), unfrozen)
Pmin = target.transform.lowers[P_idx_unc]
Pmax = target.transform.uppers[P_idx_unc]
function _P_to_unc(P_d)
    z = (log(P_d) - log(Pmin)) / (log(Pmax) - log(Pmin))
    z = clamp(z, 1e-12, 1-1e-12)
    return log(z / (1-z))
end

println("Scan log_post as a function of P (all other params at our found mode):")
println("  P (yr)     log_post     Δ vs found mode")
P_grid = [100, 150, 200, 300, 372, 500, 800, 1200, 1800, 2500, 2880, 4000, 6000, 10000, 20000]
for P_yr in P_grid
    x = copy(x0_u)
    x[P_idx_unc] = _P_to_unc(P_yr * 365.25)
    lp = LogDensityProblems.logdensity(target, x)
    @printf("  %6d    %10.2f    %+10.2f\n", P_yr, lp, lp - ll0)
end

println()
println("Now also let Mo and the other nuisances re-optimize at each P:")
println("(NelderMead initialized at the found mode, with P pinned)")
println("  P (yr)    log_post  ")

function obj_pin_P(x_unc, P_unc, P_idx)
    function _obj(y)
        xx = copy(x_unc)
        # y has length n-1; map to all indices except P
        ji = 0
        for i in 1:length(xx)
            if i == P_idx
                xx[i] = P_unc
            else
                ji += 1
                xx[i] = y[ji]
            end
        end
        lp = LogDensityProblems.logdensity(target, xx)
        return isfinite(lp) ? -lp : 1e10
    end
    return _obj
end

for P_yr in P_grid
    P_unc = _P_to_unc(P_yr * 365.25)
    obj = obj_pin_P(x0_u, P_unc, P_idx_unc)
    # Init y from our found-mode unconstrained values (excluding P)
    y0 = [x0_u[i] for i in 1:length(x0_u) if i != P_idx_unc]
    res = Optim.optimize(obj, y0, NelderMead(),
                          Optim.Options(iterations=5000, g_tol=1e-4))
    lp = -Optim.minimum(res)
    @printf("  %6d   %10.2f\n", P_yr, lp)
end
