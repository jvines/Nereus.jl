#!/usr/bin/env julia
# Render the HD 114762 2-Keplerian fit (fit_HD114762_2kep_dr4.jl).
# Rebuilds the target — cheap and deterministic — and reads the saved posterior,
# so plot style iterates without re-running Pigeons.
#
# Usage:  julia --project=. test/astrometry/plots_HD114762_2kep.jl
#   env HD114762_RV, NEREUS_GAIA_DR4_XML, NEREUS_OUTDIR

using Nereus
using MCMCChains
using Statistics: median
using Printf

const SID = 3937211745905473024
const M_PRI, PLX, PLX_ERR = 0.83, 25.36, 0.30

rvfile = get(ENV, "HD114762_RV", "")
isfile(rvfile) || error("set HD114762_RV")
tb = Dict{String, Vector{Float64}}()
for line in eachline(rvfile)
    s = strip(line); (isempty(s) || startswith(s, "#")) && continue
    t, rv, er, ins = split(s)
    append!(get!(() -> Float64[], tb, ins),
            (parse(Float64, t) - 2_400_000.5, parse(Float64, rv), parse(Float64, er)))
end
mkrv(k) = let v = reshape(tb[k], 3, :); (t = v[1, :], rv = v[2, :], rv_err = v[3, :]) end
hires, lick = mkrv("j"), mkrv("lick")

xml = get(ENV, "NEREUS_GAIA_DR4_XML", "")
(isempty(xml) || !isfile(xml)) && (xml = fetch_gaia_dr4_prerelease())
src = read_gaia_epoch_votable(xml, SID)

t_rv    = vcat(hires.t, lick.t)
rv      = vcat(hires.rv, lick.rv)
rv_err  = vcat(hires.rv_err, lick.rv_err)
rv_inst = vcat(fill(1, length(hires.t)), fill(2, length(lick.t)))
data = Data(; t_rv = t_rv, rv = rv, rv_err = rv_err, rv_inst = rv_inst, iad = src.iad)

priors = Dict{String, PriorSpec}(
    "P_k1"     => LogUniformPrior(60.0, 110.0),
    "M_sec_k1" => LogUniformPrior(0.003, 0.5),
    "inc_k1"   => SinePrior(),
    "Omega_k1" => UniformPrior(0.0, 2π),
    "P_k2"     => LogUniformPrior(1.1e4, 1.0e7),
    "M_sec_k2" => LogUniformPrior(1e-3, 0.5),
    "plx"      => NormalPrior(PLX, PLX_ERR),
)
params = Params(; max_kplanet = 2, planet_modes = [RVAS, RV_ONLY],
                instruments = InstrumentConfig(rv = ["HIRES", "Lick"], pm = String[]),
                parametrization = ParametrizationConfig(mass = :M_sec_driven),
                priors = priors, data = data, M_s = M_PRI, trend_order = 0)

OUTDIR = get(ENV, "NEREUS_OUTDIR", joinpath(@__DIR__, "plots_HD114762_2kep"))
chains, _ = load_chains(joinpath(OUTDIR, "chains.nc"))
@printf("posterior %d iter × %d chains\n", size(chains, 1), size(chains, 3))

for (fn, name) in [(plot_iad_residuals,       "iad_residuals"),
                   (plot_rv_timeseries,       "rv_timeseries"),
                   (plot_rv_phasefold,        "rv_phasefold"),
                   (plot_rv_astrom_phasefold, "rv_astrom_phasefold"),
                   (plot_orbit_skyplane,      "orbit_skyplane")]
    try
        fn(chains, params, data; output = OUTDIR)
        println("saved $name")
    catch err
        println("$name FAILED: ", sprint(showerror, err))
    end
end

try
    plot_corner(chains, params; output = OUTDIR,
                params_to_plot = ["P_k1", "M_sec_k1", "inc_k1", "Omega_k1",
                                  "P_k2", "M_sec_k2"])
    println("saved corner")
catch err
    println("corner FAILED: ", sprint(showerror, err))
end
println("plots in $OUTDIR")
