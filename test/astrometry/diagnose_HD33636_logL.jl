#!/usr/bin/env julia
# Diagnostic: at Bean+ 2007's best-fit values vs Nereus's converged
# (alternative) mode, which has higher log-likelihood? Settles the
# "sampling failure vs likelihood gap" question.
#
# Run: julia --project=Nereus.jl test/astrometry/diagnose_HD33636_logL.jl

using Nereus
using DelimitedFiles
using Statistics: median
using Printf
using LogDensityProblems

# ----- Re-build target identically to fit_HD33636_IAD_RV.jl -----
RV_FILE = joinpath(@__DIR__, "..", "data", "hd33636.csv")
raw = readdlm(RV_FILE, ',', Any, '\n'; header = true)
data_mat = raw[1]
_to_f(col) = Float64[v isa AbstractString ? parse(Float64, v) : Float64(v) for v in col]
bjd      = _to_f(data_mat[:, 1])
rv       = _to_f(data_mat[:, 2])
rv_err   = _to_f(data_mat[:, 3])
inst_str = String.(data_mat[:, 16])

keep_inst = Set(["HIRES_PRE", "HIRES_POST", "HRS"])
keep = findall(s -> s in keep_inst, inst_str)
bjd    = bjd[keep];  rv = rv[keep];  rv_err = rv_err[keep]
inst_str = inst_str[keep]
inst_names = sort(unique(inst_str))
inst_map   = Dict(name => i for (i, name) in enumerate(inst_names))
rv_inst    = Int[inst_map[s] for s in inst_str]
iad = fetch_hip_iad(24205; verbose = false)
data = Data(; t_rv = bjd, rv = rv, rv_err = rv_err, rv_inst = rv_inst, iad = iad)
ic = InstrumentConfig(rv = inst_names)

const M_PRI = 1.02
parametrization = ParametrizationConfig(
    ew = :sesinw, time = :Tp, geom = :b_rr, mass = :M_sec_driven,
)
priors = Dict{String, PriorSpec}(
    "P_k1"       => LogUniformPrior(1000.0, 5000.0),
    "M_sec_k1"   => LogUniformPrior(0.005, 0.5),
    "sesinw_k1"  => UniformPrior(-1.0, 1.0),
    "secosw_k1"  => UniformPrior(-1.0, 1.0),
    "Tp_k1"      => UniformPrior(2_451_000.0, 2_454_500.0),
    "inc_k1"     => SinePrior(),
    "Omega_k1"   => UniformPrior(0.0, 2π),
    "plx"        => NormalPrior(35.25, 1.02, 5.0, 70.0),
    ("gamma_$inst"  => UniformPrior(-1000.0, 1000.0) for inst in inst_names)...,
    ("sigma_$inst"  => LogUniformPrior(0.5, 50.0)    for inst in inst_names)...,
    "M_pri"       => FixedPrior(M_PRI),
)
params = Params(;
    max_kplanet=1, planet_modes=[RVAS], instruments=ic, data=data,
    M_s=M_PRI, parametrization=parametrization, priors=priors,
    stability=:none, trend_order=0,
)
target = NereusTarget(params, data; unconstrained = false)
layout = params.layout

# ----- Helper: build Theta at a named-parameter dict --------
function _make_theta(vals::Dict{String, Float64})
    theta = Theta{Float64}(params)
    for (name, v) in vals
        haskey(layout.name_to_idx, name) || continue
        theta.values[layout.name_to_idx[name]] = v
    end
    return theta
end

# ----- Evaluate log L at two parameter points ---------------
function eval_point(label::String, vals::Dict{String, Float64})
    theta = _make_theta(vals)
    lp = log_prior(theta)
    ll_rv = rv_log_likelihood(theta, data)
    ll_iad = isfinite(lp) ? astrom_log_likelihood(theta, data) : -Inf
    ll_total = ll_rv + ll_iad
    @printf("\n%-40s\n", label)
    @printf("  log_prior     = %12.3f\n", lp)
    @printf("  log L (RV)    = %12.3f\n", ll_rv)
    @printf("  log L (IAD)   = %12.3f\n", ll_iad)
    @printf("  log L total   = %12.3f\n", ll_total)
    @printf("  log post      = %12.3f\n", lp + ll_total)
    return (lp = lp, ll_rv = ll_rv, ll_iad = ll_iad, lp_post = lp + ll_total)
end

# Bean+ 2007 best-fit values
bean = Dict(
    "P_k1"       => 2447.3,
    "M_sec_k1"   => 142.0 / 1047.57,
    "sesinw_k1"  => sqrt(0.4805) * sind(169.4),
    "secosw_k1"  => sqrt(0.4805) * cosd(169.4),
    "Tp_k1"      => 2_451_772.0,
    "inc_k1"     => deg2rad(4.1),
    "Omega_k1"   => 0.0,
    "plx"        => 35.25,
    "gamma_HIRES_PRE"  => 0.0, "gamma_HIRES_POST" => 0.0, "gamma_HRS" => 0.0,
    "sigma_HIRES_PRE"  => 5.0, "sigma_HIRES_POST" => 3.0, "sigma_HRS" => 3.0,
)
# Same as Bean, but inc near 90° (Nereus's converged alt mode)
edge = copy(bean)
edge["inc_k1"]   = deg2rad(85.0)
edge["M_sec_k1"] = 0.010    # ~10 M_J, matches m·sini at sin i ≈ 1

# Nereus's ACTUAL converged mode from the ptemcee production run
# (chain medians)
nereus = Dict(
    "P_k1"       => 2185.6,
    "M_sec_k1"   => 91.8 / 1047.57,
    "sesinw_k1"  => 0.294,
    "secosw_k1"  => 0.045,
    "Tp_k1"      => 2_451_625.8,
    "inc_k1"     => 0.060,            # 3.4°  ✓ Bean
    "Omega_k1"   => 1.65,
    "plx"        => 35.4,
    "gamma_HIRES_PRE"  => 18.9,
    "gamma_HIRES_POST" =>  3.7,
    "gamma_HRS"        =>  2.4,
    "sigma_HIRES_PRE"  => 42.1,       # chain-inflated — systematic-offset absorption
    "sigma_HIRES_POST" => 45.3,
    "sigma_HRS"        =>  6.6,
)

println("=" ^ 60)
println("Log-likelihood diagnostic at three parameter points")
println("=" ^ 60)
b = eval_point("Bean+ 2007 (P=2447, M=142 MJ, e=0.48, inc=4°)", bean)
e = eval_point("Bean's orbit but inc=85° (low-mass edge-on)", edge)
p = eval_point("Nereus converged mode (P=2107, M=21 MJ, inc=85°)", nereus)

println("\n" * "=" ^ 60)
println("Verdict")
println("=" ^ 60)
@printf("  Δ log post (NEREUS_mode − Bean)  = %+.2f\n", p.lp_post - b.lp_post)
@printf("  Δ log post (edge_at_Bean − Bean)  = %+.2f\n", e.lp_post - b.lp_post)
@printf("  Δ log L_IAD (NEREUS_mode − Bean) = %+.2f\n", p.ll_iad - b.ll_iad)
@printf("  Δ log L_RV  (NEREUS_mode − Bean) = %+.2f\n", p.ll_rv - b.ll_rv)

if b.lp_post > p.lp_post
    println("\n→ Bean+'s mode has HIGHER log-posterior than Nereus's converged mode.")
    println("  Diagnosis: sampling failure. PT didn't find Bean's mode but it exists.")
    println("  Fix: more chains/temps, longer warmup, or explicit init constraint.")
elseif p.lp_post > b.lp_post
    println("\n→ Nereus's mode has HIGHER log-posterior than Bean+'s mode.")
    println("  Diagnosis: likelihood gap. The joint formulation has a higher peak")
    println("  somewhere Nereus finds and Bean+ doesn't. Likely the IAD design")
    println("  matrix's linear pm corrections are absorbing the orbital reflex.")
    println("  Fix: anchor (μα, μδ, π) corrections to HIP catalogue priors.")
else
    println("\n→ Bean and Nereus mode are equally probable. Multi-modal posterior;")
    println("  both are legitimate solutions of the joint likelihood.")
end
