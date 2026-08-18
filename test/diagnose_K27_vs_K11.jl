# Take PT GP-only median sample, evaluate log L at K=11 (PT median),
# K=27 (INS v4 reported), and a sweep across the K prior. Tells us
# whether K=27 is genuinely a high-L region or whether INS was just
# stuck.

using Nereus
using DelimitedFiles
using Statistics
using Printf

const REPO_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const RV_FILE   = joinpath(@__DIR__, "data", "hd18599.csv")
const LC_FILE   = joinpath(REPO_ROOT, "data", "HD18599", "HD18599_cleaned_lc.csv")
const PT_PATH   = joinpath(REPO_ROOT, "Nereus.jl", "results", "HD18599_RVPM_transdim", "chains.nc")

# Build target (same as fit_HD18599_RVPM_Desidera.jl)
raw = readdlm(RV_FILE, ',', Any, '\n'; header=true)
data_mat = raw[1]
inst_raw = String.(data_mat[:, 16])
prov_raw = String.(data_mat[:, 17])
PAPER_INSTRUMENTS = Set(["HARPS_PRE", "HARPS_POST", "FEROS"])
keep = Int[]; inst_str = String[]
for i in 1:size(data_mat, 1)
    ins  = strip(inst_raw[i])
    prov = strip(prov_raw[i])
    if ins == "HARPS_POST" && prov == "ESO_PHASE3"
        continue
    end
    ins in PAPER_INSTRUMENTS || continue
    push!(keep, i); push!(inst_str, String(ins))
end
data_mat = data_mat[keep, :]
_dparse(col) = Float64[v isa AbstractString ? parse(Float64, v) : Float64(v) for v in col]
bjd = _dparse(data_mat[:, 1])
rv  = _dparse(data_mat[:, 2])
err = _dparse(data_mat[:, 3])
bis = _dparse(data_mat[:, 4])
inst_names = sort(unique(inst_str))
inst_to_idx = Dict(name => i for (i, name) in enumerate(inst_names))
rv_inst = Int[inst_to_idx[s] for s in inst_str]
# 5σ MAD per instrument
function mad_outlier_mask(v, inst; k=5.0)
    keep = trues(length(v))
    for id in unique(inst)
        idx = findall(==(id), inst)
        med = median(view(v, idx))
        mad = median(abs.(view(v, idx) .- med))
        mad == 0 && continue
        σ = 1.4826*mad
        for j in idx
            abs(v[j] - med) > k*σ && (keep[j] = false)
        end
    end
    keep
end
m = mad_outlier_mask(rv, rv_inst)
bjd = bjd[m]; rv = rv[m]; err = err[m]; bis = bis[m]; rv_inst = rv_inst[m]

# LC file has '#'-prefixed metadata header followed by 'time,flux,flux_err'.
# Skip all comment lines manually before passing to readdlm.
lc_lines = readlines(LC_FILE)
data_lines = filter(l -> !startswith(l, "#"), lc_lines)
# First non-# line is the column header
@assert occursin("bjd", lowercase(data_lines[1])) || occursin("time", lowercase(data_lines[1])) "expected LC column header"
data_lines = data_lines[2:end]
lc_mat = reduce(vcat, [permutedims(parse.(Float64, split(l, ','))) for l in data_lines])
t_phot   = vec(lc_mat[:, 1])
flux     = vec(lc_mat[:, 2])
flux_err = vec(lc_mat[:, 3])
data = Data(; t_rv=bjd, rv=rv, rv_err=err, rv_inst=rv_inst,
              indicators=Dict("bisector_span" => bis),
              t_phot=t_phot, flux=flux, flux_err=flux_err,
              phot_inst=ones(Int, length(t_phot)))
ic = InstrumentConfig(rv=inst_names, pm=["TESS"])
gp_rv = CeleriteRotation(channel = :rv)
P_REF, T0_REF = 4.1374685534602405, 2.4583545857470357e6
M_S, R_S = 0.807, 0.798
rv_max = maximum(abs, rv)
@printf("DEBUG rv: n=%d max=%.6f sample=%s\n", length(rv), rv_max,
        string(rv[1:min(5, length(rv))]))
@assert rv_max > 0.1 "rv data parsed badly — rv_max=$(rv_max)"
priors = Dict{String, PriorSpec}()
priors["P_k1"]      = NormalPrior(P_REF, 1e-3, P_REF - 0.01, P_REF + 0.01)
priors["K_k1"]      = UniformPrior(0.0, 50.0)
priors["sesinw_k1"] = UniformPrior(-1.0, 1.0)
priors["secosw_k1"] = UniformPrior(-1.0, 1.0)
priors["Tc_k1"]     = NormalPrior(T0_REF, 0.05, T0_REF - 0.5, T0_REF + 0.5)
priors["b_k1"]      = UniformPrior(0.0, 1.0)
priors["rr_k1"]     = NormalPrior(0.031, 0.005, 0.005, 0.10)
for n in inst_names; priors["gamma_$n"] = UniformPrior(-3*rv_max, 3*rv_max); end
priors["offset_TESS"]  = NormalPrior(0.0, 1e-3, -5e-3, 5e-3)
priors["jitter_TESS"]  = LogUniformPrior(1e-5, 5e-3)
priors["q1_TESS"]      = UniformPrior(0.0, 1.0)
priors["q2_TESS"]      = UniformPrior(0.0, 1.0)
priors["rho_s"]        = NormalPrior(2.241, 0.479, 0.1, 10.0)
priors["gp_sigma"]     = LogUniformPrior(0.1, rv_max)
priors["gp_period"]    = NormalPrior(8.74, 0.1, 8.0, 9.5)
priors["gp_Q0"]        = LogUniformPrior(1.0, 10.0)
priors["gp_dQ"]        = UniformPrior(0.0, 3.0)
priors["gp_f"]         = UniformPrior(0.05, 0.5)
priors["dvdt"]         = UniformPrior(0.0, 0.1)
@printf("DEBUG: rv length=%d, rv_max=%.6f, sample=%s\n",
        length(rv), rv_max, string(rv[1:min(5, length(rv))]))
parametrization = ParametrizationConfig(time=:Tc, geom=:b_rr, use_rho_s=true)
params = Params(; max_kplanet=1, planet_modes=[RVPM], instruments=ic, data=data,
                 M_s=M_S, R_s=R_S, parametrization=parametrization, priors=priors,
                 noise_models=[gp_rv], transdim_noise=false, trend_order=1)

# Load PT trans-dim chain, slice to GP-only
pt_chains, pt_params = load_chains(PT_PATH)
mask = vec(Bool.(Int.(Array(pt_chains[:noise_active_4]))))
mask .&= .!vec(Bool.(Int.(Array(pt_chains[:noise_active_1]))))
mask .&= .!vec(Bool.(Int.(Array(pt_chains[:noise_active_2]))))
mask .&= .!vec(Bool.(Int.(Array(pt_chains[:noise_active_3]))))
gp_only_idx = findall(mask)

# Find the PT sample with max log_post (approximate via highest L; we'll just
# pick the GP-only sample with K closest to median, since we don't have log_L
# stored directly)
K_pt = vec(Array(pt_chains[:K_k1]))
K_med = median(K_pt[gp_only_idx])
ref_idx = gp_only_idx[argmin(abs.(K_pt[gp_only_idx] .- K_med))]
@printf("Reference PT sample idx=%d, K=%.3f (GP-only slice median K=%.3f)\n",
        ref_idx, K_pt[ref_idx], K_med)

layout = params.layout
theta = Theta{Float64}(params)
# Initialize from PT median sample (every Desidera param)
for n_sym in names(pt_chains, :parameters)
    name = String(n_sym)
    if haskey(layout.name_to_idx, name)
        val = Float64(pt_chains[n_sym].data[ref_idx])
        # Clamp to prior bounds
        if haskey(priors, name)
            lo, hi = bounds(priors[name])
            val = clamp(val, lo, hi)
        end
        theta.values[layout.name_to_idx[name]] = val
    end
end

# Evaluate baseline log L (at PT K_med ≈ 11.5)
lp_base = log_prior(theta)
ll_base = rv_log_likelihood(theta, data) + transit_log_likelihood(theta, data)
@printf("\nBaseline (PT median sample): K=%.3f, gp_period=%.3f, gp_sigma=%.3f\n",
        theta.values[layout.name_to_idx["K_k1"]],
        theta.values[layout.name_to_idx["gp_period"]],
        theta.values[layout.name_to_idx["gp_sigma"]])
@printf("  log prior = %.3f,  log L = %.3f,  log post = %.3f\n",
        lp_base, ll_base, lp_base + ll_base)

# K sweep: swap only K, keep all other params at PT median
println("\nK-sweep with all other params fixed at PT median:")
println("  K      log L         log post")
for K_test in [0.0, 5.0, 8.0, 11.0, 11.5, 15.0, 20.0, 24.0, 27.0, 30.0, 35.0, 40.0, 45.0]
    theta.values[layout.name_to_idx["K_k1"]] = K_test
    lp = log_prior(theta)
    ll = rv_log_likelihood(theta, data) + transit_log_likelihood(theta, data)
    @printf("  %5.1f   %12.3f   %12.3f\n", K_test, ll, lp + ll)
end
