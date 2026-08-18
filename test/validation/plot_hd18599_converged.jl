#!/usr/bin/env julia
# Plots for the converged HD 18599 joint run: K detection significance,
# bimodal e, K-e modes, and the RV + transit phase-folds. House style:
# NEREUS_CMAP (:cool), no titles, info in axis labels.
# Run AFTER hd18599_joint_converged.jl (reads its saved chains + K_e_samples).

using Nereus, MCMCChains, DelimitedFiles, Statistics, Printf
using CairoMakie

const RV_FILE = joinpath(@__DIR__, "..", "data", "hd18599.csv")
const LC_FILE = joinpath(@__DIR__, "..", "..", "..", "data", "HD18599", "HD18599_cleaned_lc.csv")
const P_REF, T0_REF, M_S, R_S = 4.1374685534602405, 2.4583545857470357e6, 0.807, 0.798
const RHO_S = M_S / R_S^3
const PAPER_INST = Set(["HARPS_PRE", "HARPS_POST", "FEROS"])
const OUTDIR = joinpath(@__DIR__, "..", "..", "results", "HD18599_joint_converged")
const CMAP = :cool   # = NEREUS_CMAP (house colormap)

# ---------- rebuild data + params (same loader as the run) ----------
raw = readdlm(RV_FILE, ',', Any, '\n'; header=true); dm = raw[1]
keep=Int[]; inst_str=String[]
for i in 1:size(dm,1)
    ins=strip(String(dm[i,16])); prov=strip(String(dm[i,17]))
    (ins=="HARPS_POST" && prov=="ESO_PHASE3") && continue
    ins in PAPER_INST || continue; push!(keep,i); push!(inst_str,ins)
end
dm=dm[keep,:]; bjd=Float64.(dm[:,1]); rv=Float64.(dm[:,2]); rverr=Float64.(dm[:,3])
inst_names=sort!(unique(inst_str)); imap=Dict(n=>i for (i,n) in enumerate(inst_names))
rv_inst=[imap[s] for s in inst_str]
let kc=trues(length(bjd))
    for id in unique(rv_inst)
        idx=findall(==(id),rv_inst); m=median(rv[idx]); thr=5*max(1.4826*median(abs.(rv[idx].-m)),1e-6)
        for k in idx; abs(rv[k]-m)>thr && (kc[k]=false); end
    end
    global bjd=bjd[kc]; global rv=rv[kc]; global rverr=rverr[kc]; global rv_inst=rv_inst[kc]; global dm=dm[kc,:]
end
col=Dict(:bis=>4,:fwhm=>6,:halpha=>10,:logrhk=>12)
function load_adfmt(cv)
    v=Float64[let x=dm[i,cv]; (x==="" || (x isa AbstractString && strip(x)=="")) ? 0.0 : Float64(x) end for i in 1:size(dm,1)]
    for id in unique(rv_inst)
        idx=findall(==(id),rv_inst); fin=filter(k->isfinite(v[k]),idx); isempty(fin)&&continue
        μ=mean(v[fin]); for k in fin; v[k]-=μ; end
        vmax=maximum(abs,v[fin]); vmax==0&&continue
        rvv=rv[idx]; rmax=maximum(abs,rvv.-mean(rvv)); rmax==0&&continue
        for k in fin; v[k]=v[k]/vmax*rmax; end
    end
    v
end
adkey=Dict(:bis=>"bisector_span",:fwhm=>"fwhm_AD",:halpha=>"halpha_AD",:logrhk=>"log_rhk_AD")
chs=[:bis,:fwhm,:halpha,:logrhk]
indicators=Dict{String,Vector{Float64}}(); ierrs=Dict{String,Vector{Float64}}()
for ch in chs; indicators[adkey[ch]]=load_adfmt(col[ch]); ierrs[adkey[ch]]=fill(1.0,length(bjd)); end
lc = readdlm(LC_FILE, ',', Float64; comments=true, comment_char='#', header=true)[1]
data=Data(; t_rv=bjd, rv=rv, rv_err=rverr, rv_inst=rv_inst,
            t_phot=lc[:,1], flux=lc[:,2], flux_err=lc[:,3], phot_inst=ones(Int,size(lc,1)),
            indicators=indicators, indicator_errs=ierrs)
ic=InstrumentConfig(rv=inst_names, pm=["TESS"])
rvmax=maximum(abs,rv)
pri=Dict{String,PriorSpec}(
    "n_p"=>FixedPrior(1.0), "P_k1"=>NormalPrior(P_REF,1e-3,P_REF-0.01,P_REF+0.01),
    "Tc_k1"=>NormalPrior(T0_REF,0.05,T0_REF-0.3,T0_REF+0.3), "K_k1"=>UniformPrior(0.0,50.0),
    "sesinw_k1"=>UniformPrior(-0.7,0.7), "secosw_k1"=>UniformPrior(-0.7,0.7),
    "b_k1"=>UniformPrior(0.0,1.0), "rr_k1"=>UniformPrior(0.005,0.10),
    "rho_s"=>NormalPrior(RHO_S,0.4,0.3,4.0), "q1_TESS"=>UniformPrior(0.0,1.0), "q2_TESS"=>UniformPrior(0.0,1.0),
    "offset_TESS"=>NormalPrior(0.0,1e-3,-5e-3,5e-3), "jitter_TESS"=>LogUniformPrior(1e-5,5e-3))
for nm in inst_names; pri["gamma_$nm"]=UniformPrior(-3rvmax,3rvmax); pri["sigma_$nm"]=LogUniformPrior(1e-3,30.0); end
params=Params(; max_kplanet=1, planet_modes=[RVPM], instruments=ic, data=data, M_s=M_S, R_s=R_S,
    parametrization=ParametrizationConfig(time=:Tc, geom=:b_rr, use_rho_s=true),
    priors=pri, noise_models=NoiseModel[ActivityDecorrelation(indicators=[adkey[c] for c in chs])], stability=:none)

# ---------- K / e samples ----------
M = readdlm(joinpath(OUTDIR,"K_e_samples.csv"), ','); K=M[:,1]; e=M[:,2]
q(x,p)=quantile(x,p)
Kmed=median(K); K3lo=q(K,0.0015)
@printf("K=%.2f  3σ-lo=%.2f → %s ;  e bimodal: low<0.22 %.0f%%, high≥0.22 %.0f%%\n",
        Kmed,K3lo, K3lo>0 ? "DETECTION" : "non-detection", 100mean(e.<0.22),100mean(e.>=0.22))

# ---------- figure 1: K detection + e bimodality + K-e modes ----------
fig = Figure(; size=(1150, 360))
# (a) K posterior
axK = Axis(fig[1,1]; xlabel=rich("K", subscript("b"), " [m s", superscript("-1"), "]"),
           ylabel="posterior density")
hist!(axK, K; bins=80, normalization=:pdf, color=(:steelblue,0.6))
vlines!(axK, [0.0]; color=:black, linewidth=1.5)
vlines!(axK, [Kmed]; color=:crimson, linewidth=2)
vlines!(axK, [q(K,0.16),q(K,0.84)]; color=:crimson, linestyle=:dash, linewidth=1)
vlines!(axK, [K3lo]; color=:darkorange, linestyle=:dot, linewidth=2)       # 3σ lower bound
vlines!(axK, [11.0]; color=:gray40, linestyle=:dashdot, linewidth=1.5)     # two-paper K=11
text!(axK, 11.0, 0.0; text="Vines/Desidera", rotation=π/2, align=(:left,:top), color=:gray40, fontsize=11, offset=(2,-2))
# (b) e posterior
axe = Axis(fig[1,2]; xlabel=rich("e", subscript("b")), ylabel="posterior density")
hist!(axe, e; bins=80, normalization=:pdf, color=(:teal,0.6))
vlines!(axe, [0.2]; color=:gray40, linestyle=:dashdot, linewidth=1.5)
vlines!(axe, [0.34]; color=:gray60, linestyle=:dot, linewidth=1.5)
# (c) K-e modes
axKe = Axis(fig[1,3]; xlabel=rich("e", subscript("b")), ylabel=rich("K", subscript("b")," [m s",superscript("-1"),"]"))
hb = hexbin!(axKe, e, K; bins=60, colormap=CMAP)
hlines!(axKe, [11.0]; color=:gray40, linestyle=:dashdot, linewidth=1.5)
hlines!(axKe, [0.0]; color=:black, linewidth=1)
Colorbar(fig[1,4], hb; label="samples")
save(joinpath(OUTDIR,"HD18599_K_e_detection.png"), fig; px_per_unit=2)
println("wrote HD18599_K_e_detection.png")

# ---------- figure 2/3: Nereus RV + transit phase-folds (winning model) ----------
ch1, _ = load_chains(joinpath(OUTDIR,"chains_s1.nc"))
# plot_* take `output` as a DIRECTORY and write output/models/<name>.png
try
    plot_rv_phasefold(ch1, params, data; output=OUTDIR)
    println("wrote ", OUTDIR, "/models/rv_phased_K1.png")
catch err; @warn "rv phasefold failed" exception=(err, catch_backtrace()); end
try
    plot_transit_phasefold(ch1, params, data; output=OUTDIR)
    println("wrote ", OUTDIR, "/models/transit phasefold")
catch err; @warn "transit phasefold failed" exception=(err, catch_backtrace()); end
