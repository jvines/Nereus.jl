#!/usr/bin/env julia
# "Subtract-and-BLS shenanigans" on the correct data-driven ephemerides:
# flatten -> fit b transit -> subtract -> BLS residual reveals d -> fit d ->
# subtract -> BLS reveals e -> fit e -> subtract -> BLS is clean. Each stage:
# full BLS periodogram (left, peak marked) + phase-fold with fitted model
# overlaid (right, robust module binning).

using Nereus
using CairoMakie
using Statistics: std, median, mean
using Printf

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))
lc = load_tess_lc(joinpath(DATADIR, "WASP-47_k2_c03_everest_lc.csv"))
t=lc.t; f=lc.flux; e=lc.flux_err; t1=t[1]
runmed(y,w)=(n=length(y);m=similar(y);h=w÷2;for i in 1:n;lo=max(1,i-h);hi=min(n,i+h);m[i]=median(@view y[lo:hi]);end;m)
flat = f ./ runmed(f,151)

# --- full BLS periodogram via the production per-period kernel, using the
#     NEW a/Rs-scaled duration grid (no hand-tuned durations) ---
const RHO   = 1.04 / 1.137^3                 # WASP-47 stellar density [ρ_sun]
const MULTS = [0.3, 0.5, 0.7, 1.0, 1.3]
const NPB   = 300
function bls_pgram(flux, periods)
    w = 1.0 ./ e.^2; W=sum(w); fm=sum(w.*flux)/W; wf=w.*(flux.-fm)
    tt = t .- t1; bw=zeros(NPB); bwf=zeros(NPB); snr=similar(periods)
    for (i,P) in enumerate(periods)
        durs_p = Nereus._duty_grid(RHO, P, MULTS)   # a/Rs-scaled per period
        s,_,_,_ = Nereus._bls_one_period(tt, w, wf, W, P, durs_p, NPB, bw, bwf)
        snr[i]=s
    end
    snr
end
periods = exp.(range(log(0.5), log(12.0); length=5000))

# --- empirical Tc at a given period (deepest weighted bin) ---
function tcscan(flux,P)
    ph=@. mod((t-t1)/P+0.5,1.0)-0.5; w=1.0 ./ e.^2
    edges=range(-0.5,0.5;length=121); best=Inf; bestc=0.0
    for k in 1:120
        m=(ph.>=edges[k]).&(ph.<edges[k+1]); count(m)<4 && continue
        v=sum(w[m].*flux[m])/sum(w[m]); v<best && (best=v; bestc=(edges[k]+edges[k+1])/2)
    end
    t1 + bestc*P
end

# --- fit one transit (PM-only), return per-cadence model + params ---
function fit_transit(flux, Pseed, T0seed; sigP=1.5e-3)
    data=Data(; t_phot=t, flux=flux, flux_err=e, phot_inst=ones(Int,length(t)))
    pri=Dict{String,PriorSpec}()
    pri["P_k1"]=NormalPrior(Pseed,sigP,Pseed-6*sigP,Pseed+6*sigP)
    pri["Tc_k1"]=NormalPrior(T0seed,0.05,T0seed-0.2,T0seed+0.2)
    pri["b_k1"]=UniformPrior(0.0,1.0); pri["rr_k1"]=UniformPrior(0.003,0.20)
    pri["offset_K2"]=NormalPrior(0.0,1e-3,-5e-3,5e-3); pri["jitter_K2"]=LogUniformPrior(1e-5,5e-3)
    pri["q1_K2"]=UniformPrior(0.0,1.0); pri["q2_K2"]=UniformPrior(0.0,1.0); pri["rho_s"]=NormalPrior(0.71,0.2,0.1,5.0)
    params=Params(;max_kplanet=1,planet_modes=[PM_ONLY],instruments=InstrumentConfig(pm=["K2"]),data=data,M_s=1.04,R_s=1.137,
        parametrization=ParametrizationConfig(time=:Tc,geom=:b_rr,use_rho_s=true),priors=pri,stability=:none)
    res=sample_map(NereusTarget(params,data;unconstrained=true); n_starts=8, maxiter=4000)
    θ=Theta(params); for (nm,v) in zip(res.param_names,res.x_map); set_param!(θ,nm,v); end
    model,_=phot_predictions(θ,data); blk=params.layout.planet_blocks[1]
    P=planet_P(θ,1); Tc=planet_time_anchor(θ,1); rr=θ.values[blk.r]; bb=θ.values[blk.b]
    (model=model, P=P, Tc=Tc, rr=rr, b=bb, depth=rr^2, conv=res.converged, railed=res.railed)
end

# === iterative fit -> subtract -> BLS ===
stages = NamedTuple[]
resid = copy(flat)
# stage 0: raw flat, expect b
push!(stages, (name="flat (search b)", flux=copy(resid), pgram=bls_pgram(resid, periods), fit=nothing))
# b
fb = fit_transit(resid, 4.158517, tcscan(resid,4.158517))
@printf("b: P=%.6f Tc=%.4f rr=%.4f depth=%dppm b=%.2f conv=%s rail=%s\n", fb.P,fb.Tc,fb.rr,round(Int,fb.depth*1e6),fb.b,fb.conv,fb.railed)
stages[1] = merge(stages[1], (fit=fb,))
resid = resid .- fb.model .+ 1.0
push!(stages, (name="−b (search d)", flux=copy(resid), pgram=bls_pgram(resid, periods), fit=nothing))
# d
pd = periods[argmax(stages[end].pgram[(periods.>6).&(periods.<12)])]  # peak in d window
# robust: restrict argmax to 6-12
dwin = (periods.>6).&(periods.<12); pd = periods[dwin][argmax(stages[end].pgram[dwin])]
fd = fit_transit(resid, pd, tcscan(resid,pd))
@printf("d: P=%.6f Tc=%.4f rr=%.4f depth=%dppm b=%.2f conv=%s rail=%s\n", fd.P,fd.Tc,fd.rr,round(Int,fd.depth*1e6),fd.b,fd.conv,fd.railed)
stages[2] = merge(stages[2], (fit=fd,))
resid = resid .- fd.model .+ 1.0
push!(stages, (name="−b−d (search e)", flux=copy(resid), pgram=bls_pgram(resid, periods), fit=nothing))
# e
# search [0.5,3.0] — INCLUDES the 2.37=3×Pe alias; a/Rs grid should still pick 0.79
ewin = (periods.>0.5).&(periods.<3.0); pe = periods[ewin][argmax(stages[end].pgram[ewin])]
fe = fit_transit(resid, pe, tcscan(resid,pe); sigP=8e-4)
@printf("e: P=%.6f Tc=%.4f rr=%.4f depth=%dppm b=%.2f conv=%s rail=%s\n", fe.P,fe.Tc,fe.rr,round(Int,fe.depth*1e6),fe.b,fe.conv,fe.railed)
stages[3] = merge(stages[3], (fit=fe,))
resid = resid .- fe.model .+ 1.0
push!(stages, (name="−b−d−e (clean?)", flux=copy(resid), pgram=bls_pgram(resid, periods), fit=nothing))

# === figure: col1 periodograms, col2 folds with model overlay ===
Pe_fund = fe.P
refs = [(fb.P,"b"),(fd.P,"d"),(fe.P,"e"),(3*Pe_fund,"3×e")]
fig=Figure(size=(1500,1150))
for (i,st) in enumerate(stages)
    axp=Axis(fig[i,1]; xlabel = i==4 ? "log₁₀ Period [d]" : "", ylabel="BLS SNR",
             xgridvisible=false, ygridvisible=false)
    for (Pr,lab) in refs
        vlines!(axp,[log10(Pr)]; color=(:gray,0.4), linestyle=:dot, linewidth=1)
    end
    lines!(axp, log10.(periods), st.pgram; color=:steelblue, linewidth=1.2)
    if st.fit !== nothing
        fr=st.fit; jfit=argmin(abs.(periods .- fr.P)); snrf=st.pgram[jfit]
        scatter!(axp, [log10(fr.P)], [snrf]; color=:red, markersize=12)
        text!(axp, 0.015, 0.95; text="$(st.name)\nfit P=$(round(fr.P,digits=4))d  BLS SNR=$(round(snrf,digits=1))",
              space=:relative, align=(:left,:top), fontsize=13)
    else
        jmax=argmax(st.pgram); Ppk=periods[jmax]
        scatter!(axp, [log10(Ppk)], [st.pgram[jmax]]; color=:orange, markersize=11)
        text!(axp, 0.015, 0.95;
              text="$(st.name)\nresidual peak P=$(round(Ppk,digits=3))d (≈$(round(Ppk/fe.P,digits=1))×Pₑ e-subtraction alias), SNR=$(round(st.pgram[jmax],digits=1))\npoint-estimate e not fully removed; resid std=$(round(Int,std(resid)*1e6))ppm",
              space=:relative, align=(:left,:top), fontsize=11)
    end
    # col2: fold the residual-before on the fitted ephemeris, overlay model
    if st.fit !== nothing
        fr=st.fit; ph=@. mod((t-fr.Tc)/fr.P+0.5,1.0)-0.5
        nm = argmin(abs.([fr.P-4.1585, fr.P-9.0292, fr.P-0.7894]))==1 ? "b" :
             argmin(abs.([fr.P-4.1585, fr.P-9.0292, fr.P-0.7894]))==2 ? "d" : "e"
        win = nm=="e" ? 0.12 : 0.055
        sel=abs.(ph).<win
        axf=Axis(fig[i,2]; xlabel = i==4 ? "phase" : "", ylabel="flux", xgridvisible=false, ygridvisible=false)
        scatter!(axf, ph[sel], st.flux[sel]; markersize=3, color=(:steelblue,0.4))
        xb,yb,eb = Nereus.bin_phasefold(ph[sel], st.flux[sel]; target= nm=="e" ? 25 : 14, xlo=-win, xhi=win)
        errorbars!(axf, xb,yb,eb; color=(:red,0.45), whiskerwidth=0, linewidth=1.5)
        scatter!(axf, xb,yb; markersize=9, color=:red)
        o=sortperm(ph[sel]); lines!(axf, ph[sel][o], fr.model[sel][o]; color=:black, linewidth=2)
        lo=minimum(yb.-eb); hi=maximum(yb.+eb); pad=0.3*(hi-lo)+5e-5; ylims!(axf,lo-pad,hi+pad)
        text!(axf,0.02,0.05; text="$nm fit: rr=$(round(fr.rr,digits=4))  depth=$(round(Int,fr.depth*1e6))ppm",
              space=:relative, align=(:left,:bottom), fontsize=12)
    else
        axf=Axis(fig[i,2]; xgridvisible=false, ygridvisible=false)
        text!(axf,0.5,0.5; text="final residual\nstd=$(round(Int,std(resid)*1e6))ppm", space=:relative, align=(:center,:center), fontsize=14)
    end
end
out=joinpath(@__DIR__,"WASP47_k2_subtract_bls.png"); save(out,fig; px_per_unit=2); println("saved $out")
