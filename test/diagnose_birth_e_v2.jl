#!/usr/bin/env julia
# e-birth quality at the CONSOLIDATED joint state (b+d+c, the A/B run's modal
# state). The A/B unfroze e's selections (21→79) but conversion collapsed
# (43%→2.5%). Measure here, with a FRESH state-consistent cache:
#  1. the peak list + e's informed mixture share (post BLS-floor)
#  2. 800 informed slot-3 birth draws: e-share, e ΔlogL distribution
# Good-but-rare draws → tune selection/NTRY. Bad draws → cache content bug.

using Nereus
using DelimitedFiles: readdlm
using Statistics: median, quantile
using Random
using Printf

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))
lc = load_tess_lc(joinpath(DATADIR, "WASP-47_k2_c03_everest_lc.csv"))
runmed(y,w)=(n=length(y);m=similar(y);h=w÷2;for i in 1:n;lo=max(1,i-h);hi=min(n,i+h);m[i]=median(@view y[lo:hi]);end;m)
pt_t=lc.t; pt_f=lc.flux./runmed(lc.flux,151); pt_e=lc.flux_err; pt_i=ones(Int,length(pt_t))
rr=readdlm(joinpath(DATADIR,"WASP47_RVs_combined.csv"),',',Any,'\n';header=true)
raw=rr[1]; hdr=vec(rr[2]); cl(n)=findfirst(==(n),hdr); fcol=cl("flag"); keep=trues(size(raw,1))
for i in 1:size(raw,1); fl=raw[i,fcol]; fl===missing&&continue
    String(strip(string(fl))) in ("transit","transit_night","anomalous")&&(keep[i]=false); end
raw=raw[keep,:]; bjd=Float64.(raw[:,cl("bjd")]); rv=Float64.(raw[:,cl("rv")]); rve=Float64.(raw[:,cl("rv_err")])
istr=String.(raw[:,cl("instrument")]); inames=sort(unique(istr)); i2i=Dict(n=>i for (i,n) in enumerate(inames))
rinst=[i2i[s] for s in istr]

data=Data(;t_rv=bjd,rv=rv,rv_err=rve,rv_inst=rinst,t_phot=pt_t,flux=pt_f,flux_err=pt_e,phot_inst=pt_i)
ic=InstrumentConfig(rv=inames,pm=["K2"]); pri=Dict{String,PriorSpec}()
for k in 1:3
    pri["P_k$k"]=LogUniformPrior(0.5,30.0); pri["K_k$k"]=UniformPrior(0.0,200.0)
    pri["sesinw_k$k"]=UniformPrior(-0.7,0.7); pri["secosw_k$k"]=UniformPrior(-0.7,0.7)
    pri["Tc_k$k"]=UniformPrior(pt_t[1],pt_t[1]+30.0); pri["b_k$k"]=UniformPrior(0.0,1.0); pri["rr_k$k"]=UniformPrior(0.005,0.20)
end
pri["P_k4"]=LogUniformPrior(100.0,1500.0); pri["K_k4"]=UniformPrior(0.0,200.0)
pri["sesinw_k4"]=UniformPrior(-0.7,0.7); pri["secosw_k4"]=UniformPrior(-0.7,0.7)
for n in inames; ri=rv[rinst.==i2i[n]]; isempty(ri)&&continue; ce=median(ri); sp=max(maximum(ri)-minimum(ri),1.0)
    pri["gamma_$n"]=UniformPrior(ce-3sp,ce+3sp); pri["sigma_$n"]=ModJeffreysPrior(0.1,15.0); end
pri["offset_K2"]=NormalPrior(0.0,1e-3,-5e-3,5e-3); pri["jitter_K2"]=LogUniformPrior(1e-5,5e-3)
pri["q1_K2"]=UniformPrior(0.0,1.0); pri["q2_K2"]=UniformPrior(0.0,1.0)
pri["rho_s"]=NormalPrior(0.71,0.3,0.1,5.0)
params=Params(;max_kplanet=4,planet_modes=[RVPM,RVPM,RVPM,RV_ONLY],instruments=ic,data=data,M_s=1.04,R_s=1.137,
    parametrization=ParametrizationConfig(time=:Tc,geom=:b_rr,use_rho_s=true),priors=pri,trend_order=0,
    stability=:none,external_priors=[ExternalPrior(:ecc,NormalPrior(0.0,0.3),true)])

# consolidated state from the A/B run: slots 1(b), 2(d), 4(c) active; 3 free
tds=TransDimState(; max_planets=4, n_noise=0)
θ=Theta{Float64}(params; td=tds)
set_param!(θ,"P_k1",4.1592);  set_param!(θ,"Tc_k1",2456978.8266); set_param!(θ,"rr_k1",0.1088); set_param!(θ,"b_k1",0.68)
set_param!(θ,"K_k1",141.9);   set_param!(θ,"sesinw_k1",0.0); set_param!(θ,"secosw_k1",0.0)
set_param!(θ,"P_k2",9.0304);  set_param!(θ,"Tc_k2",2456979.2718); set_param!(θ,"rr_k2",0.0297); set_param!(θ,"b_k2",0.46)
set_param!(θ,"K_k2",3.6);     set_param!(θ,"sesinw_k2",0.0); set_param!(θ,"secosw_k2",0.0)
set_param!(θ,"P_k4",610.43);  set_param!(θ,"K_k4",30.5)
set_param!(θ,"sesinw_k4",0.0); set_param!(θ,"secosw_k4",0.0)
for n in inames; set_param!(θ,"gamma_$n", median(rv[rinst.==i2i[n]])); set_param!(θ,"sigma_$n",3.0); end
set_param!(θ,"offset_K2",0.0); set_param!(θ,"jitter_K2",1.2e-4)
set_param!(θ,"q1_K2",0.3); set_param!(θ,"q2_K2",0.3); set_param!(θ,"rho_s",0.71)
activate_planet!(θ.td,1); activate_planet!(θ.td,2); activate_planet!(θ.td,4)

# 1. fresh state-consistent BLS cache + peak list
cache=Nereus.BLSCache()
Pk,wk,dk,t0k,durk = Nereus._bls_peaks_filtered(θ, data, 0.5, 30.0, cache)
println("== BLS peak list at b+d+c (slot-3 window [0.5,30]) ==")
for i in eachindex(Pk)
    near = abs(Pk[i]-0.789554)/0.789554<0.01 ? "  <-- e!" : ""
    @printf("  P=%9.5f w=%.3f depth=%6.0fppm t0=%.4f dur=%.4f%s\n", Pk[i],wk[i],dk[i]*1e6,t0k[i],durk[i],near)
end

# 2. 800 informed births from this state
ws = Nereus.PTWorkspace(params, 4, 0; n_obs=length(bjd), n_phot=length(pt_t))
ll(θx)=Nereus.rv_log_likelihood(θx,data,ws)+Nereus.transit_log_likelihood(θx,data,ws)
ll0=ll(θ); @printf("\nlogL(b+d+c) = %.1f\n", ll0)
rng=MersenneTwister(11)
n_e=0; n_tot=0; dll_e=Float64[]; slots=Dict{Int,Int}(); other=Dict{Float64,Int}()
for i in 1:800
    nt,lq = Nereus.propose_planet_birth(θ, rng, Nereus.JointInformedBirth(); data=data)
    isfinite(lq) || continue
    global n_tot += 1
    kb = findfirst(j->nt.td.planet_active[j] && !θ.td.planet_active[j], 1:4)
    kb === nothing && continue
    slots[kb]=get(slots,kb,0)+1
    Pp = planet_P(nt,kb)
    if abs(Pp-0.789554)/0.789554 < 0.01
        global n_e += 1
        push!(dll_e, ll(nt)-ll0)
        if n_e <= 4   # probe: which component is invalid, with what params?
            rvll = Nereus.rv_log_likelihood(nt,data,ws)
            trll = Nereus.transit_log_likelihood(nt,data,ws)
            blk = params.layout.planet_blocks[kb]
            @printf("  probe e-birth: rv_ll=%.1f  tr_ll=%.1f | P=%.5f Tc=%.4f rr=%.4f b=%.3f K=%.2f e1=%.3f e2=%.3f\n",
                    rvll, trll, Pp, planet_time_anchor(nt,kb), nt.values[blk.r], nt.values[blk.b],
                    has_K(blk) ? planet_K(nt,kb) : NaN,
                    nt.values[blk.e1], nt.values[blk.e2])
        end
    else
        key=round(Pp,digits=2); other[key]=get(other,key,0)+1
    end
end
@printf("\n== %d valid births | slot distribution: %s ==\n", n_tot, join(["$k:$v" for (k,v) in sort(collect(slots))],"  "))
@printf("e-period births: %d (%.0f%%)\n", n_e, 100n_e/max(n_tot,1))
if !isempty(dll_e)
    q=quantile(dll_e,[0.1,0.5,0.9])
    @printf("e ΔlogL: q10=%.0f med=%.0f q90=%.0f max=%.0f  frac>+50=%.3f frac>0=%.3f\n",
            q..., maximum(dll_e), count(>(50.0),dll_e)/length(dll_e), count(>(0.0),dll_e)/length(dll_e))
end
top=sort(collect(other); by=x->-x[2])[1:min(8,length(other))]
println("other periods: ", join(["$(k)d×$(v)" for (k,v) in top],"  "))
