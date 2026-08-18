#!/usr/bin/env julia
# Phase 2 — does the 12-coefficient ActivityDecorrelation OVER-ABSORB the
# planet on HD 18599's time sampling + indicators?
#
# Controlled injection-recovery on the REAL HD 18599 RV epochs, instruments,
# errors, and REAL activity indicators (AD format, scaled to the RV range):
#
#   RV_inj(t) = Keplerian(P=4.1375 d, K=11 m/s, e=0.25)        [the planet]
#             + Σ_j b_j · I_j(t)                                [activity ≡ linear-AD, TRUE model]
#             + N(0, rv_err²)                                   [white]
#
# The activity is EXACTLY a linear combination of the indicators, so AD is the
# correct generative activity model and gets every fair chance. Recover K under:
#   (i)   white       — no decorrelation (control; must return K≈11)
#   (ii)  BIS-only AD  — 1 indicator  (3 per-instrument coeffs)
#   (iii) 4-indic AD   — the suspect  (12 per-instrument coeffs)
# ACT=0 variant (no injected activity) separates pure overfitting from
# activity-absorption. K_rec ≪ 11 for (iii) ⇒ over-absorption at the 2:1
# P_rot/P_orb alias; the two-paper K≈11 consensus is then the honest value.

using Nereus, MCMCChains, DelimitedFiles, Statistics, Printf, Random

const RV_FILE = joinpath(@__DIR__, "..", "data", "hd18599.csv")
const P_REF, T0_REF = 4.1374685534602405, 2.4583545857470357e6
const K_INJ, E_INJ, W_INJ = 11.0, 0.25, 0.7      # ω rad
const ACT  = parse(Float64, get(ENV, "ACT", "1.0"))   # activity amplitude scale (0 ⇒ clean)
const PAPER_INST = Set(["HARPS_PRE", "HARPS_POST", "FEROS"])

# ---- real epochs / instruments / errors / indicators (joint-script prep) ----
raw = readdlm(RV_FILE, ',', Any, '\n'; header=true); dm = raw[1]
keep=Int[]; inst_str=String[]
for i in 1:size(dm,1)
    ins=strip(String(dm[i,16])); prov=strip(String(dm[i,17]))
    (ins=="HARPS_POST" && prov=="ESO_PHASE3") && continue
    ins in PAPER_INST || continue; push!(keep,i); push!(inst_str,ins)
end
dm=dm[keep,:]; bjd=Float64.(dm[:,1]); rv0=Float64.(dm[:,2]); rverr=Float64.(dm[:,3])
inst_names=sort!(unique(inst_str)); imap=Dict(n=>i for (i,n) in enumerate(inst_names))
rv_inst=[imap[s] for s in inst_str]
let kc=trues(length(bjd))
    for id in unique(rv_inst)
        idx=findall(==(id),rv_inst); m=median(rv0[idx]); thr=5*max(1.4826*median(abs.(rv0[idx].-m)),1e-6)
        for k in idx; abs(rv0[k]-m)>thr && (kc[k]=false); end
    end
    global bjd=bjd[kc]; global rverr=rverr[kc]; global rv_inst=rv_inst[kc]; global dm=dm[kc,:]
end
col=Dict(:bis=>4,:fwhm=>6,:halpha=>10,:logrhk=>12)
function load_adfmt(cv)
    v=Float64[let x=dm[i,cv]; (x==="" || (x isa AbstractString && strip(x)=="")) ? 0.0 : Float64(x) end for i in 1:size(dm,1)]
    for id in unique(rv_inst)
        idx=findall(==(id),rv_inst); fin=filter(k->isfinite(v[k]),idx); isempty(fin)&&continue
        μ=mean(v[fin]); for k in fin; v[k]-=μ; end
    end
    v   # mean-subtracted per instrument; raw (NOT scaled to RV) so injected activity amplitude is controlled below
end
adkey=Dict(:bis=>"bisector_span",:fwhm=>"fwhm_AD",:halpha=>"halpha_AD",:logrhk=>"log_rhk_AD")
chs=[:bis,:fwhm,:halpha,:logrhk]
Iraw=Dict(ch=>load_adfmt(col[ch]) for ch in chs)
# scale each indicator to unit std so the injected b_j have comparable footing
Inorm=Dict(ch=>(s=std(Iraw[ch]); s>0 ? Iraw[ch]./s : Iraw[ch]) for ch in chs)
indicators=Dict{String,Vector{Float64}}(); ierrs=Dict{String,Vector{Float64}}()
for ch in chs; indicators[adkey[ch]]=Inorm[ch]; ierrs[adkey[ch]]=fill(1.0,length(bjd)); end

n=length(bjd)
@printf("Loaded %d real RV epochs (%s), %d indicators. ACT=%.2f\n",
        n, join(inst_names,","), length(chs), ACT)

# true activity = linear combo of (unit-std) indicators; amplitude ~ a few m/s
b_true = Dict(:bis=>3.0, :fwhm=>2.0, :halpha=>1.5, :logrhk=>2.5)   # m/s per unit-std indicator
rv_act = zeros(n)
for ch in chs; rv_act .+= ACT .* b_true[ch] .* Inorm[ch]; end
@printf("injected activity RMS = %.2f m/s ;  planet K = %.1f m/s\n", std(rv_act), K_INJ)

# ---- Keplerian RV (Tc anchor) ----
function kepler_rv(t, P, Tc, K, e, w)
    n_obs=length(t); out=zeros(n_obs)
    # mean anomaly from Tc: true anomaly at transit f_tc = π/2 - w
    f_tc=π/2 - w
    E_tc=2*atan(sqrt((1-e)/(1+e))*tan(f_tc/2))
    M_tc=E_tc - e*sin(E_tc)
    for i in 1:n_obs
        M=2π*(t[i]-Tc)/P + M_tc
        E=M; for _ in 1:60; E -= (E-e*sin(E)-M)/(1-e*cos(E)); end
        f=2*atan(sqrt((1+e)/(1-e))*tan(E/2))
        out[i]=K*(cos(f+w)+e*cos(w))
    end
    out
end
kep = kepler_rv(bjd, P_REF, T0_REF, K_INJ, E_INJ, W_INJ)

function recover(adcfg::Vector{Symbol}, label, seed)
    rng=MersenneTwister(1000+seed)
    rv = kep .+ rv_act .+ rverr .* randn(rng, n)        # γ_true = 0
    data=Data(; t_rv=bjd, rv=rv, rv_err=rverr, rv_inst=rv_inst,
                indicators=indicators, indicator_errs=ierrs)
    ic=InstrumentConfig(rv=inst_names)
    rvmax=maximum(abs,rv)
    pri=Dict{String,PriorSpec}(
        "n_p"=>FixedPrior(1.0),
        "P_k1"=>NormalPrior(P_REF,1e-3,P_REF-0.01,P_REF+0.01),
        "Tc_k1"=>NormalPrior(T0_REF,0.05,T0_REF-0.3,T0_REF+0.3),
        "K_k1"=>UniformPrior(0.0,50.0),
        "sesinw_k1"=>UniformPrior(-0.7,0.7), "secosw_k1"=>UniformPrior(-0.7,0.7))
    for nm in inst_names; pri["gamma_$nm"]=UniformPrior(-3rvmax,3rvmax); pri["sigma_$nm"]=LogUniformPrior(1e-3,30.0); end
    nmods = NoiseModel[]
    if !isempty(adcfg)
        push!(nmods, ActivityDecorrelation(indicators=[adkey[c] for c in adcfg]))
    end
    params=Params(; max_kplanet=1, planet_modes=[RV_ONLY], instruments=ic, data=data,
        parametrization=ParametrizationConfig(time=:Tc), priors=pri,
        noise_models=nmods, stability=:none)
    tgt=NereusTarget(params, data)
    res=sample_ptemcee(tgt, data; n_temps=10, n_walkers=48, n_steps=7000,
        n_burnin=3000, seed=seed, init_strategy=:prior, show_progress=false)
    K=vec(Array(res.chains[:,:K_k1,:]))
    (median(K), quantile(K,0.16), quantile(K,0.84))
end

configs=[(Symbol[],"white (no AD)"), ([:bis],"BIS-only AD (3 coef)"),
         (chs,"4-indicator AD (12 coef)")]
@printf("\n%-26s  %-22s\n", "config", "K_rec (med [16,84])  vs K_inj=11")
for (cfg,lab) in configs
    Ks=Float64[]; los=Float64[]; his=Float64[]
    for sd in 1:3
        m,lo,hi=recover(cfg,lab,sd); push!(Ks,m); push!(los,lo); push!(his,hi)
    end
    @printf("%-26s  K=%5.2f  [%.2f, %.2f]  (3-seed mean of medians)\n",
            lab, mean(Ks), mean(los), mean(his))
end
