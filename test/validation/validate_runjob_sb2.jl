#!/usr/bin/env julia
# End-to-end run_job validation for an SB2 double-lined binary + circumprimary
# planet: drives the PRODUCTION dispatcher with a `component` RV column and
# planet_modes ["BINARY_RV","RV_ONLY"], and asserts (a) the job returns proper
# JSON with status ok, (b) K_A/K_B/K_p are recovered in the fitted table,
# (c) the derived table carries the binary's mass ratio + minimum masses, and
# (d) the SB2-aware plots are emitted.
using Nereus, Printf, Random

out = joinpath(@__DIR__, "..", "..", "results", "runjob_sb2_check")
rm(out; force=true, recursive=true); mkpath(out)
rng = MersenneTwister(11)

# ---- truth: SB2 binary (K_A,K_B anti-phase) + circumprimary RV planet -------
P_BIN,TC_BIN,E_BIN,W_BIN = 41.3,5.7,0.15,0.5
K_A,K_B = 820.0,1240.0
P_PL,TC_PL,K_PL = 7.83,2.1,24.0
GAMMA,JIT = 15.0,6.0     # jitter clearly off the 0 bound → assess_fit rail check passes
function kg(t,P,Tc,e,w)
    f_tc=π/2-w; E_tc=2*atan(sqrt((1-e)/(1+e))*tan(f_tc/2)); M_tc=E_tc-e*sin(E_tc)
    M=2π*(t-Tc)/P+M_tc; E=M; for _ in 1:80; E-=(E-e*sin(E)-M)/(1-e*cos(E)); end
    f=2*atan(sqrt((1+e)/(1-e))*tan(E/2)); cos(f+w)+e*cos(w)
end
nep=55; tep=sort(rand(rng,nep).*300.0)
t=vcat(tep,tep); comp=vcat(fill(1,nep),fill(2,nep))
rv=[comp[i]==1 ? GAMMA+K_A*kg(t[i],P_BIN,TC_BIN,E_BIN,W_BIN)+K_PL*kg(t[i],P_PL,TC_PL,0.1,1.0) :
                 GAMMA-K_B*kg(t[i],P_BIN,TC_BIN,E_BIN,W_BIN) for i in eachindex(t)] .+ JIT.*randn(rng,2nep)
σ=fill(3.0,2nep)
@printf("synthetic SB2: %d RV (%d primary + %d secondary) | K_A=%.0f K_B=%.0f K_p=%.0f\n",
        2nep, nep, nep, K_A, K_B, K_PL)

cfg = Dict(
  "version"=>"1.0", "seed"=>1, "output_dir"=>out,
  "star"=>Dict("M_s"=>1.0, "R_s"=>1.0),
  "data"=>Dict("rv"=>Dict("values"=>Dict(
      "bjd"=>t, "rv"=>rv, "rv_err"=>σ,
      "instrument"=>fill("SB2SPEC", 2nep),
      "component"=>comp))),                        # ← the reserved SB2 component column
  "model"=>Dict("max_kplanet"=>2, "planet_modes"=>["BINARY_RV","RV_ONLY"],
     "parametrization"=>Dict("mass"=>"K_driven","time"=>"Tc","ew"=>"sesinw"),
     "stability"=>"none"),
  "priors"=>Dict(
     "P_k1"=>Dict("type"=>"NormalPrior","args"=>[P_BIN,0.05,P_BIN-0.5,P_BIN+0.5]),
     "K_A_k1"=>Dict("type"=>"UniformPrior","args"=>[300.0,1500.0]),
     "K_B_k1"=>Dict("type"=>"UniformPrior","args"=>[300.0,2200.0]),
     "Tc_k1"=>Dict("type"=>"NormalPrior","args"=>[TC_BIN,0.3,TC_BIN-2,TC_BIN+2]),
     "sesinw_k1"=>Dict("type"=>"UniformPrior","args"=>[-0.7,0.7]),
     "secosw_k1"=>Dict("type"=>"UniformPrior","args"=>[-0.7,0.7]),
     "P_k2"=>Dict("type"=>"NormalPrior","args"=>[P_PL,0.02,P_PL-0.2,P_PL+0.2]),
     "K_k2"=>Dict("type"=>"UniformPrior","args"=>[0.0,80.0]),
     "Tc_k2"=>Dict("type"=>"NormalPrior","args"=>[TC_PL,0.2,TC_PL-1,TC_PL+1]),
     "sesinw_k2"=>Dict("type"=>"UniformPrior","args"=>[-0.5,0.5]),
     "secosw_k2"=>Dict("type"=>"UniformPrior","args"=>[-0.5,0.5])),
  "noise_models"=>[],
  "sampler"=>Dict("name"=>"ptemcee","kwargs"=>Dict(
      "n_temps"=>8,"n_walkers"=>48,"n_steps"=>3500,"n_burnin"=>1800,"show_progress"=>false)),
  "output"=>Dict("plots"=>["rv_timeseries","rv_phasefold"], "save_pdf"=>false),
)

println("\nrun_job (SB2 via production dispatcher)...")
s = Nereus.run_job(cfg)

ok = true
chk(n,c) = (global ok; ok &= c; @printf("  [%s] %s\n", c ? "PASS" : "FAIL", n))

chk("status ok", get(s,"status","")=="ok")
fp = get(get(s,"fitted",Dict()),"parameters",Dict())
val(k) = get(get(fp,k,Dict()),"value",NaN)
KA,KB,KP = val("K_A_k1"), val("K_B_k1"), val("K_k2")
@printf("\nrecovery: K_A=%.1f (t %.0f) | K_B=%.1f (t %.0f) | K_p=%.1f (t %.0f)\n",
        KA,K_A,KB,K_B,KP,K_PL)
chk("K_A recovered (±15)", isfinite(KA) && abs(KA-K_A)<15)
chk("K_B recovered (±15)", isfinite(KB) && abs(KB-K_B)<15)
chk("K_p recovered (±6)",  isfinite(KP) && abs(KP-K_PL)<6)

# derived table must carry the binary's SB2 quantities (mass ratio + min masses)
dp = get(get(s,"derived",Dict()),"parameters",Dict())
haskey_any(sub) = any(k->occursin(sub,k), keys(dp))
@printf("\nderived keys sample: %s\n", join(first(collect(keys(dp)), min(8,length(keys(dp)))), ", "))
chk("derived has mass_ratio_q", haskey_any("mass_ratio_q"))
chk("derived has M_A_sini3",    haskey_any("M_A_sini3"))
chk("derived has M_B_sini3",    haskey_any("M_B_sini3"))

# SB2-aware plots emitted (tables are small LaTeX/CSV → 400-byte floor)
enz(rel) = (f=joinpath(out,rel); isfile(f) && filesize(f)>400)
println("\n=== emitted SB2 plots ===")
for (lab,rel) in [("SB2 timeseries","plots/models/rv_sb2_timeseries.png"),
                  ("double-lined binary fold","plots/models/rv_sb2_binary_fold_P1.png"),
                  ("planet fold (P2)","plots/models/RV_phasefold_P2.png")]
    e = enz(rel); @printf("  %-26s %s\n", lab, e ? "✓" : "MISSING"); chk("$lab emitted", e)
end
# science tables in every format
for fmt in ("csv","json","tex")
    chk("fitted+derived .$fmt", enz("tables/fitted.$fmt") && enz("tables/derived.$fmt"))
end

println(ok ? "\n✅ RUN_JOB SB2 VALIDATION PASS" : "\n❌ RUN_JOB SB2 — FAILURES ABOVE")
