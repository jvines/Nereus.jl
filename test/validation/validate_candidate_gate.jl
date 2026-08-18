#!/usr/bin/env julia
# Validate the candidate promotion gate (conjunction of vetoes).
#   A. a coherent planet, period clear of rotation/indicators → :candidate
#   B. an activity signal at P_rot, incoherent, with an indicator counterpart
#      → :reject, and the coherence/indicator/rotation vetoes all fire
#   C. physicality/detection unit checks
using Nereus, Statistics, Printf, Random
Random.seed!(20260711)

# seasonal sampling, ~12 yr
t = Float64[]; sid = Int[]
for k in 0:11; t0=k*365.25+20; append!(t, sort!(t0 .+ 130 .* rand(16))); append!(sid, fill(k+1,16)); end
n = length(t); err = fill(1.5, n)

const P_PL, K_PL = 40.0, 6.0          # coherent planet
const P_ROT      = 25.0               # rotation
const A_ACT      = 5.0                # activity amplitude (RV + BIS)

# incoherent activity at P_rot: phase reshuffled every ~3 rotations (90 d blocks)
blk(ti)=floor(Int,ti/90)+1
φb = Dict(b=>2π*rand() for b in unique(blk.(t)))
act = [A_ACT*sin(2π*t[i]/P_ROT + φb[blk(t[i])]) for i in 1:n]

planet = K_PL .* sin.(2π .* t ./ P_PL .+ 0.7)          # coherent
rv  = 12.0 .+ planet .+ act .+ err .* randn(n)          # RV = γ + planet + activity + noise
bis = act .+ 1.0 .* randn(n)                            # BIS traces the 25 d activity

ok = true
chk(nm,c)=(global ok; ok &= c; @printf("  [%s] %s\n", c ? "PASS" : "FAIL", nm))

println("=== A. coherent planet at 40 d (activity 25 d subtracted for the coherence test) ===")
va = vet_candidate(t, rv, err, P_PL, K_PL, 0.05;
                   residual = planet .+ err .* randn(n),        # other-signals-removed residual
                   indicators = Dict("BIS"=>bis), P_rot = P_ROT,
                   occupancy = 0.99, dlnZ = 40.0, k_floor = 1.0)
@printf("decision=%s  det=%s ind=%s rot=%s alias=%s coh=%s phys=%s\n",
        va.decision, va.detection, va.activity_indicator, va.rotation, va.alias,
        va.coherence, va.physicality)
isempty(va.reasons) || println("  reasons: ", join(va.reasons, " | "))
chk("A → :candidate",              va.decision === :candidate)
chk("A coherence pass",            va.coherence === :pass)
chk("A indicator pass (25≠40)",    va.activity_indicator === :pass)
chk("A rotation pass (40≠25)",     va.rotation === :pass)

println("\n=== B. activity signal at P_rot=25 d (incoherent, BIS counterpart) ===")
vb = vet_candidate(t, rv, err, P_ROT, A_ACT, 0.05;
                   residual = act .+ err .* randn(n),
                   indicators = Dict("BIS"=>bis), P_rot = P_ROT,
                   occupancy = 0.99, dlnZ = 40.0, k_floor = 1.0)
@printf("decision=%s  det=%s ind=%s rot=%s alias=%s coh=%s phys=%s\n",
        vb.decision, vb.detection, vb.activity_indicator, vb.rotation, vb.alias,
        vb.coherence, vb.physicality)
println("  reasons: ", join(vb.reasons, " | "))
chk("B → :reject",                 vb.decision === :reject)
chk("B coherence fail (incoherent)", vb.coherence === :fail)
chk("B indicator fail (BIS@25)",   vb.activity_indicator === :fail)
chk("B rotation fail (=P_rot)",    vb.rotation === :fail)

println("\n=== C. unit checks ===")
vc = vet_candidate(t, rv, err, P_PL, K_PL, 0.999;                # railed eccentricity
                   residual = planet, indicators=Dict("BIS"=>bis), P_rot=P_ROT, k_floor=1.0)
chk("C railed e → physicality fail → :reject", vc.physicality === :fail && vc.decision === :reject)
vd = vet_candidate(t, rv, err, P_PL, K_PL, 0.05;                 # low occupancy
                   residual = planet, indicators=Dict("BIS"=>bis), P_rot=P_ROT,
                   occupancy = 0.2, k_floor=1.0)
chk("C low occupancy → detection fail → :reject", vd.detection === :fail && vd.decision === :reject)

println(ok ? "\n✅ CANDIDATE GATE VALIDATION PASS" : "\n❌ CANDIDATE GATE — FAILURES ABOVE")
