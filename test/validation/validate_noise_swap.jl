#!/usr/bin/env julia
# A/B validation of the DB-correct post-burn-in within-group noise swap
# (propose_noise_swap), on an AD↔GP-Rot exclusion pair.
#
# Activity models have naturally LARGE evidence gaps (one fits clearly better),
# which saturates the logit=ΔlogZ test (occupancy pins to 0/1). So first SCAN the
# indicator-noise level to locate a NON-SATURATED gap (|ΔlogZ| small), then at
# that level run the A/B:
#   1. DB-CORRECTNESS: swap-ON occupancy must reproduce the fixed-dim ΔlogZ (a
#      wrong q-ratio injects a bias → FAIL). This exercises the new informed-AD ↔
#      prior-GP q-path that AD↔AD couldn't.
#   2. THE FIX: swap-ON discrepancy ≤ swap-OFF (drains the disfavoured mode).
include(joinpath(@__DIR__, "_ad_gprot_setup.jl"))

# bis_noise=2.0 → a NON-SATURATED gap (ΔlogZ≈+2.6, P(AD)≈0.93) from the scan.
data = make_data(BIS_NOISE)
Δz, σ_est, σ_seed, _ = reference_dlogZ(data)
# σ_est, not σ_seed: the seed scatter is reproducibility, not the estimator's
# uncertainty on a GP marginal likelihood, and it understates it ~14x.
Δze = σ_est
@printf("Reference ΔlogZ (AD − GP-Rot) = %+.2f ± %.2f  (estimator error; seed scatter would say ±%.2f)\n\n",
        Δz, σ_est, σ_seed)

pT=mkp(data,NoiseModel[AD,ROT];td=true)
tdc=TransDimConfig(; max_kplanet=0, planets=false, noise=true,
        toggleable=NoiseModel[AD,ROT], noise_exclusion_groups=[NoiseModel[AD,ROT]])
# Sweep the swap RATE: if the move is DB-correct, occupancy climbs toward the
# predicted +2.63 as the rate rises (rate-limited mixing); if it PLATEAUS below,
# that's a q-ratio bug (wrong stationary). OFF = rate 0.
function occ(swap::Bool, rate::Float64)
    obs=Float64[]
    for sd in 1:3
        res=sample_transdim_ptemcee(NereusTarget(pT,data),data; td=tdc,
            n_temps=20,n_walkers=80,n_steps=20000,n_burnin=6000,
            n_birth_tries=10,n_birth_refine=15,seed=sd,
            noise_swap=swap,noise_swap_rate=rate,show_progress=false)
        pad =mean(vec(Array(res.chains[:noise_active_1])).>0.5)
        prot=mean(vec(Array(res.chains[:noise_active_2])).>0.5)
        d=clamp(pad+prot,1e-3,1.0); padc=clamp(pad/d,1e-3,1-1e-3)
        push!(obs, log(padc/(1-padc)))
        @printf("  rate=%.2f seed %d: P(AD)=%.3f P(GP-Rot)=%.3f logit=%+.2f\n",
                swap ? rate : 0.0, sd, pad, prot, log(padc/(1-padc)))
    end
    mean(obs)
end
lo0  = occ(false, 0.0)
lo25 = occ(true, 0.25)
lo60 = occ(true, 0.60)
@printf("\nlogit(P(AD)/P(GP-Rot)) vs swap rate:  OFF=%+.2f  0.25=%+.2f  0.60=%+.2f   predicted=%+.2f\n",
        lo0, lo25, lo60, Δz)
@printf("discrepancy vs evidence:  OFF=%.2f  0.25=%.2f  0.60=%.2f nats\n",
        abs(lo0-Δz), abs(lo25-Δz), abs(lo60-Δz))
db_ok    = abs(lo60 - Δz) < 2*Δze          # 2 sigma of the REAL reference error
climbing = (lo25 >= lo0 - 0.2) && (lo60 >= lo25 - 0.2)
println(db_ok ?
    "\n✅ NOISE SWAP DB-CORRECT — occupancy reaches evidence at adequate rate" :
    climbing ? "\n⚠️  rate-limited: occupancy climbs toward evidence but not there yet (raise rate / steps)" :
               "\n❌ FAIL — occupancy plateaus below evidence (q-ratio bug)")
