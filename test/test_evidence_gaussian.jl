# The four β-path evidence estimators (TI, TI+, SS+, H+) against an analytic
# log Z: a normalized 2-D Gaussian likelihood N(0, I) under a uniform prior on
# the box [-L, L]^d, so Z = (2L)^-d to within erf(L/√2)^d (1e-23 at L = 10).
#
# The draws at each rung are EXACT and independent: the tempered posterior
# ∝ exp(-β|x|²/2) on the box factorizes into one truncated normal per
# coordinate, sampled by rejection. So this tests the estimators' arithmetic
# (the integrals, the ratios, the H+ switch and the sign of a descending
# ladder), not a sampler's mixing.
using Test
using Nereus
using Nereus: EvidenceAccumulator, update_evidence!, evidence_report
using Random

# One coordinate of the β-tempered posterior on [-L, L].
function _tempered_coord(rng, β, L)
    σ = 1 / sqrt(β)
    if σ <= L
        while true                       # normal proposal, acceptance >= 0.68
            z = σ * randn(rng)
            abs(z) <= L && return z
        end
    else
        while true                       # uniform proposal, acceptance >= 0.61
            u = L * (2rand(rng) - 1)
            rand(rng) <= exp(-u^2 / (2σ^2)) && return u
        end
    end
end

# Fill an accumulator with n exact draws per rung, on the ladder `betas` in the
# order given.
function _gaussian_box_evidence(betas; d = 2, L = 10.0, n = 20_000, seed = 1)
    rng = MersenneTwister(seed)
    acc = EvidenceAccumulator(length(betas))
    logL = zeros(length(betas))
    for _ in 1:n
        for (k, β) in enumerate(betas)
            r2 = 0.0
            for _ in 1:d
                r2 += _tempered_coord(rng, β, L)^2
            end
            logL[k] = -r2 / 2 - d / 2 * log(2π)
        end
        update_evidence!(acc, logL, betas)
    end
    return evidence_report(acc, betas)
end

@testset "β-path evidence on a Gaussian in a box" begin
    d, L = 2, 10.0
    truth = -d * log(2L)
    TOL = 0.5
    # Descending, as every PT sampler stores it: β = 1 first, 1e-4 last
    # (σ = 100 against a box of half-width 10, so the last rung is the prior).
    betas = 10 .^ range(0, -4; length = 16)
    rep = _gaussian_box_evidence(betas; d, L)
    est = (TI = rep.ti[1], var"TI+" = rep.ti_plus[1], var"SS+" = rep.ss_plus[1],
           var"H+" = rep.hybrid[1])
    @info "β-path estimators vs analytic log Z" truth est
    for (name, z) in pairs(est)
        @test abs(z - truth) < TOL
    end
    @test maximum(values(est)) - minimum(values(est)) < 1.0

    # The descending-ladder sign: SS+ sums log[Z(β_n)/Z(β_1)] and the TI
    # integrals run from 1 down to β_min, so each has to be negated for a
    # descending ladder. Missing that flips log Z to +6 here. The same draws in
    # ascending order must give the same numbers.
    rep_up = _gaussian_box_evidence(reverse(betas); d, L)
    @test rep_up.ti[1]      ≈ rep.ti[1]      atol = 0.05
    @test rep_up.ti_plus[1] ≈ rep.ti_plus[1] atol = 0.05
    @test rep_up.ss_plus[1] ≈ rep.ss_plus[1] atol = 0.05
    @test all(<(0), values(est))
end
