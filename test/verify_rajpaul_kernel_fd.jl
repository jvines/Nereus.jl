#!/usr/bin/env julia
# Finite-difference verification of the Rajpaul-GP derivative blocks.
#
# The Rajpaul+ 2015 framework couples channels to a latent G(t) and its
# derivative Ġ(t), so the covariance needs the four blocks
#   k_GG(t,t') = k(τ),     k_GĠ(t,t')  = ∂k/∂t',
#   k_ĠG(t,t') = ∂k/∂t,    k_ĠĠ(t,t') = ∂²k/∂t∂t',
# with k the unit-amplitude QP kernel
#   k(τ) = exp(−τ²/(2λe²) − sin²(πτ/P)/(2λp²)),  τ = t − t'.
# `activity_gp_covariance` inlines closed forms (f, f′, f″). One wrong
# sign or factor there biases every multi-channel activity fit, so the
# closed forms are checked here against central finite differences of
# k itself — derivative-free ground truth.

using Nereus
using Test
using Printf

k_qp(t, tp, P, λe, λp) = exp(-(t - tp)^2 / (2λe^2) -
                              sin(π * (t - tp) / P)^2 / (2λp^2))

function fd_blocks(t, tp, P, λe, λp; h = 1e-5, h2 = 1e-4)
    k_GG = k_qp(t, tp, P, λe, λp)
    # ∂/∂t' and ∂/∂t central differences
    k_GdG = (k_qp(t, tp + h, P, λe, λp) - k_qp(t, tp - h, P, λe, λp)) / (2h)
    k_dGG = (k_qp(t + h, tp, P, λe, λp) - k_qp(t - h, tp, P, λe, λp)) / (2h)
    # ∂²/∂t∂t' via 4-point cross stencil. Larger step: the stencil's
    # roundoff floor is ε/4h² (≈3e-7 absolute at h=1e-5, above the test
    # tolerance); h2 = ε^(1/4) balances truncation vs roundoff.
    k_dGdG = (k_qp(t + h2, tp + h2, P, λe, λp) - k_qp(t + h2, tp - h2, P, λe, λp) -
              k_qp(t - h2, tp + h2, P, λe, λp) + k_qp(t - h2, tp - h2, P, λe, λp)) / (4h2^2)
    return k_GG, k_GdG, k_dGG, k_dGdG
end

# Recover the four blocks from activity_gp_covariance by probing with
# unit coupling vectors: a=(1,0), b=(0,0) etc. on a 2-point dataset.
function code_blocks(t, tp, P, λe, λp)
    ts = [t, tp]; ch = [:rv, :bis]
    Σ(a1, b1, a2, b2) = activity_gp_covariance(ts, ch, [a1, a2], [b1, b2],
                                                 1.0, P, λe, λp)[1, 2]
    k_GG   = Σ(1.0, 0.0, 1.0, 0.0)   # a_i a_j k_GG
    k_GdG  = Σ(1.0, 0.0, 0.0, 1.0)   # a_i b_j k_GĠ
    k_dGG  = Σ(0.0, 1.0, 1.0, 0.0)   # b_i a_j k_ĠG
    k_dGdG = Σ(0.0, 1.0, 0.0, 1.0)   # b_i b_j k_ĠĠ
    return k_GG, k_GdG, k_dGG, k_dGdG
end

@testset "Rajpaul QP kernel derivative blocks vs finite differences" begin
    P, λe, λp = 12.3, 47.0, 0.55
    npass = 0
    for τ in (-9.7, -3.1, -0.4, 0.7, 2.9, 6.05, 11.8)
        t, tp = 5.0 + τ, 5.0
        fd   = fd_blocks(t, tp, P, λe, λp)
        code = code_blocks(t, tp, P, λe, λp)
        for (name, f, c) in zip(("k_GG", "k_GdG", "k_dGG", "k_dGdG"), fd, code)
            scale = max(abs(f), 1e-8)
            ok = abs(f - c) / scale < 1e-5
            ok || @printf("MISMATCH τ=%.2f %s: fd=%.10g code=%.10g\n",
                           τ, name, f, c)
            @test ok
            npass += ok
        end
    end
    # Symmetry sanity: swapping (i,j) must transpose the derivative blocks
    Σf = activity_gp_covariance([1.0, 4.2], [:rv, :bis],
                                 [1.0, 0.0], [0.0, 1.0], 1.0, P, λe, λp)
    @test Σf[1, 2] ≈ Σf[2, 1]
    println("blocks verified: $npass/28 comparisons pass")
end
