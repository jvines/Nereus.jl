using Nereus, LinearAlgebra, Random, Printf
import Nereus: activity_gp_covariance_blocked, activity_gp_joint_logpdf_lowrank
rng = MersenneTwister(11)
for (N, C) in [(114,5), (60,3), (40,2), (114,4)]
    t = sort(rand(rng, N)) .* 90.0
    amp, P, λe, λp = 1.0, 8.6, 47.0, 0.6
    a = randn(rng, C); b = randn(rng, C)
    # dense reference: build Σ = cov + diag noise, exact logpdf
    y = randn(rng, C*N) .* 2.0
    d = (0.5 .+ rand(rng, C*N)).^2            # per-point variances
    Σ = activity_gp_covariance_blocked(t, a, b, amp, P, λe, λp)
    for i in 1:C*N; Σ[i,i] += d[i]; end
    F = cholesky(Symmetric(Σ); check=false)
    dense = -0.5*(dot(y, F\y) + logdet(F) + C*N*log(2π))
    lowr = activity_gp_joint_logpdf_lowrank(t, a, b, amp, P, λe, λp, y, d)
    @printf("N=%3d C=%d:  dense=%.6f  lowrank=%.6f  Δ=%.2e %s\n",
            N, C, dense, lowr, abs(dense-lowr),
            abs(dense-lowr) < 1e-6*abs(dense) ? "✓" : "✗ MISMATCH")
end
# benchmark N=114 C=5
t = sort(rand(rng,114)).*90.0; a=randn(rng,5); b=randn(rng,5)
y=randn(rng,570).*2; d=(0.5 .+ rand(rng,570)).^2
Σ=activity_gp_covariance_blocked(t,a,b,1.0,8.6,47.0,0.6); for i in 1:570; Σ[i,i]+=d[i]; end
densef()= (F=cholesky(Symmetric(copy(Σ));check=false); -0.5*(dot(y,F\y)+logdet(F)+570*log(2π)))
densef(); activity_gp_joint_logpdf_lowrank(t,a,b,1.0,8.6,47.0,0.6,y,d)  # warm
td=@elapsed for _ in 1:200; densef(); end
tl=@elapsed for _ in 1:200; activity_gp_joint_logpdf_lowrank(t,a,b,1.0,8.6,47.0,0.6,y,d); end
@printf("\nN=114 C=5 timing:  dense=%.0f μs  lowrank=%.0f μs  → %.1f× faster\n",
        1e6*td/200, 1e6*tl/200, td/tl)
