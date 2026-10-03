# Type stability of the semiseparable multi-series GP (src/noise/multiseries_gp.jl).
#
# SSSHO and SSESP build their term lists at run time (one complex SHO term or
# two real ones, depending on Q; nharm+1 ESP harmonics). Those lists used to be
# a Vector{Any} turned into a Tuple, so the generator matrices came back as
# Any and every element of the O(N·r) fill was dynamically dispatched: the ESP
# likelihood cost ~25x the solve it fed. These tests pin the concrete types and
# check that the container change moved no bits.
using Nereus, Test
import ForwardDiff
using Nereus: multiseries_loglike, _terms, _generators, _base_generators,
              _esp_factors, _k_kp_kpp, _kernel_rank, k0, mk2_0,
              SSSHO, SSESP, SSMEP, SSES, SSMatern32

@testset "multiseries GP: concrete term lists and generators" begin
    # deterministic, unsorted, 3-series input with repeated epochs (the
    # ActivityGP layout puts every channel on the RV epochs)
    n = 60
    ep = [3.7 * i + 2.0 * sin(1.3 * i) for i in 1:n]
    t = vcat(ep, ep, ep[1:2:end])
    sid = vcat(fill(1, n), fill(2, n), fill(3, length(1:2:n)))
    N = length(t)
    y = [sin(0.37 * i) + 0.2 * cos(1.7 * i) for i in 1:N]
    v = [0.2 + 0.05 * abs(sin(0.11 * i)) for i in 1:N]
    α = [1.0, 0.6, -0.4]; β = [0.3, -0.2, 0.5]
    ts = sort(t); dt = diff(ts)

    kernels = (SSSHO(1.0, 9.0, 3.0),            # underdamped: one complex term
               SSSHO(1.0, 9.0, 0.3),            # overdamped: two real terms
               SSESP(1.2, 15.0, 8.0, 0.6; nharm = 0),
               SSESP(1.2, 15.0, 8.0, 0.6; nharm = 3),
               SSMEP(1.1, 12.0, 8.0, 0.5), SSES(1.3, 9.0), SSMatern32(1.2, 8.0))

    @testset "inference: $(nameof(typeof(k)))" for k in kernels
        G = @inferred _generators(k, ts, dt)
        @test G isa NTuple{5, Matrix{Float64}}
        ll = @inferred multiseries_loglike(t, y, v, sid, α, β, k)
        @test ll isa Float64 && isfinite(ll)
    end

    # The run-time term lists are now Vectors; the old code iterated the same
    # terms as a Tuple. Same terms, same order => identical generators and
    # identical k, k', k'' (=== on every entry).
    @testset "Vector terms == Tuple terms, bit for bit" begin
        for k in (SSSHO(1.0, 9.0, 3.0), SSSHO(1.0, 9.0, 0.3))
            tv = _terms(k)
            @test tv isa Vector && isconcretetype(eltype(tv))
            @test length(tv) == (k.Q < 0.5 ? 2 : 1)
            @test _kernel_rank(tv) == 2
            @test _base_generators(tv, ts, dt) == _base_generators(Tuple(tv), ts, dt)
            for τ in (-7.3, 0.0, 0.4, 25.0)
                @test _k_kp_kpp(tv, τ) === _k_kp_kpp(Tuple(tv), τ)
            end
        end
        for nh in (0, 1, 3)
            _, t2 = _esp_factors(SSESP(1.2, 15.0, 8.0, 0.6; nharm = nh))
            @test t2 isa Vector && isconcretetype(eltype(t2)) && length(t2) == nh + 1
            G1 = _base_generators(t2, ts, dt); G2 = _base_generators(Tuple(t2), ts, dt)
            @test all(map((a, b) -> all(a .=== b), G1, G2))
            for τ in (-7.3, 0.0, 0.4, 25.0)
                @test _k_kp_kpp(t2, τ) === _k_kp_kpp(Tuple(t2), τ)
            end
        end
    end

    # Dual hyperparameters still promote through the Vector term lists.
    @testset "ForwardDiff Dual through the run-time term lists" begin
        p = ForwardDiff.Dual{Nothing}(9.0, 1.0)
        D = typeof(p)
        for k in (SSSHO(1.0, p, 3.0), SSSHO(1.0, p, 0.3), SSESP(1.2, 15.0, p, 0.6; nharm = 3))
            ll = multiseries_loglike(t, y, v, sid, α, β, k)
            @test ll isa D && isfinite(ForwardDiff.value(ll))
            kf = typeof(k).name.wrapper(map(x -> x isa D ? ForwardDiff.value(x) : x,
                                            ntuple(i -> getfield(k, i), fieldcount(typeof(k))))...)
            # the Dual solve takes the generic (non-BLAS) dot, so ulp-level only
            @test ForwardDiff.value(ll) ≈ multiseries_loglike(t, y, v, sid, α, β, kf) rtol = 1e-12
        end
    end
end
