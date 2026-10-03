# NightlyOffset on a white base: per-night closed form (src/noise/parametric_noise.jl).
#
# Σ = diag(v) + Σ_g σ_g² 1_g 1_gᵀ is block diagonal, one rank-1 block per night,
# so the likelihood is O(N) by the determinant lemma and Sherman–Morrison per
# night instead of a dense Woodbury over all nights. The result is not
# bit-identical to the dense path; these tests hold it to the dense result,
# to a 256-bit reference, and keep the GP-base paths bit-identical.
using Nereus, Test, LinearAlgebra
import ForwardDiff
using Nereus: NightlyOffset, HarmonicBlock, CeleriteSHO, MaternGP, _NightlyFactor,
              _additive_factor, _nightly_factor, _night_group_ids, _channel_suffix,
              _woodbury_white_ll, _woodbury_celerite_ll, _woodbury_matern_ll,
              dense_additive_ll

# The dense factor as it was built before (one name lookup per night).
function nightly_factor_ref(m, theta::Nereus.Theta{T}, t, inst, names) where {T}
    ids, gi = _night_group_ids(m, t, inst, names)
    s = _channel_suffix(m.channel)
    σ = T[theta.values[theta.params.layout.name_to_idx["night_sigma_$(names[ci])$s"]] for ci in gi]
    F = zeros(T, length(t), length(gi))
    for i in eachindex(t)
        ids[i] == 0 || (F[i, ids[i]] = σ[ids[i]])
    end
    return F
end

# 256-bit reference of log N(y | 0, diag(v) + F Fᵀ) for a nightly factor.
function exact_ll(y, v, F::_NightlyFactor)
    setprecision(BigFloat, 256) do
        ng = size(F, 2); a = zeros(BigFloat, ng); b = zeros(BigFloat, ng)
        q = big(0.0); ld = big(0.0)
        for i in eachindex(y)
            yi = big(y[i]); vi = big(v[i]); q += yi^2 / vi; ld += log(vi)
            g = F.ids[i]; g == 0 || (a[g] += 1 / vi; b[g] += yi / vi)
        end
        for g in 1:ng
            s2 = big(F.σ[g])^2; d = 1 + s2 * a[g]; ld += log(d); q -= s2 * b[g]^2 / d
        end
        Float64(-(q + ld + length(y) * log(2 * big(π))) / 2)
    end
end

function setup(models, t, inst, names; truth = Dict{String,Float64}())
    n = length(t)
    data = Nereus.Data(; t_rv = t, rv = zeros(n), rv_err = fill(0.5, n), rv_inst = inst)
    p = Nereus.Params(; max_kplanet = 0, planet_modes = Nereus.PlanetDataSources[],
                      instruments = Nereus.InstrumentConfig(rv = names), data = data,
                      M_s = 1.0, noise_models = Nereus.NoiseModel[models...])
    th = Nereus.Theta{Float64}(p)
    for (k, v) in truth
        haskey(p.layout.name_to_idx, k) && Nereus.set_param!(th, k, v)
    end
    return p, th
end

@testset "NightlyOffset closed form (white base)" begin
    names = ["A", "B"]
    # two instruments, nights of 1-4 points, unsorted across instruments
    t = Float64[]; inst = Int[]
    for (m, base) in ((1, 100.0), (2, 101.3)), g in 1:12
        for j in 1:(1 + mod(3g + m, 4))
            push!(t, base + 2.0 * g + 0.02 * j); push!(inst, m)
        end
    end
    n = length(t)
    y = [2.0 * sin(0.7 * i) + 0.3 * cos(2.1 * i) for i in 1:n]
    v = [0.2 + 0.05 * abs(sin(1.3 * i)) for i in 1:n]
    truth = Dict("night_sigma_A" => 0.8, "night_sigma_B" => 1.7)

    @testset "factor: $(label)" for (label, m) in (("all instruments", NightlyOffset(gap = 0.5)),
                                                   ("instrument B only", NightlyOffset(gap = 0.5, instruments = ["B"])))
        p, th = setup([m], t, inst, names; truth)
        F = _additive_factor([m], th, t, inst, names)
        @test F isa _NightlyFactor{Float64}
        Fd = nightly_factor_ref(m, th, t, inst, names)
        @test size(F) == size(Fd) && all(Matrix(F) .=== Fd) && all(F .=== Fd)
        @test all(_nightly_factor(m, th, t, inst, names) .=== Fd)
        ll = _woodbury_white_ll(y, v, F, 2π)
        ll_dense = _woodbury_white_ll(y, v, Fd, 2π)            # the previous result
        @test isfinite(ll)
        @test abs(ll - ll_dense) <= 1e-13 * abs(ll_dense)
        @test abs(ll - dense_additive_ll(y, v, t, inst, names, [m], th)) <= 1e-10 * abs(ll)
        @test abs(ll - exact_ll(y, v, F)) <= 1e-14 * abs(ll) + 1e-13
        # σ = 0 on every night: the plain white likelihood
        F0 = _NightlyFactor{Float64}(F.ids, zeros(size(F, 2)))
        @test _woodbury_white_ll(y, v, F0, 2π) ≈ -0.5 * sum(@. y^2 / v + log(2π * v)) rtol = 1e-14
    end

    @testset "one long, well-measured night (cancellation)" begin
        # σ² Σ1/v ~ 1e11: the textbook Σy²/v − σ²b²/(1+σ²a) and the dense
        # Woodbury both lose ~6 digits here; the scatter form does not.
        N = 5000
        yl = [0.1 + 1e-4 * sin(0.37 * i) for i in 1:N]
        vl = [1e-8 * (1 + 0.5 * cos(0.11 * i)) for i in 1:N]
        F = _NightlyFactor{Float64}(ones(Int, N), [0.5])
        ref = exact_ll(yl, vl, F)
        @test abs(_woodbury_white_ll(yl, vl, F, 2π) - ref) <= 1e-14 * abs(ref)
        @test abs(_woodbury_white_ll(yl, vl, Matrix(F), 2π) - ref) <= 1e-9 * abs(ref)
    end

    @testset "GP bases and several additive terms keep the dense path" begin
        o = sortperm(t)                       # the GP solvers want sorted times
        t, inst, y, v = t[o], inst[o], y[o], v[o]
        mn = NightlyOffset(gap = 0.5)
        p, th = setup([CeleriteSHO(), mn], t, inst, names;
                      truth = merge(truth, Dict("gp_log_S0" => 0.5, "gp_log_Q" => 1.0,
                                                "gp_log_omega0" => 0.3)))
        F = _additive_factor([mn], th, t, inst, names)
        @test _woodbury_celerite_ll(y, v, t, th, CeleriteSHO(), F, 2π) ===
              _woodbury_celerite_ll(y, v, t, th, CeleriteSHO(), Matrix(F), 2π)
        p, th = setup([MaternGP(), mn], t, inst, names;
                      truth = merge(truth, Dict("matern_sigma" => 0.8, "matern_rho" => 12.0)))
        F = _additive_factor([mn], th, t, inst, names)
        @test _woodbury_matern_ll(y, v, t, th, MaternGP(), F, 2π) ===
              _woodbury_matern_ll(y, v, t, th, MaternGP(), Matrix(F), 2π)
        mh = HarmonicBlock(nharm = 2)
        p, th = setup([mn, mh], t, inst, names;
                      truth = merge(truth, Dict("harm_period" => 9.0, "harm_amp_A" => 1.0,
                                                "harm_amp_B" => 0.5)))
        Fc = _additive_factor([mn, mh], th, t, inst, names)
        @test Fc isa Matrix{Float64}
        @test all(Fc[:, 1:(size(Fc, 2) - 4)] .=== nightly_factor_ref(mn, th, t, inst, names))
    end

    @testset "ForwardDiff through σ and the data" begin
        m = NightlyOffset(gap = 0.5)
        p, th = setup([m], t, inst, names; truth)
        ix = [p.layout.name_to_idx["night_sigma_A"], p.layout.name_to_idx["night_sigma_B"]]
        function f(x, dense)
            thx = Nereus.Theta{eltype(x)}(p)
            thx.values .= th.values; thx.values[ix] .= x[1:2]
            F = _additive_factor([m], thx, t, inst, names)
            yx = y .+ x[3]
            return _woodbury_white_ll(yx, v .* x[4], dense ? Matrix(F) : F, eltype(x)(2π))
        end
        x0 = [0.8, 1.7, 0.1, 1.3]
        g = ForwardDiff.gradient(x -> f(x, false), x0)
        gd = ForwardDiff.gradient(x -> f(x, true), x0)
        @test g ≈ gd rtol = 1e-10
    end

    @testset "end to end through rv_log_likelihood" begin
        m = NightlyOffset(gap = 0.5)
        p, th = setup([m], t, inst, names; truth)
        for (k, val) in ("gamma_A" => 0.3, "gamma_B" => -0.2, "sigma_A" => 0.4, "sigma_B" => 0.7)
            Nereus.set_param!(th, k, val)
        end
        rv = y .+ [inst[i] == 1 ? 0.3 : -0.2 for i in 1:n]
        data = Nereus.Data(; t_rv = t, rv = rv, rv_err = fill(0.5, n), rv_inst = inst)
        var = [0.25 + (inst[i] == 1 ? 0.4 : 0.7)^2 for i in 1:n]
        Fd = nightly_factor_ref(m, th, t, inst, names)
        ll = Nereus.rv_log_likelihood(th, data)
        @test abs(ll - _woodbury_white_ll(y, var, Fd, 2π)) <= 1e-12 * abs(ll)
    end
end
