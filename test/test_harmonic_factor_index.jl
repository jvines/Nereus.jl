# HarmonicBlock factor: the per-instrument amplitude slot is resolved once per
# instrument instead of by building a "harm_amp_<inst>" name for every point
# (src/noise/parametric_noise.jl, _harmonic_factor). On a 20,000-point light
# curve that string work cost more than the factor itself. The factor must be
# unchanged bit for bit, which these tests check against the per-point lookup.
using Nereus, Test
import ForwardDiff
using Nereus: HarmonicBlock, _harmonic_factor, _covered_inst_idx, _is_covered,
              _channel_suffix

# The per-point lookup the factor used before (reference).
function harmonic_factor_ref(m, theta::Nereus.Theta{T}, times, inst, inst_names) where {T}
    N = length(times); K = m.nharm; s = _channel_suffix(m.channel)
    layout = theta.params.layout
    external = !isempty(m.freqs)
    ω = external ? T(0) : T(2π) / theta.values[layout.name_to_idx["harm_period$s"]]
    cov = _covered_inst_idx(m.instruments, inst_names)
    F = zeros(T, N, 2K)
    for i in 1:N
        _is_covered(cov, inst[i]) || continue
        A = theta.values[layout.name_to_idx["harm_amp_$(inst_names[inst[i]])$s"]]
        for k in 1:K
            arg = external ? T(2π) * T(m.freqs[k]) * T(times[i]) : ω * k * T(times[i])
            F[i, 2k - 1] = A * cos(arg)
            F[i, 2k]     = A * sin(arg)
        end
    end
    return F
end

@testset "HarmonicBlock factor: per-instrument amplitude lookup" begin
    names = ["A", "B", "C"]
    n = 90
    t = [10.0 + 0.37 * i + 0.05 * sin(3.1 * i) for i in 1:n]
    inst = [mod(7i, 3) + 1 for i in 1:n]                 # interleaved instruments
    data = Nereus.Data(; t_rv = t, rv = zeros(n), rv_err = fill(0.5, n), rv_inst = inst)
    ic = Nereus.InstrumentConfig(rv = names)
    build(m) = Nereus.Params(; max_kplanet = 0, planet_modes = Nereus.PlanetDataSources[],
                             instruments = ic, data = data, M_s = 1.0,
                             noise_models = Nereus.NoiseModel[m])
    bits_equal(X, Y) = size(X) == size(Y) && all(X .=== Y)

    for m in (HarmonicBlock(nharm = 3),                                  # all instruments
              HarmonicBlock(nharm = 2, instruments = ["C", "A"]),        # a subset
              HarmonicBlock(nharm = 2, freqs = [0.21, 1.7]))             # external comb
        p = build(m)
        th = Nereus.Theta{Float64}(p)
        for (j, nm) in enumerate(p.layout.unfrozen_names)
            v = nm == "harm_period" ? 7.3 : startswith(nm, "harm_amp_") ? 0.4 + 0.3j : nothing
            v === nothing || Nereus.set_param!(th, nm, v)
        end
        F = _harmonic_factor(m, th, t, inst, names)
        @test bits_equal(F, harmonic_factor_ref(m, th, t, inst, names))
        # each covered row carries its own instrument's amplitude
        cov = _covered_inst_idx(m.instruments, names)
        for i in 1:n
            if _is_covered(cov, inst[i])
                A = Nereus.get_param(th, "harm_amp_$(names[inst[i]])")
                @test hypot(F[i, 1], F[i, 2]) ≈ A
            else
                @test all(iszero, F[i, :])
            end
        end
        # Dual parameters: same factor as the per-point lookup
        thd = Nereus.Theta{ForwardDiff.Dual{Nothing, Float64, 1}}(p)
        for k in eachindex(th.values)
            thd.values[k] = ForwardDiff.Dual{Nothing}(th.values[k], k == 1 ? 1.0 : 0.0)
        end
        @test bits_equal(_harmonic_factor(m, thd, t, inst, names),
                         harmonic_factor_ref(m, thd, t, inst, names))
    end

    # No per-point allocation: the factor itself plus a constant (the per-point
    # name lookup allocated ~33 B per point on top of it).
    nl = 3000
    tl = [0.01 * i for i in 1:nl]; il = [mod(i, 3) + 1 for i in 1:nl]
    dl = Nereus.Data(; t_rv = tl, rv = zeros(nl), rv_err = fill(0.5, nl), rv_inst = il)
    m = HarmonicBlock(nharm = 2)
    pl = Nereus.Params(; max_kplanet = 0, planet_modes = Nereus.PlanetDataSources[],
                       instruments = ic, data = dl, M_s = 1.0, noise_models = Nereus.NoiseModel[m])
    thl = Nereus.Theta{Float64}(pl); Nereus.set_param!(thl, "harm_period", 7.3)
    f() = _harmonic_factor(m, thl, tl, il, names)
    Fl = f()
    @test (@allocated f()) <= sizeof(Fl) + 4096
end
