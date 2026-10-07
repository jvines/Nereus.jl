# The celerite recursion takes consecutive differences times[n] - times[n-1],
# so it is only valid over time-ordered points. RV data are routinely stored
# instrument by instrument (CORALIE after HARPS), which is not time order;
# taken as given, the recursion scored them wrongly or returned -Inf. The
# points are now scored in time order whatever order they arrive in, so a
# shuffled data set must give the log-likelihood of the sorted one -- on every
# celerite path: the kernels, the solve, the per-instrument GPs, the additive
# (Woodbury) bases, the sampler workspaces and the full RV / photometry
# likelihoods. A dense Cholesky, which is order-free, is the oracle.
using Nereus, Test, Random, LinearAlgebra
using Nereus: Theta, Params, Data, InstrumentConfig, PlanetDataSources, NoiseModel,
              CeleriteSHO, CeleriteRotation, CeleriteRotationFM17, MaternGP,
              NightlyOffset, HarmonicBlock, PTWorkspace, gp_log_likelihood,
              celerite_loglike, celerite_solve, gp_mean_at, sho_coefficients,
              _celerite_coeffs, _eval_channel_likelihood, _woodbury_celerite_ll,
              _additive_factor, dense_additive_ll, rv_predictions

# Dense kernel matrix of a celerite coefficient set (order-free oracle).
function _dense_celerite(t, ar, cr, ac, bc, cc, dc)
    n = length(t)
    K = zeros(n, n)
    for i in 1:n, j in 1:n
        τ = abs(t[i] - t[j])
        k = 0.0
        for r in eachindex(ar); k += ar[r] * exp(-cr[r] * τ); end
        for c in eachindex(ac)
            k += exp(-cc[c] * τ) * (ac[c] * cos(dc[c] * τ) + bc[c] * sin(dc[c] * τ))
        end
        K[i, j] = k
    end
    return K
end

function _dense_ll(y, C)
    F = cholesky(Symmetric(C))
    return -0.5 * (dot(y, F \ y) + logdet(F) + length(y) * log(2π))
end

# Two instruments, each internally in time order, stored one after the other.
# Their baselines interleave, so the concatenation is far from time order.
function _two_instrument_epochs(rng; nA = 30, nB = 22)
    tA = sort(1000.0 .+ 300 .* rand(rng, nA))
    tB = sort(1000.0 .+ 300 .* rand(rng, nB))
    return vcat(tA, tB), vcat(fill(1, nA), fill(2, nB))
end

function _rv_params(t, inst, rv, err, noise)
    d = Data(; t_rv = t, rv = rv, rv_err = err, rv_inst = inst)
    p = Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
               instruments = InstrumentConfig(rv = ["A", "B"]), data = d,
               M_s = 1.0, noise_models = NoiseModel[noise...])
    return p, d
end

function _set!(th, vals)
    for (k, v) in vals
        haskey(th.params.layout.name_to_idx, k) && Nereus.set_param!(th, k, v)
    end
    return th
end

const _HYP = Dict(
    "gp_log_S0" => 1.2, "gp_log_Q" => 0.8, "gp_log_omega0" => -0.4,
    "gp_sigma" => 3.0, "gp_period" => 23.0, "gp_Q0" => 1.5, "gp_dQ" => 2.0, "gp_f" => 0.4,
    "gp_log_amp" => 2.0, "gp_log_timescale" => 3.5, "gp_log_period" => log(17.0),
    "gp_log_factor" => -0.5,
    "matern_sigma" => 2.5, "matern_rho" => 6.0,
    "gamma_A" => 0.3, "gamma_B" => -0.8, "sigma_A" => 0.7, "sigma_B" => 1.1,
    "harm_period" => 19.0, "harm_amp_A" => 1.3, "harm_amp_B" => 0.6,
    "night_sigma_A" => 0.9, "night_sigma_B" => 0.4,
    # instrument-restricted GPs carry the instrument in their names
    "gp_log_S0_A" => 1.0, "gp_log_Q_A" => 1.1, "gp_log_omega0_A" => -0.2,
    "gp_sigma_B" => 2.0, "gp_period_B" => 13.0, "gp_Q0_B" => 0.9, "gp_dQ_B" => 1.0,
    "gp_f_B" => 0.6,
)

_same_bits(a, b) = length(a) == length(b) &&
                    all(reinterpret(UInt64, a) .== reinterpret(UInt64, b))

@testset "celerite GPs score points in time order" begin
    rng = MersenneTwister(20261006)
    t, inst = _two_instrument_epochs(rng)
    n = length(t)
    @test !issorted(t)
    srt = sortperm(t)
    y = 3 .* randn(rng, n)
    v = 0.4 .+ rand(rng, n)

    @testset "kernel $(nameof(typeof(nm)))" for nm in (CeleriteSHO(), CeleriteRotation(),
                                                       CeleriteRotationFM17())
        p, _ = _rv_params(t, inst, y, sqrt.(v), [nm])
        th = _set!(Theta{Float64}(p), _HYP)
        ar, cr, ac, bc, cc, dc = _celerite_coeffs(th, nm)
        dense = _dense_ll(y, _dense_celerite(t, ar, cr, ac, bc, cc, dc) + Diagonal(v))

        ll_sorted = gp_log_likelihood(y[srt], v[srt], t[srt], th, nm)
        ll_given  = gp_log_likelihood(y, v, t, th, nm)
        @test isfinite(ll_given)
        @test ll_given === ll_sorted                     # the same points, sorted
        @test ll_given ≈ dense rtol = 1e-9
        @test celerite_loglike(t, y, v, ar, cr, ac, bc, cc, dc) === ll_sorted

        # Any other order of the same points: the same value.
        q = randperm(rng, n)
        @test gp_log_likelihood(y[q], v[q], t[q], th, nm) === ll_sorted

        # The solve comes back in the caller's order.
        α = celerite_solve(t, y, v, ar, cr, ac, bc, cc, dc)
        α_dense = (_dense_celerite(t, ar, cr, ac, bc, cc, dc) + Diagonal(v)) \ y
        @test α ≈ α_dense rtol = 1e-7
        @test _same_bits(α[srt], celerite_solve(t[srt], y[srt], v[srt], ar, cr, ac, bc, cc, dc))

        # And so does the GP mean.
        tp = collect(range(1000.0, 1300.0; length = 41))
        @test gp_mean_at(y, v, t, tp, th, nm) ≈
              gp_mean_at(y[srt], v[srt], t[srt], tp, th, nm) rtol = 1e-10
    end

    @testset "Matérn (semiseparable) stays order-free" begin
        nm = MaternGP()
        p, _ = _rv_params(t, inst, y, sqrt.(v), [nm])
        th = _set!(Theta{Float64}(p), _HYP)
        @test gp_log_likelihood(y, v, t, th, nm) ≈
              gp_log_likelihood(y[srt], v[srt], t[srt], th, nm) rtol = 1e-12
    end

    @testset "additive covariance on a celerite base: $label" for (label, add) in (
            ("HarmonicBlock", HarmonicBlock(nharm = 2)),
            ("NightlyOffset", NightlyOffset(gap = 0.5)))
        base = CeleriteSHO()
        p, _ = _rv_params(t, inst, y, sqrt.(v), [base, add])
        th = _set!(Theta{Float64}(p), _HYP)
        F = _additive_factor([add], th, t, inst, ["A", "B"])
        ll = _woodbury_celerite_ll(y, v, t, th, base, F, 2π)
        @test isfinite(ll)
        @test ll ≈ dense_additive_ll(y, v, t, inst, ["A", "B"], [add], th;
                                     base_nm = base) rtol = 1e-9
        Fs = _additive_factor([add], th, t[srt], inst[srt], ["A", "B"])
        @test ll ≈ _woodbury_celerite_ll(y[srt], v[srt], t[srt], th, base, Fs, 2π) rtol = 1e-12
    end

    # Full RV likelihood: the data as stored (instrument by instrument) against
    # the same data sorted by time -- general and sampler-workspace paths.
    @testset "RV likelihood, $label" for (label, noise) in (
            ("global rotation", [CeleriteRotation()]),
            ("global SHO", [CeleriteSHO()]),
            ("global FM17", [CeleriteRotationFM17()]),
            ("per-instrument GPs", [CeleriteSHO(instruments = ["A"]),
                                    CeleriteRotation(instruments = ["B"])]),
            ("SHO + HarmonicBlock", [CeleriteSHO(), HarmonicBlock(nharm = 2)]))
        err = sqrt.(v)
        p, d = _rv_params(t, inst, y, err, noise)
        ps, ds = _rv_params(t[srt], inst[srt], y[srt], err[srt], noise)
        th = _set!(Theta{Float64}(p), _HYP)
        ths = _set!(Theta{Float64}(ps), _HYP)
        ll = Nereus._rv_log_likelihood_core(th, d)
        lls = Nereus._rv_log_likelihood_core(ths, ds)
        @test isfinite(ll)
        @test ll ≈ lls rtol = 1e-12

        ws = PTWorkspace(p, 0, length(p.config.noise_models); n_obs = n)
        @test Nereus._rv_log_likelihood_core(th, d, ws) ≈ lls rtol = 1e-12
        @test Nereus._rv_log_likelihood_core(th, d, ws) ≈ lls rtol = 1e-12   # cached order

        # Dense oracle: Σ = diag(var) + every GP block on its points.
        preds, var = rv_predictions(th, d)
        r = d.rv .- preds
        C = Matrix(Diagonal(var))
        for nm in noise
            nm isa Nereus.CovarianceNoise || continue
            sel = isempty(nm.instruments) ? collect(1:n) :
                  findall(i -> ["A", "B"][inst[i]] in nm.instruments, 1:n)
            C[sel, sel] .+= _dense_celerite(t[sel], _celerite_coeffs(th, nm)...)
        end
        for nm in noise
            nm isa Nereus.AdditiveCovariance || continue
            F = Matrix(_additive_factor([nm], th, t, inst, ["A", "B"]))
            C .+= F * F'
        end
        @test ll ≈ _dense_ll(r, C) rtol = 1e-9
    end

    @testset "workspace path: unsorted times allocate nothing per call" begin
        p, d = _rv_params(t, inst, y, sqrt.(v), [CeleriteRotation()])
        th = _set!(Theta{Float64}(p), _HYP)
        ws = PTWorkspace(p, 0, 1; n_obs = n)
        nm = p.config.noise_models[1]
        a = gp_log_likelihood(y, v, d.t_rv, th, nm, ws)
        @test a === gp_log_likelihood(y, v, d.t_rv, th, nm)
        f(r, v, t, th, nm, ws) = @allocated gp_log_likelihood(r, v, t, th, nm, ws)
        f(y, v, d.t_rv, th, nm, ws)
        @test f(y, v, d.t_rv, th, nm, ws) == 0
    end

    @testset "photometry channel GP" begin
        tp1 = sort(10 .* rand(rng, 40)); tp2 = sort(10 .* rand(rng, 35))
        tph = vcat(tp1, tp2); pinst = vcat(fill(1, 40), fill(2, 35))
        np = length(tph)
        flux = 1 .+ 1e-3 .* randn(rng, np); ferr = fill(5e-4, np)
        mk(tt, ff, ee, ii) = begin
            dd = Data(; t_phot = tt, flux = ff, flux_err = ee, phot_inst = ii)
            pp = Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
                        instruments = InstrumentConfig(String[], ["T1", "T2"]), data = dd,
                        M_s = 1.0, R_s = 1.0,
                        noise_models = NoiseModel[CeleriteSHO(channel = :phot)])
            pp, dd
        end
        p, d = mk(tph, flux, ferr, pinst)
        o = sortperm(tph)
        ps, ds = mk(tph[o], flux[o], ferr[o], pinst[o])
        hyp = Dict("gp_log_S0_phot" => log(1e-6), "gp_log_Q_phot" => 0.0,
                   "gp_log_omega0_phot" => 0.5)
        th = _set!(Theta{Float64}(p), hyp); ths = _set!(Theta{Float64}(ps), hyp)
        ll = Nereus.transit_log_likelihood(th, d)
        @test isfinite(ll)
        @test ll ≈ Nereus.transit_log_likelihood(ths, ds) rtol = 1e-12
    end
end
