# Does the ptemcee logZ bias CANCEL in a model comparison? For selection we need
# comparable Δlog Z, not exact logZ. Test: same HD 114762 data, 1-planet (b) vs
# noise-only (0-planet). Compute logZ for both via TI+, SS+, mode-Laplace, and
# nested (ground truth). Δ = planet - noise. If an estimator's bias is model-
# independent it cancels and its Δ matches nested's Δ.
#
# Prediction: the phase-transition bias hits ONLY the planet model (noise has no
# signal → no transition), so TI's Δ will be WRONG; mode-Laplace's bias (railed
# jitter, common to both) should cancel better.

using Nereus, MCMCChains, Printf, Statistics, LinearAlgebra
using LogDensityProblems

const SID = 3937211745905473024
const M_PRI, PLX, PLX_ERR = 0.83, 25.36, 0.30
rvfile = get(ENV, "HD114762_RV", "")
tb = Dict{String, Vector{Float64}}()
for line in eachline(rvfile)
    s = strip(line); (isempty(s) || startswith(s, "#")) && continue
    t, rv, er, ins = split(s)
    append!(get!(() -> Float64[], tb, ins),
            (parse(Float64, t) - 2_400_000.5, parse(Float64, rv), parse(Float64, er)))
end
mkrv(k) = let v = reshape(tb[k], 3, :); (t = v[1, :], rv = v[2, :], rv_err = v[3, :]) end
hires, lick = mkrv("j"), mkrv("lick")
xml = get(ENV, "NEREUS_GAIA_DR4_XML", ""); (isempty(xml) || !isfile(xml)) && (xml = fetch_gaia_dr4_prerelease())
src = read_gaia_epoch_votable(xml, SID)
t_rv = vcat(hires.t, lick.t); rv = vcat(hires.rv, lick.rv); rv_err = vcat(hires.rv_err, lick.rv_err)
rv_inst = vcat(fill(1, length(hires.t)), fill(2, length(lick.t)))
data = Data(; t_rv = t_rv, rv = rv, rv_err = rv_err, rv_inst = rv_inst, iad = src.iad)
instruments = InstrumentConfig(rv = ["HIRES", "Lick"], pm = String[])
pz = ParametrizationConfig(mass = :M_sec_driven)

planet_priors = Dict{String, PriorSpec}(
    "P_k1" => LogUniformPrior(60.0, 110.0), "M_sec_k1" => LogUniformPrior(0.003, 0.5),
    "inc_k1" => SinePrior(), "Omega_k1" => UniformPrior(0.0, 2π),
    "plx" => NormalPrior(PLX, PLX_ERR))
noise_priors = Dict{String, PriorSpec}()   # 0-planet model has no plx/orbit params

planet_tgt() = NereusTarget(Params(; max_kplanet = 1, planet_modes = [RVAS],
    instruments = instruments, parametrization = pz, priors = planet_priors,
    data = data, M_s = M_PRI, trend_order = 1), data; unconstrained = true)
noise_tgt() = NereusTarget(Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
    instruments = instruments, parametrization = pz, priors = noise_priors,
    data = data, M_s = M_PRI, trend_order = 1), data; unconstrained = true)

# mode-anchored Laplace from a cold chain (local covariance, closest 30%)
function mode_laplace(tgt, c)
    names = tgt.params.layout.unfrozen_names; d = length(names)
    cols = [vec(Array(c[:, Symbol(nm), :])) for nm in names]
    X = permutedims(hcat(cols...)); N0 = size(X, 2); pt = tgt.transform
    Y = Matrix{Float64}(undef, d, N0); lq = Vector{Float64}(undef, N0)
    for i in 1:N0
        yi = Nereus.transform_forward(X[:, i], pt); Y[:, i] = yi
        lq[i] = LogDensityProblems.logdensity(tgt, yi)
    end
    keep = isfinite.(lq); Y = Y[:, keep]; lq = lq[keep]
    imax = argmax(lq); mode = Y[:, imax]
    Σ = cov(Y; dims = 2) + 1e-8 * I(d); L = cholesky(Symmetric(Σ)).L
    md = [norm(L \ (Y[:, i] .- mode)) for i in 1:length(lq)]
    Yc = Y[:, md .< quantile(md, 0.3)]; Sc = cov(Yc; dims = 2) + 1e-8 * I(d)
    return lq[imax] + 0.5 * d * log(2π) + 0.5 * logdet(Symmetric(Sc)) + log(0.3)
end

function run_model(name, mk)
    tgt = mk()
    r = sample_ptemcee(tgt, tgt.data; n_temps = 24, n_walkers = 100, n_steps = 3000,
                       n_burnin = 1000, seed = 42, show_progress = false)
    ml = mode_laplace(tgt, r.chains)
    lzn = try
        _, z = sample_nested(mk(), data; n_live = 400, bounds = :multi,
                             proposal = :rwalk, n_walks = 25, dlogz = 0.5, seed = 42)
        z
    catch err
        @warn "nested failed for $name" err = sprint(showerror, err); NaN
    end
    @printf("%-8s | TI+=%10.1f  SS+=%10.1f  Laplace=%9.1f  nested=%9.1f\n",
            name, r.evidence.ti_plus[1], r.evidence.ss_plus[1], ml, lzn)
    flush(stdout)
    return (ti = r.evidence.ti_plus[1], ss = r.evidence.ss_plus[1], lap = ml, nes = lzn)
end

println("model    |         TI+          SS+       Laplace       nested"); flush(stdout)
P = run_model("planet", planet_tgt)
Nz = run_model("noise", noise_tgt)
println("\n=== Δ log Z = planet − noise  (does the bias cancel?) ===")
@printf("Δ TI+     = %.1f\n", P.ti - Nz.ti)
@printf("Δ SS+     = %.1f\n", P.ss - Nz.ss)
@printf("Δ Laplace = %.1f\n", P.lap - Nz.lap)
@printf("Δ nested  = %.1f   <- ground truth\n", P.nes - Nz.nes)
println("\n(planet b is a ~600 m/s + astrometric signal → Δ should be large & positive.")
println(" whichever estimator's Δ matches nested's Δ gives usable model selection.)")
