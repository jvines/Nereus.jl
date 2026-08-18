# #2 done right: evidence from ptemcee's posterior samples, MODE-centered.
# Earlier failure was my bug — I centered φ / Laplace on the posterior MEAN,
# which for a multimodal (e-alias) posterior sits in a low-density VALLEY
# (logq(μ)=-14469 vs samples -2329). Mode-centering fixes it. Target: -2465.

using Nereus, MCMCChains, Printf, Statistics, LinearAlgebra
using LogExpFunctions: logsumexp
using Distributions: MvNormal, logpdf
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
parametrization = ParametrizationConfig(mass = :M_sec_driven)
priors = Dict{String, PriorSpec}(
    "P_k1" => LogUniformPrior(60.0, 110.0), "M_sec_k1" => LogUniformPrior(0.003, 0.5),
    "inc_k1" => SinePrior(), "Omega_k1" => UniformPrior(0.0, 2π),
    "P_k2" => LogUniformPrior(1.1e4, 1.0e7), "M_sec_k2" => LogUniformPrior(1e-3, 0.5),
    "plx" => NormalPrior(PLX, PLX_ERR))
tgt = NereusTarget(
    Params(; max_kplanet = 2, planet_modes = [RVAS, RV_ONLY], instruments = instruments,
           parametrization = parametrization, priors = priors, data = data,
           M_s = M_PRI, trend_order = 0), data; unconstrained = true)

println("Reference: nested = -2465.4"); flush(stdout)
r = sample_ptemcee(tgt, tgt.data; n_temps = 24, n_walkers = 100, n_steps = 3000,
                   n_burnin = 1000, seed = 42, show_progress = false)
c = r.chains; names = tgt.params.layout.unfrozen_names; d = length(names)
cols = [vec(Array(c[:, Symbol(nm), :])) for nm in names]
N0 = length(cols[1]); X = permutedims(hcat(cols...)); pt = tgt.transform
Y = Matrix{Float64}(undef, d, N0); logq = Vector{Float64}(undef, N0)
for i in 1:N0
    yi = Nereus.transform_forward(X[:, i], pt); Y[:, i] = yi
    logq[i] = LogDensityProblems.logdensity(tgt, yi)
end
keep = isfinite.(logq); Y = Y[:, keep]; logq = logq[keep]; N = length(logq)
Σ = cov(Y; dims = 2) + 1e-8 * I(d)
imax = argmax(logq); mode = Y[:, imax]
@printf("logq: mean-point=%.1f  MODE=%.1f  (samples median %.1f)\n",
        LogDensityProblems.logdensity(tgt, vec(mean(Y, dims = 2))), logq[imax], median(logq))
@printf("mode-Laplace logZ = %.1f\n\n", logq[imax] + 0.5 * d * log(2π) + 0.5 * logdet(Symmetric(Σ)))

println("Mode-centered Learned Harmonic Mean:")
@printf("  %-6s %12s %10s\n", "s", "logZ", "ESS/N")
for s in (1.0, 0.7, 0.5, 0.35, 0.25, 0.15)
    φ = MvNormal(mode, Symmetric(s^2 .* Σ))
    lw = [logpdf(φ, Y[:, i]) for i in 1:N] .- logq
    logZ = log(N) - logsumexp(lw)
    ln = lw .- logsumexp(lw); ess = exp(-logsumexp(2 .* ln))
    @printf("  %-6.2f %12.1f %10.4f\n", s, logZ, ess / N)
end

# dominant-mode-restricted: closest 50% of samples to the mode, refit local Gaussian
L = cholesky(Symmetric(Σ)).L
md = [norm(L \ (Y[:, i] .- mode)) for i in 1:N]
sel = md .< quantile(md, 0.5)
Ys = Y[:, sel]; lq = logq[sel]; Ns = size(Ys, 2)
ms = vec(mean(Ys, dims = 2)); Ss = cov(Ys; dims = 2) + 1e-8 * I(d)
# mode-Laplace with the LOCAL (mode-restricted) covariance — should tighten -2330→-2465.
# The half of samples nearest the mode occupy log(0.5) of the mass, so add log(frac).
for frac in (0.5, 0.3, 0.15)
    selc = md .< quantile(md, frac)
    Yc = Y[:, selc]; Sc = cov(Yc; dims = 2) + 1e-8 * I(d)
    logZ_local = logq[imax] + 0.5 * d * log(2π) + 0.5 * logdet(Symmetric(Sc)) + log(frac)
    @printf("mode-Laplace (local Σ, closest %.0f%%) logZ = %.1f\n", 100frac, logZ_local)
end
println("\nDominant-mode-restricted LHM (closest 50%, local Σ, + log(frac) correction):")
for s in (1.0, 0.7, 0.5)
    φ = MvNormal(ms, Symmetric(s^2 .* Ss))
    lw = [logpdf(φ, Ys[:, i]) for i in 1:Ns] .- lq
    logZc = log(Ns) - logsumexp(lw)
    ln = lw .- logsumexp(lw); ess = exp(-logsumexp(2 .* ln))
    @printf("  s=%.2f  logZ=%.1f  ESS/N=%.4f\n", s, logZc + log(Ns / N), ess / Ns)
end
println("\n=== target -2465: does mode-centering recover it with usable ESS? ===")
