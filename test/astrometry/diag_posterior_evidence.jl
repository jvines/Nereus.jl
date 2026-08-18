# Evidence from ptemcee's POSTERIOR SAMPLES (not the tempered path) — sidesteps
# the phase transition entirely. ptemcee recovers the orbit correctly; its cold
# chain is a good posterior sample. Two posterior-based estimators:
#   (1) Learned Harmonic Mean (McEwen+ 2021): 1/Z = E_post[φ(y)/q(y)], φ a
#       normalized density with THINNER tails than the posterior (narrowed
#       Gaussian in unconstrained space) so φ/q is bounded → finite variance.
#   (2) Sample-Laplace: if p(y)≈N(ȳ,Σ), logZ = logq(ȳ) + d/2 log2π + ½ logdet Σ.
# Target: nested = -2465.4, Laplace(linear) = -2501.6.

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
t_rv = vcat(hires.t, lick.t); rv = vcat(hires.rv, lick.rv)
rv_err = vcat(hires.rv_err, lick.rv_err)
rv_inst = vcat(fill(1, length(hires.t)), fill(2, length(lick.t)))
data = Data(; t_rv = t_rv, rv = rv, rv_err = rv_err, rv_inst = rv_inst, iad = src.iad)
instruments = InstrumentConfig(rv = ["HIRES", "Lick"], pm = String[])
parametrization = ParametrizationConfig(mass = :M_sec_driven)
priors = Dict{String, PriorSpec}(
    "P_k1" => LogUniformPrior(60.0, 110.0), "M_sec_k1" => LogUniformPrior(0.003, 0.5),
    "inc_k1" => SinePrior(), "Omega_k1" => UniformPrior(0.0, 2π),
    "P_k2" => LogUniformPrior(1.1e4, 1.0e7), "M_sec_k2" => LogUniformPrior(1e-3, 0.5),
    "plx" => NormalPrior(PLX, PLX_ERR),
)
build() = NereusTarget(
    Params(; max_kplanet = 2, planet_modes = [RVAS, RV_ONLY], instruments = instruments,
           parametrization = parametrization, priors = priors, data = data,
           M_s = M_PRI, trend_order = 0), data; unconstrained = true)

println("Reference: nested = -2465.4,  Laplace(linear) = -2501.6")
tgt = build()
r = sample_ptemcee(tgt, tgt.data; n_temps = 24, n_walkers = 100, n_steps = 3000,
                   n_burnin = 1000, seed = 42, show_progress = false)
c = r.chains
names = tgt.params.layout.unfrozen_names
d = length(names)
cols = [vec(Array(c[:, Symbol(nm), :])) for nm in names]
N = length(cols[1])
X = permutedims(hcat(cols...))          # d × N bounded samples
pt = tgt.transform

# bounded x → unconstrained y (verify direction with a round-trip)
x1 = X[:, 1]
y1 = Nereus.transform_forward(x1, pt)
x1b = Nereus.transform_inverse(y1, pt)
@printf("transform round-trip max|x-inv(fwd(x))| = %.2e  (should be ~0)\n", maximum(abs, x1 .- x1b))

Y = Matrix{Float64}(undef, d, N)
logq = Vector{Float64}(undef, N)
for i in 1:N
    yi = Nereus.transform_forward(X[:, i], pt)
    Y[:, i] = yi
    logq[i] = LogDensityProblems.logdensity(tgt, yi)
end
keep = isfinite.(logq)
Y = Y[:, keep]; logq = logq[keep]; N = length(logq)
@printf("N posterior samples = %d,  logq: median=%.0f max=%.0f\n", N, median(logq), maximum(logq))

μ = vec(mean(Y, dims = 2))
Σ = cov(Y; dims = 2) + 1e-8 * I(d)

# (2) sample-Laplace
logq_μ = LogDensityProblems.logdensity(tgt, μ)
logZ_lap = logq_μ + 0.5 * d * log(2π) + 0.5 * logdet(Σ)
@printf("\nsample-Laplace  logZ = %.1f\n", logZ_lap)

# (1) learned harmonic mean, narrowing s (thinner tails → bounded φ/q)
println("\nLearned Harmonic Mean (narrowed-Gaussian φ):")
@printf("  %-6s %12s %10s\n", "s", "logZ", "ESS/N")
for s in (1.0, 0.7, 0.5, 0.35, 0.25)
    φ = MvNormal(μ, Symmetric(s^2 .* Σ))
    logφ = [logpdf(φ, Y[:, i]) for i in 1:N]
    w = logφ .- logq                     # log weights
    logZ = log(N) - logsumexp(w)
    # ESS of the (unnormalized) weights → reliability
    lw = w .- logsumexp(w)
    ess = exp(-logsumexp(2 .* lw))
    @printf("  %-6.2f %12.1f %10.3f\n", s, logZ, ess / N)
end
println("\n=== Does a posterior-based estimator recover ~-2465 from ptemcee's OWN samples? ===")
