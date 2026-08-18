# The PT-family evidence on HD 114762 (2-Keplerian) is catastrophically biased
# LOW (-15k..-38k) vs the true ≈-2465 (nested/Laplace). Diagnosis: swap
# acceptance is 2-3% → temperatures don't communicate → walkers strand at
# railed/low-L configs → ⟨logL⟩_β polluted → TI/SS/H all wrong. PREDICTION:
# a FINER ladder (more temps → higher swap acceptance) drives the PT evidence
# UP toward -2465. This directly tests the fix (better ladder), not a new prior.

using Nereus, MCMCChains, Printf, Statistics

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
e_of(c) = median(vec(Array(c[:, :sesinw_k1, :])) .^ 2 .+ vec(Array(c[:, :secosw_k1, :])) .^ 2)

@printf("TRUE evidence ≈ -2465 (nested) / -2501 (Laplace); logL_max ≈ -2000\n\n")
@printf("%-5s %-6s | %10s %10s | %8s %8s | %s\n",
        "nT", "steps", "TI+", "SS+", "swap_min", "swap_avg", "e_mode")
println("-"^70)
for (nT, nS, nB) in [(12, 2500, 800), (24, 2500, 800), (48, 3000, 1000), (72, 3500, 1200)]
    tgt = build()
    r = sample_ptemcee(tgt, tgt.data; n_temps = nT, n_walkers = 80, n_steps = nS,
                       n_burnin = nB, seed = 42, show_progress = false)
    sw = r.acceptance_swap
    @printf("%-5d %-6d | %10.1f %10.1f | %8.3f %8.3f | %.3f\n",
            nT, nS, r.evidence.ti_plus[1], r.evidence.ss_plus[1],
            isempty(sw) ? NaN : minimum(sw), isempty(sw) ? NaN : mean(sw), e_of(r.chains))
    flush(stdout)
end
println("\n=== Does swap acceptance climb and PT evidence rise toward -2465 with a finer ladder? ===")
