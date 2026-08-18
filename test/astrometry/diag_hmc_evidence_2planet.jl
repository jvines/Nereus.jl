# Confirm the true joint evidence on the 2-Keplerian HD 114762 model with a
# THIRD independent method: PT-HMC (NUTS-within-PT). NUTS mixes each temperature
# far better than ptemcee's stretch moves, so if the evidence really is ≈-2465
# (nested) / -2501 (linear-trend Laplace), HMC's SS+/H+ should land there too —
# proving ptemcee's -15k is a mixing failure, not an ill-defined evidence.

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

println("Reference: nested = -2465.4,  linear-trend Laplace = -2501.6")
println("Running PT-HMC (16 temps, adaptive ladder)..."); flush(stdout)
c, ti_plus, rep = sample_pt_hmc(build(); n_temps = 16, n_sweeps = 1200, n_warmup = 400, seed = 42)
e = median(vec(Array(c[:, :sesinw_k1, :])) .^ 2 .+ vec(Array(c[:, :secosw_k1, :])) .^ 2)
P = median(vec(Array(c[:, :P_k1, :])))
@printf("\nPT-HMC: TI+=%.1f  SS+=%.1f  H+=%.1f | P_b=%.2f e=%.3f\n",
        rep.ti_plus[1], rep.ss_plus[1], rep.hybrid[1], P, e)
println("\n=== Do nested / Laplace / HMC-SS+ agree ~-2500 (⇒ ptemcee's -15k is a mixing bug)? ===")
