# TransDimConfig — user-facing configuration for trans-dimensional sampling.

"""
    BirthStrategy

Abstract type for planet birth proposal strategies.
"""
abstract type BirthStrategy end

"""
    PriorBirth()

Draw new planet parameters from the prior. Baseline strategy.
Low acceptance on weak signals, but always valid.
"""
struct PriorBirth <: BirthStrategy end

"""
    InformedBirth()

Propose period from Lomb-Scargle periodogram of residuals,
other parameters from prior. Much higher acceptance on real signals.
"""
struct InformedBirth <: BirthStrategy end

"""
    JointInformedBirth()

Joint-data informed-birth strategy. Combines the periodogram peaks
from RV residuals (Lomb-Scargle, weighted by 1/σ²) and photometry
residuals (Box Least Squares) into a single proposal mixture. Each
peak's combined weight is `max(LS_norm, BLS_norm)` so RV-only signals
(e.g. WASP-47c, no transit) and transit-only signals (e.g. WASP-47e,
RV undetectable) are both surfaced. BLS peaks within ±3% of m·P/n for
any active planet period P (m,n ∈ 1:5) are filtered out as harmonics
to prevent the chain from wandering onto aliases of already-found
planets.

Falls back to InformedBirth if no photometry data is available.
"""
struct JointInformedBirth <: BirthStrategy end

"""
    DonorBirth()

Clone a planet from another particle in the population (PT chain
or NS live point) with small mutation. Only available in samplers
with a population; silently falls back to other strategies in
standalone RJMCMC.
"""
struct DonorBirth <: BirthStrategy end

"""
    MoMSBirth(off_values, scales)

Mixtures of Mutually Singular distributions (van den Bergh, Clyde,
Raftery, Marsman, arXiv:2604.27791) flavored birth/death proposal:
each inactive planet sits at a designated off-location β_k_off in
parameter space, and an *add* move proposes new parameters via a
Gaussian random walk centered on β_k_off; a *delete* move
deterministically resets to β_k_off. The Metropolis-Hastings
acceptance probability uses the proposal density evaluated at the
proposed (add) or saved (delete) values.

Mathematically equivalent to RJMCMC under an identity dimension-map,
but with no auxiliary variables and no Jacobian — the bookkeeping is
just standard fixed-dim Metropolis-Hastings.

`off_values[k]` is the off-location vector for planet `k` (one entry
per *unfrozen* slot in that planet's block); `scales[k]` are the
diagonal random-walk standard deviations. Both are typically
constructed from the prior via `MoMSBirth(params::Params)` and the
scales adapted during warmup by `sample_moms`.
"""
mutable struct MoMSBirth <: BirthStrategy
    off_values::Vector{Vector{Float64}}
    scales::Vector{Vector{Float64}}
    slot_indices::Vector{Vector{Int}}   # unfrozen layout indices per planet
end

MoMSBirth() = MoMSBirth(Vector{Vector{Float64}}(),
                         Vector{Vector{Float64}}(),
                         Vector{Vector{Int}}())

"""
    TransDimConfig(; max_kplanet, kwargs...)

Configuration for trans-dimensional sampling. Opt-in: passing
`td=TransDimConfig(...)` to a sampler enables variable-dimension
exploration. `td=nothing` (default) means fixed-dim.

# Keywords
- `max_kplanet::Int`         — upper bound on N_p (required)
- `planets::Bool=true`       — enable planet birth/death
- `noise::Bool=false`        — enable noise component toggling
- `toggleable::Vector{NoiseModel}=[]` — which noise components can toggle
- `birth_strategies::Vector{BirthStrategy}=[PriorBirth(), InformedBirth()]`
- `birth_weights::Vector{Float64}=[0.3, 0.7]`
- `transdim_fraction::Float64=0.2` — fraction of moves that are birth/death
- `noise_exclusion_groups::Vector{Vector{NoiseModel}}=[]` — user-declared
  sets of mutually-exclusive toggleable noise models. At most one model
  in each group can be active at any time; the chain enforces this at
  proposal time. Use this when two models capture the same physics with
  different mathematical structures and shouldn't be allowed to compose.
  Canonical example: `[[bis_activity, gp_rotation]]` — both model stellar
  rotation modulation in RVs, so letting them coexist leaks planet
  signal into the joint activity fit (Nereus HD 18599 study, 2026).

  The independent Stage-2/Stage-3 exclusion (`SequentialNoise` AR/MA
  cannot coexist with `CovarianceNoise` GP) is *always* enforced
  regardless of this field — it's a hard mathematical constraint
  (sequential noise has no defined behavior on top of GP residuals),
  not a modelling choice. AR + MA stay composable: they are independent
  statistical processes (ARMA), not redundant models of the same thing.
"""
struct TransDimConfig
    planets::Bool
    max_kplanet::Int
    noise::Bool
    toggleable::Vector{NoiseModel}
    birth_strategies::Vector{BirthStrategy}
    birth_weights::Vector{Float64}
    transdim_fraction::Float64
    noise_exclusion_groups::Vector{Vector{NoiseModel}}
    # Fraction of between-model move slots spent on a within-model alias /
    # harmonic period jump instead of a birth/death. Green's update move: a
    # planet born at 2P of the truth cannot migrate by local steps, and cannot
    # escape by death-then-rebirth either (the death has to be accepted first).
    # Defaults to 0.0 — opt-in, no change to existing behaviour.
    alias_jump_fraction::Float64
end

function TransDimConfig(;
    max_kplanet::Int,
    planets::Bool = true,
    noise::Bool = false,
    toggleable::Vector{<:NoiseModel} = NoiseModel[],
    birth_strategies::Vector{<:BirthStrategy} = BirthStrategy[PriorBirth(), InformedBirth()],
    birth_weights::Vector{Float64} = Float64[0.3, 0.7],
    transdim_fraction::Float64 = 0.2,
    noise_exclusion_groups::Vector{<:Vector{<:NoiseModel}} = Vector{NoiseModel}[],
    alias_jump_fraction::Float64 = 0.0,
)
    max_kplanet >= 0 || throw(ArgumentError("max_kplanet must be ≥ 0"))
    length(birth_strategies) == length(birth_weights) || throw(ArgumentError(
        "birth_strategies ($(length(birth_strategies))) and birth_weights " *
        "($(length(birth_weights))) must have same length"))
    abs(sum(birth_weights) - 1.0) < 1e-10 || throw(ArgumentError(
        "birth_weights must sum to 1.0, got $(sum(birth_weights))"))
    0.0 ≤ transdim_fraction ≤ 1.0 || throw(ArgumentError(
        "transdim_fraction must be in [0, 1], got $transdim_fraction"))
    0.0 ≤ alias_jump_fraction ≤ 1.0 || throw(ArgumentError(
        "alias_jump_fraction must be in [0, 1], got $alias_jump_fraction"))
    if noise && isempty(toggleable)
        throw(ArgumentError("noise=true but no toggleable noise models provided"))
    end
    # Every model in every exclusion group must also be in `toggleable`,
    # otherwise the constraint can never fire (and the user has probably
    # made a typo).
    for (gi, group) in enumerate(noise_exclusion_groups)
        for nm in group
            nm in toggleable || throw(ArgumentError(
                "noise_exclusion_groups[$gi] references a model not in " *
                "toggleable: $nm"))
        end
        length(group) >= 2 || throw(ArgumentError(
            "noise_exclusion_groups[$gi] needs at least 2 models to be " *
            "meaningful, got $(length(group))"))
    end
    return TransDimConfig(
        planets, max_kplanet, noise,
        Vector{NoiseModel}(toggleable),
        Vector{BirthStrategy}(birth_strategies),
        birth_weights, transdim_fraction,
        [Vector{NoiseModel}(g) for g in noise_exclusion_groups],
        alias_jump_fraction,
    )
end
