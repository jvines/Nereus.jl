# Parameters held at a fixed value early in burn-in, for the PT ensemble
# samplers (pt_emcee, transdim_pt_emcee, pt_whitening).
#
# An intermediate-astrometry jitter `sigma_iad_<instrument>` opens a second way
# up the likelihood from the prior: inflate s until the reflex stops mattering.
# On HD 114762 RV + Gaia DR4 (a 1.9 mas reflex against ~0.12 mas per-CCD
# errors) that way is the cheaper one, and pt_emcee started from the prior
# (24 x 100 walkers, 8000/4000 steps) ended in it on every seed tried: s 0.7-1
# mas, M_sec 0.04-0.09 Msun, i 7-14 deg, 470-970 nats below the orbit (s 0.36
# mas, M_sec 0.198 Msun, i 3.5 deg), R-hat 2.7-4.1. No walker in any rung ever
# reached the orbit, so the stranded-walker prune had nothing to move the rest
# onto. Without the jitter the same fit finds the orbit from the prior.
#
# So the jitter is held at 0 for the first part of burn-in and then let go.
# While every walker of every rung has the same value of a coordinate, nothing
# moves it: the stretch move proposes x_p + z(x_w - x_p), which is x_p exactly
# where the two agree; a swap exchanges whole states; the prune copies whole
# walkers or scatters only the other coordinates; and a held coordinate may not
# be one the node flip, the λ slide or a trans-dim move acts on. At s = 0 the IAD
# likelihood is the unjittered one bit for bit (`_iad_normal_equations!`), so
# the held phase IS the fit without jitter. At the release each walker's value
# is drawn from its prior and the walker re-evaluated, and the rest of burn-in
# lets it settle. Burn-in only: the recorded chain is ordinary MCMC on the full
# posterior. The stretch's Jacobian z^(m - 1) counts the coordinates the move
# scales, so while h coordinates are held it is z^(n_dim - h - 1), as for a
# parked trans-dim slot.

"""
    default_burnin_hold(params) -> Dict{String,Float64}

The parameters the PT ensemble samplers hold early in burn-in when not told
otherwise (`burnin_hold = nothing`): every free intermediate-astrometry jitter
`sigma_iad_<instrument>`, at 0. Empty for a model without one, so a fit without
an IAD `"jitter": true` block runs exactly as it did before the hold existed.
"""
default_burnin_hold(params::Params) =
    Dict{String,Float64}(nm => 0.0 for nm in params.layout.unfrozen_names
                         if startswith(nm, "sigma_iad_"))

# What a sampler holds, resolved against its layout: unfrozen positions in
# ascending order, the values in each parameter's own units, and the last step
# held (the release comes at the end of it). `step == 0` holds nothing.
struct BurninHold
    dims::Vector{Int}
    values::Vector{Float64}
    step::Int
end

"""
    _burnin_hold(params, hold, frac, n_burnin; forbid, sampler) -> BurninHold

Resolve a sampler's `burnin_hold` / `burnin_hold_frac` keywords. `hold` is
`nothing` or `true` for `default_burnin_hold`, `false` for none, or a
dictionary of parameter name => value. The hold lasts `round(frac * n_burnin)`
steps. Throws on a name that is not a free parameter, a value outside its
prior's support, a full-circle angle (its window is re-charted during burn-in),
or a position in `forbid` (coordinates another move of `sampler` changes).
"""
function _burnin_hold(params::Params, hold, frac::Real, n_burnin::Integer;
                      forbid::AbstractVector{Int} = Int[],
                      sampler::AbstractString = "pt_emcee")
    0 <= frac <= 1 || throw(ArgumentError(
        "burnin_hold_frac is a fraction of burn-in, in [0, 1]; got $frac"))
    spec = if hold === nothing || hold === true
        default_burnin_hold(params)
    elseif hold === false
        Dict{String,Float64}()
    elseif hold isa AbstractDict
        Dict{String,Float64}(String(k) => Float64(v) for (k, v) in pairs(hold))
    else
        throw(ArgumentError("burnin_hold must be true, false, nothing or a " *
                            "dictionary of parameter name => value; got $(repr(hold))"))
    end
    L = params.layout
    circ = circular_indices(params)
    dims = Int[]
    for (nm, x) in spec
        j = findfirst(==(nm), L.unfrozen_names)
        j === nothing && throw(ArgumentError(
            "burnin_hold names $(repr(nm)), which is not a free parameter of this " *
            "model. Free parameters: $(join(L.unfrozen_names, ", "))"))
        ps = L.unfrozen_priors[j]
        s = log_scale_shift(ps)
        (ps.lo <= x <= ps.hi && (s === nothing || x + s > 0)) || throw(ArgumentError(
            "burnin_hold: $nm = $x is outside its prior's support [$(ps.lo), $(ps.hi)]"))
        j in circ && throw(ArgumentError(
            "burnin_hold: $nm is a full-circle angle, whose window $sampler " *
            "re-charts during burn-in; it cannot be held"))
        push!(dims, j)
    end
    sort!(dims)
    step = round(Int, frac * n_burnin)
    h = (isempty(dims) || step <= 0) ? BurninHold(Int[], Float64[], 0) :
        BurninHold(dims, Float64[spec[L.unfrozen_names[j]] for j in dims], step)
    _burnin_hold_forbid(h, params, forbid; sampler)
    return h
end

# Throw if `h` holds one of the positions `forbid`: coordinates another move of
# `sampler` writes, which would break the hold. Separate from `_burnin_hold` for
# a sampler that knows those moves only after its checkpoint fingerprint.
function _burnin_hold_forbid(h::BurninHold, params::Params, forbid;
                             sampler::AbstractString = "pt_emcee")
    for j in h.dims
        j in forbid && throw(ArgumentError(
            "burnin_hold: $sampler moves $(params.layout.unfrozen_names[j]) with a " *
            "move of its own (node flip, λ slide, or a trans-dim birth, death or " *
            "refinement); it cannot be held"))
    end
    return nothing
end

# The checkpoint fingerprint's share: nothing at all without a hold, so a run
# without one resumes from a checkpoint written before the hold existed.
_burnin_hold_fp(h::BurninHold, params::Params) = h.step == 0 ? (;) :
    (; burnin_hold = (params.layout.unfrozen_names[h.dims], h.values, h.step))

# Draw each held coordinate of `buf` from its prior, in the coordinates the
# sampler's walkers live in: y = log(x + s) where `flat_shift` gives an s.
function _burnin_hold_draw!(buf::AbstractVector{Float64}, h::BurninHold,
                            priors::AbstractVector, rng::AbstractRNG;
                            flat_shift = nothing)
    for j in h.dims
        x = quantile(priors[j], rand(rng))
        s = flat_shift === nothing ? nothing : flat_shift[j]
        buf[j] = s === nothing ? x : (x + s > 0 ? log(x + s) : -Inf)
    end
    return buf
end

function _burnin_hold_info(sampler::AbstractString, h::BurninHold,
                           params::Params, n_burnin::Integer)
    h.step == 0 && return nothing
    held = join(("$(params.layout.unfrozen_names[j]) = $(v)"
                 for (j, v) in zip(h.dims, h.values)), ", ")
    @info "$sampler: holding $held for the first $(h.step) of $n_burnin burn-in " *
          "steps, then drawing each walker's value from its prior. Until then the " *
          "fit is the one without these parameters, which keeps an IAD jitter from " *
          "absorbing the reflex before the orbit is found. The recorded chain is " *
          "ordinary MCMC; `burnin_hold = false` turns this off."
    return nothing
end
