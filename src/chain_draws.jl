# One parameter's draws out of an MCMCChains.Chains, as a flat vector.

using MCMCChains

"""
    _draws(chains, name) -> Vector

`vec(Array(chains[name]))`: one parameter's draws, iterations fastest, then
chains. `chains[name]` copies the column and `Array` copies it again; this
reads the chain array directly, in one copy, with the same values in the same
order. Anything else (not a `Chains`, a name it lacks, a non-Float64 array)
goes the long way, errors included.
"""
function _draws(chains::MCMCChains.Chains, name::Union{Symbol, AbstractString})
    d = chains.value.data
    if eltype(d) === Float64 && ndims(d) == 3
        j = findfirst(==(Symbol(name)), names(chains))
        j === nothing || return vec(d[:, j, :])
    end
    return vec(Array(chains[name]))
end
_draws(chains, name) = vec(Array(chains[name]))
