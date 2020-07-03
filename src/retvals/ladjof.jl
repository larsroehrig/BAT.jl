# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    ladjof(r)
    ladjof(f::Function, v::Any)
    ladjof(r::Any, f::Function, v::Any)

*Experimental feature, not part of stable public API.*

Extracts the variate from a value `r` returned by some function.

By default, `varof` supports `NamedTuple`s like `(v = ..., ...), and returns
`r.v`.

The behavior of `varof` should be specialized for custom return types.
"""
function ladjof end
export ladjof

ladjof(r::NamedTuple) = r.gradlogd
