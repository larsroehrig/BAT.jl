# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    logvalof(r::NamedTuple{(...,:logd,...)})::Real
    logvalof(r::LogDVal)::Real

Extract the log-density value from a result `r`.

Examples:

```julia
logvalof((..., logd = logd, ...)) == logd
logvalof(LogDVal(logd)) == logd
```
"""
function logvalof end
export logvalof

function logvalof(d::Real)
    throw(ArgumentError("Can't the a logarithmic value for d, unknown if it represents a lin or log value itself."))
end

logvalof(r::NamedTuple) = r.logd



"""
    LogDVal{T<:Real}

Represent the logarithm of the value of a statistical density at some point.
`LogDVal` provides means to unambiguously distinguish between linear and
log result values of density functions.

Constructor:

    LogDVal(logd::Real)

Use [`logvalof`](@ref) to extract the actual log-density value from
a `LogDVal`:

```julia
    logvalof(LogDVal(d)) == d
```
"""
struct LogDVal{T<:Real}
    value::T
end

export LogDVal

logvalof(d::LogDVal) = d.value
