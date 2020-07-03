# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    BAT.varof(r)

*Experimental feature, not part of stable public API.*

Extracts the variate from a value `r` returned by some function.

By default, `varof` supports `NamedTuple`s like `(v = ..., ...), and returns
`r.v`.

The behavior of `varof` should be specialized for custom return types.
"""
function varof end

varof(r::NamedTuple{(:v)}) = r.v
varof(r::NamedTuple{(:v, :ladj)}) = r.v


"""
    BAT.varof(s::DensitySample)

*Experimental feature, not part of stable public API.*

Returns the variate in `s`, i.e. `s.v`.
"""
varof(s::DensitySample) = s.v
# ToDo: Specialize `varof.(sv::DensitySampleVector)`



struct EvalVarOf{F<:Function} <: Function
    f::F
end

"""
    BAT.varof(f::Function)

*BAT-internal, not part of stable public API.*

Equivalent to `(args...; kwargs...) -> varof(f(args...; kwargs...))`.
"""
varof(f::Function) = EvalVarOf(f)

@inline EvalVarOf(f, args...; kwargs...) = varof(f(args...; kwargs...))



"""
    ladjof(r, f::Function, v::Number)
    ladjof(r, f::Function, v::AbstractVector{<:Number})

*BAT-internal, not part of stable public API.*

Returns the phase space volume factor of the change-of-variables transform
`r = trafo(v)`.

If `r` is a `NamedTuple` like `(v = new_v, ladj = ladj_at_v)` then ladjof
will return `r.ladj`. Otherwise, will return
`logabsdet(jacobian(varof(f), x))` by default, using automatic
differentiation.

Functions `f` may provice specialized methods for `ladjof`.
"""
function ladjof end

ladjof(r::NamedTuple{(:v, :ladj)}, f::Function, x::Real) = r.ladj

ladjof(r::NamedTuple{(:v, :ladj)}, f::Function, x::AbstractMatrix{<:Real,N}) where N = r.ladj

ladjof(r::Real, f::Function, x::Real) =
    _stable_logabsdet(ForwardDiff.derivative(varof(f), x))

ladjof(r::AbstractMatrix{<:Real,N}, f::Function, x::AbstractMatrix{<:Real,N}) where N =
    _stable_logabsdet(ForwardDiff.jacobian(varof(f), x))

_stable_logabsdet(x::T) where {T<:Real} = log(abs(x + eps(T)))
_stable_logabsdet(A::AbstractMatrix{T}) where {T<:Real} = logabsdet(A .+ eps(T))


function with_ladj(f::Function, v)
    r = f(v)
    new_v = varof(r)
    ladj = ladjof(r, f, v)
    (v = new_v, ladj = ladj)
end


"""
    abstract type VariateTransform <: Function

*BAT-internal, not part of stable public API.*

Abstract super-type for change-of-variables transformations.

Subtypes (e.g. `SomeTrafo <: VariateTransform`) must implement:

* `r = (trafo::SomeTrafo)(v)`

for real values and/or real-valued vectors `v`. The return value `r` must
support `getvar(r)`.

Subtypes may also specialize

* `ladjof(r, trafo::SomeTrafo, v::Number)`
"""
abstract type VariateTransform <: Function end


# ToDo: (f::VariateTransform)(s::DensitySample)::DensitySample
# ToDo: (f::VariateTransform)(s::DensitySampleVector)::DensitySampleVector



struct InvVT{T<:VariateTransform} <: VariateTransform
    orig::T
end


Base.inv(trafo::VariateTransform) = InvVT(trafo)
Base.inv(trafo::InvVT) = trafo.orig


function ladjof(r::NamedTuple{(:v)}, f::InvVT, v::Any) where N
    - ladjof((v = v,), inv(f), varof(r))
end



struct VariateTransformChain{N,T<:NTuple{N,VariateTransform}} <: VariateTransform
    transforms::T
end


import Base.∘
∘(a::VariateTransform, b::VariateTransform) = VariateTransformChain((b, a))
∘(a::VariateTransform, b::VariateTransformChain) = VariateTransformChain((b, a.transforms...))
∘(a::VariateTransformChain, b::VariateTransform) = VariateTransformChain((b.transforms..., a))
∘(a::VariateTransformChain, b::VariateTransformChain) = VariateTransformChain((b.transforms..., a.transforms...))


Base.inv(trafo::VariateTransformChain) = VariateTransformChain(reverse(map(inv, trafo.transforms)))


_eval_var_trafos(x::Any) = x

_eval_var_trafos(x::Any, f::VariateTransform) = f(x)

function _eval_var_trafos(v::Any, f1::VariateTransform, f2::VariateTransform..., fs::VariateTransform...)
    r1 = f1(x)
    v1 = 
    rs = _eval_var_trafos(varof(f1(x)), f2, ts...)

end

function (trafo::VariateTransformChain)(x::Any)
    _eval_var_trafos(x, trafo.transforms...)
end


"""
    BAT.to_unitvol(d::Distribution)::VariateTransform
    BAT.to_unitvol(d::AbstractDensity)::VariateTransform

*BAT-internal, not part of stable public API.*

Returns a change-of-variables transformation 
"""
function to_unitvol end


struct ToUnitVol{D<:Union{Distribution,AbstractDensity}} <: VariateTransform
    d::D
end


to_unitvol(d::Union{Distribution,AbstractDensity}) = ToUnitVol(d)


function to_unitvol(d::Truncated)
    d = trafo.d
    utrafo = to_unit(d)
    lb, ub = minimum(d), maximum(d)
    lb_unit, ub_unit = utrafo(lb), utrafo(ub)
    utrafo(lb), utrafo(ub)
    trunctrafo = to_unit(Uniform(lb, ub))

    trunctrafo ∘ utrafo(x)
end


double_pmass_beyond_cdf_value(u::Real) = 1 - abs(2*u - 1)

function (trafo::ToUnitVol{<:Distribution{Univariate,Continuous}})(x::Real)
    d = trafo.d
    u = cdf(d, x)
    
    is_relevant = double_pmass_beyond_cdf_value(u) > 10^-10
    x_rec = invlogcdf(d, log(u))
    if is_relevant && !approx(x, x_rec, rtol = 10^-6)
        throw(ArgumentError("Conversion to unit interval not precise enough"))
    end

    (v = u,)
end

function (trafo::InvVT{<:ToUnitVol{<:Distribution{Univariate,Continuous}}})(u::Real)
    d = trafo.d
    x = invlogcdf(d, log(u))

    is_relevant = double_pmass_beyond_cdf_value(u) > 10^-10
    u_rec = cdf(d, x)
    if is_relevant && !approx(u, u_rec, rtol = 10^-6)
        throw(ArgumentError("Conversion from unit interval not precise enough"))
    end

    (v = x,)
end


function (trafo::ToUnitVol{<:Product})(x::Real)
end

function (trafo::InvVT{<:ToUnitVol{<:Product}})(u::Real)
end



(trafo::ToUnitVol{<:Logistic})(x::Real) = logistic_to_unit(x, trafo.d.μ, trafo.d.θ)
(trafo::InvVT{<:ToUnitVol{<:Logistic}})(u::Real) = unit_to_logistic(u, trafo.d.μ, trafo.d.θ)

logistic_to_unit(x::Real, mu::Real, theta::Real) = (v = inv(exp(-(x - mu) / theta) + one(x)),)  # Exact CDF
unit_to_logistic(u::Real, mu::Real, theta::Real) = (v = log(u / (one(u) - u)) * theta + mu,)  # Exact inverse CDF

logistic_to_unit(x::Real) = logistic_to_unit(x, 0, 1)
unit_to_logistic(u::Real) = unit_to_logistic(u, 0, 1)


(trafo::ToUnitVol{<:Uniform})(x::Real) = (v = uniform_to_unit(x, trafo.d.a, trafo.d.b),)
(trafo::InvVT{<:ToUnitVol{<:Uniform}})(x::Real) = (v = unit_to_uniform(u, trafo.d.a, trafo.d.b),)

uniform_to_unit(x::Real, a::Real, b::Real) = (x - a) / (b - a)  # Exact CDF
unit_to_uniform(u::Real, a::Real, b::Real) = (b - a) * u + a  # Exact inverse CDF


(trafo::ToUnitVol{<:Normal})(x::Real) = (v = approxnormal_to_unit(x, trafo.d.μ, trafo.d.σ),)
(trafo::InvVT{<:ToUnitVol{<:Normal}})(x::Real) = (v = normal_to_unit(u, trafo.d.μ, trafo.d.σ),)

approxnormal_to_unit(x::Real, mu::Real, sigma::Real) = infinite_to_unit((x - mu) / sigma * pi / 2)
unit_to_approxnormal(u::Real, mu::Real, sigma::Real) = unit_to_infinite(u) * sigma * 2 / π + mu

normal_to_unit(x::Real, mu::Real, sigma::Real) = erfc((mu - x) / sigma * invsqrt2) / 2
unit_to_normal(u::Real, mu::Real, sigma::Real) = mu - erfcinv(2 * u) * sigma / invsqrt2


(trafo::ToUnitVol{<:Exponential})(x::Real) = (v = exp_to_unit(x, 0, trafo.d.θ),)
(trafo::InvVT{<:ToUnitVol{<:Exponential}})(x::Real) = (v = unit_to_exp(u, 0, trafo.d.θ),)

exp_to_unit(x::Real, theta::Real) = exp(x / -theta)  # Exact CDF
unit_to_exp(u::Real, theta::Real) = -theta * log(u)  # Exact inverse CDF

# lefttruncexp_to_unit(x::Real, x0::Real, theta::Real) = exp((x - x0) / -theta)  # Exact CDF
# unit_to_lefttruncexp(u::Real, x0::Real, theta::Real) = -theta * log(u) + x0  # Exact inverse CDF


(trafo::ToUnitVol{<:Weibull})(x::Real) = (v = weibull_to_unit(x, trafo.d.α, trafo.d.θ),)
(trafo::InvVT{<:ToUnitVol{<:Exponential}})(x::Real) = (v = unit_to_weibull(u, trafo.d.α, trafo.d.θ),)

weibull_to_unit(x::Real, alpha::Real, theta::Real) = 1 - exp(- (x / theta)^alpha)  # Exact CDF
unit_to_weibull(x::Real, alpha::Real, theta::Real) = (-log(1 - x))^(1/alpha) * theta  # Exact inverse CDF
