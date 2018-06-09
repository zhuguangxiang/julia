# This file is a part of Julia. License is MIT: https://julialang.org/license

##### mean #####

"""
    mean(itr)

Compute the mean of all elements in a collection.

!!! note
    If array contains `NaN` or [`missing`](@ref) values, the result is also
    `NaN` or `missing` (`missing` takes precedence if array contains both).
    Use the [`skipmissing`](@ref) function to omit `missing` entries and compute the
    mean of non-missing values.

# Examples
```jldoctest
julia> mean(1:20)
10.5

julia> mean([1, missing, 3])
missing

julia> mean(skipmissing([1, missing, 3]))
2.0
```
"""
mean(itr) = mean(identity, itr)

"""
    mean(f::Function, itr)

Apply the function `f` to each element of collection `itr` and take the mean.

```jldoctest
julia> mean(√, [1, 2, 3])
1.3820881233139908

julia> mean([√1, √2, √3])
1.3820881233139908
```
"""
function mean(f::Callable, itr)
    y = iterate(itr)
    if y == nothing
        throw(ArgumentError("mean of empty collection undefined: $(repr(itr))"))
    end
    count = 1
    value, state = y
    f_value = f(value)
    total = reduce_first(add_sum, f_value)
    y = iterate(itr, state)
    while y !== nothing
        value, state = y
        total += f(value)
        count += 1
        y = iterate(itr, state)
    end
    return total/count
end
mean(f::Callable, A::AbstractArray) = sum(f, A) / _length(A)

"""
    mean!(r, v)

Compute the mean of `v` over the singleton dimensions of `r`, and write results to `r`.

# Examples
```jldoctest
julia> v = [1 2; 3 4]
2×2 Array{Int64,2}:
 1  2
 3  4

julia> mean!([1., 1.], v)
2-element Array{Float64,1}:
 1.5
 3.5

julia> mean!([1. 1.], v)
1×2 Array{Float64,2}:
 2.0  3.0
```
"""
function mean!(R::AbstractArray, A::AbstractArray)
    sum!(R, A; init=true)
    x = max(1, _length(R)) // _length(A)
    R .= R .* x
    return R
end

"""
    mean(A::AbstractArray; dims)

Compute the mean of an array over the given dimensions.

# Examples
```jldoctest
julia> A = [1 2; 3 4]
2×2 Array{Int64,2}:
 1  2
 3  4

julia> mean(A, dims=1)
1×2 Array{Float64,2}:
 2.0  3.0

julia> mean(A, dims=2)
2×1 Array{Float64,2}:
 1.5
 3.5
```
"""
mean(A::AbstractArray; dims=:) = _mean(A, dims)

_mean(A::AbstractArray{T}, region) where {T} = mean!(reducedim_init(t -> t/2, +, A, region), A)
_mean(A::AbstractArray, ::Colon) = sum(A) / _length(A)

##### median & quantiles #####

"""
    middle(x)

Compute the middle of a scalar value, which is equivalent to `x` itself, but of the type of `middle(x, x)` for consistency.
"""
middle(x::Union{Bool,Int8,Int16,Int32,Int64,Int128,UInt8,UInt16,UInt32,UInt64,UInt128}) = Float64(x)
# Specialized functions for real types allow for improved performance
middle(x::AbstractFloat) = x
middle(x::Real) = (x + zero(x)) / 1

"""
    middle(x, y)

Compute the middle of two reals `x` and `y`, which is
equivalent in both value and type to computing their mean (`(x + y) / 2`).
"""
middle(x::Real, y::Real) = x/2 + y/2

"""
    middle(range)

Compute the middle of a range, which consists of computing the mean of its extrema.
Since a range is sorted, the mean is performed with the first and last element.

```jldoctest
julia> middle(1:10)
5.5
```
"""
middle(a::AbstractRange) = middle(a[1], a[end])

"""
    middle(a)

Compute the middle of an array `a`, which consists of finding its
extrema and then computing their mean.

```jldoctest
julia> a = [1,2,3.6,10.9]
4-element Array{Float64,1}:
  1.0
  2.0
  3.6
 10.9

julia> middle(a)
5.95
```
"""
middle(a::AbstractArray) = ((v1, v2) = extrema(a); middle(v1, v2))

"""
    median!(v)

Like [`median`](@ref), but may overwrite the input vector.
"""
function median!(v::AbstractVector)
    isempty(v) && throw(ArgumentError("median of an empty array is undefined, $(repr(v))"))
    eltype(v)>:Missing && any(ismissing, v) && return missing
    eltype(v)>:AbstractFloat && any(isnan, v) && return NaN
    inds = axes(v, 1)
    n = _length(inds)
    mid = div(first(inds)+last(inds),2)
    if isodd(n)
        return middle(partialsort!(v,mid))
    else
        m = partialsort!(v, mid:mid+1)
        return middle(m[1], m[2])
    end
end
median!(v::AbstractArray) = median!(vec(v))

"""
    median(itr)

Compute the median of all elements in a collection.
For an even number of elements no exact median element exists, so the result is
equivalent to calculating mean of two median elements.

!!! note
    If `itr` contains `NaN` or [`missing`](@ref) values, the result is also
    `NaN` or `missing` (`missing` takes precedence if `itr` contains both).
    Use the [`skipmissing`](@ref) function to omit `missing` entries and compute the
    median of non-missing values.

# Examples
```jldoctest
julia> median([1, 2, 3])
2.0

julia> median([1, 2, 3, 4])
3.5

julia> median([1, 2, missing, 4])
missing

julia> median(skipmissing([1, 2, missing, 4]))
2
```
"""
median(itr) = median!(collect(itr))

"""
    median(A::AbstractArray; dims)

Compute the median of an array along the given dimensions.

# Examples
```jldoctest
julia> median([1 2; 3 4], dims=1)
1×2 Array{Float64,2}:
 2.0  3.0
```
"""
median(v::AbstractArray; dims=:) = _median(v, dims)

_median(v::AbstractArray, dims) = mapslices(median!, v, dims)

_median(v::AbstractArray{T}, ::Colon) where {T} = median!(copyto!(Array{T,1}(undef, _length(v)), v))

# for now, use the R/S definition of quantile; may want variants later
# see ?quantile in R -- this is type 7
"""
    quantile!([q::AbstractArray, ] v::AbstractVector, p; sorted=false)

Compute the quantile(s) of vector `v` at a specified probability or vector or tuple of
probabilities `p` on the interval [0,1]. If `p` is a vector, an optional
output array `q` may also be specified. (If not provided, a new output array is created.)
The keyword argument `sorted` indicates whether `v` can be assumed to be sorted; if
`false` (the default), then the elements of `v` will be partially sorted in-place.

Quantiles are computed via linear interpolation between the points `((k-1)/(n-1), v[k])`,
for `k = 1:n` where `n = length(v)`. This corresponds to Definition 7 of Hyndman and Fan
(1996), and is the same as the R default.

!!! note
    An `ArgumentError` is thrown if `v` contains `NaN` or [`missing`](@ref) values.

* Hyndman, R.J and Fan, Y. (1996) "Sample Quantiles in Statistical Packages",
  *The American Statistician*, Vol. 50, No. 4, pp. 361-365

# Examples
```jldoctest
julia> x = [3, 2, 1];

julia> quantile!(x, 0.5)
2.0

julia> x
3-element Array{Int64,1}:
 1
 2
 3

julia> y = zeros(3)

julia> quantile!(y, x, [0.1, 0.5, 0.9]) === y
true

julia> y
3-element Array{Float64,1}:
 1.2
 2.0
 2.8
```
"""
function quantile!(q::AbstractArray, v::AbstractVector, p::AbstractArray;
                   sorted::Bool=false)
    if size(p) != size(q)
        throw(DimensionMismatch("size of p, $(size(p)), must equal size of q, $(size(q))"))
    end
    isempty(q) && return q

    minp, maxp = extrema(p)
    _quantilesort!(v, sorted, minp, maxp)

    for (i, j) in zip(eachindex(p), eachindex(q))
        @inbounds q[j] = _quantile(v,p[i])
    end
    return q
end

quantile!(v::AbstractVector, p::AbstractArray; sorted::Bool=false) =
    quantile!(similar(p,float(eltype(v))), v, p; sorted=sorted)

quantile!(v::AbstractVector, p::Real; sorted::Bool=false) =
    _quantile(_quantilesort!(v, sorted, p, p), p)

function quantile!(v::AbstractVector, p::Tuple{Vararg{Real}}; sorted::Bool=false)
    isempty(p) && return ()
    minp, maxp = extrema(p)
    _quantilesort!(v, sorted, minp, maxp)
    return map(x->_quantile(v, x), p)
end

# Function to perform partial sort of v for quantiles in given range
function _quantilesort!(v::AbstractArray, sorted::Bool, minp::Real, maxp::Real)
    isempty(v) && throw(ArgumentError("empty data vector"))

    if !sorted
        lv = length(v)
        lo = floor(Int,1+minp*(lv-1))
        hi = ceil(Int,1+maxp*(lv-1))

        # only need to perform partial sort
        sort!(v, 1, lv, Sort.PartialQuickSort(lo:hi), Base.Sort.Forward)
    end
    ismissing(v[end]) && throw(ArgumentError("quantiles are undefined in presence of missing values"))
    isnan(v[end]) && throw(ArgumentError("quantiles are undefined in presence of NaNs"))
    return v
end

# Core quantile lookup function: assumes `v` sorted
@inline function _quantile(v::AbstractVector, p::Real)
    0 <= p <= 1 || throw(ArgumentError("input probability out of [0,1] range"))

    lv = length(v)
    f0 = (lv - 1)*p # 0-based interpolated index
    t0 = trunc(f0)
    h  = f0 - t0
    i  = trunc(Int,t0) + 1

    T  = promote_type(eltype(v), typeof(v[1]*h))

    if h == 0
        return convert(T, v[i])
    else
        a = v[i]
        b = v[i+1]
        if isfinite(a) && isfinite(b)
            return convert(T, a + h*(b-a))
        else
            return convert(T, (1-h)*a + h*b)
        end
    end
end


"""
    quantile(itr, p; sorted=false)

Compute the quantile(s) of collection `itr` at a specified probability or vector or tuple of
probabilities `p` on the interval [0,1]. The keyword argument `sorted` indicates whether
`itr` can be assumed to be sorted.

Quantiles are computed via linear interpolation between the points `((k-1)/(n-1), v[k])`,
for `k = 1:n` where `n = length(v)`. This corresponds to Definition 7 of Hyndman and Fan
(1996), and is the same as the R default.

!!! note
    An `ArgumentError` is thrown if collection contains `NaN` or [`missing`](@ref) values.
    Use the [`skipmissing`](@ref) function to omit `missing` entries and compute the
    quantiles of non-missing values.

- Hyndman, R.J and Fan, Y. (1996) "Sample Quantiles in Statistical Packages",
  *The American Statistician*, Vol. 50, No. 4, pp. 361-365

# Examples
```jldoctest
julia> quantile(0:20, 0.5)
10.0

julia> quantile(0:20, [0.1, 0.5, 0.9])
3-element Array{Float64,1}:
  2.0
 10.0
 18.0

julia> quantile(skipmissing([1, 10, missing]), 0.5)
5.5
 ```
"""
quantile(itr, p; sorted::Bool=false) = quantile!(collect(itr), p, sorted=sorted)

quantile(v::AbstractVector, p; sorted::Bool=false) =
    quantile!(sorted ? v : copymutable(v), p; sorted=sorted)
