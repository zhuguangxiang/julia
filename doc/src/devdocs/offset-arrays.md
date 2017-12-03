# Arrays with custom indices

Conventionally, Julia's
arrays are indexed starting at 1, whereas some other languages start numbering at 0, and yet others
(e.g., Fortran) allow you to specify arbitrary starting indices.  While there is much merit in
picking a standard (i.e., 1 for Julia), there are some algorithms which simplify considerably
if you can index outside the range `1:size(A,d)` (and not just `0:size(A,d)-1`, either).
Consequently, Julia supports arrays with arbitrary indices.
Such array types are expected to be supplied through packages.

The purpose of this page is to address the question, "what do I have to do to support such arrays
in my own code?"  First, let's address the simplest case: if you know that your code will never
need to handle arrays with unconventional indexing, the answer is "nothing." Optionally, you can call
`Base.assert_oneindex(A)` to enforce conventional indexing; it will throw an error if `A` has
unconventional indices, but if `A` has conventional indexing this gets compiled away and
consequently has no performance cost.

## Generalizing existing code

As an overview, the steps are:

  * replace many uses of `size` with `indices`
  * replace `1:length(A)` with `eachindex(A)`, or in some cases `linearindices(A)`
  * replace `length(A)` with `length(linearindices(A))`
  * replace explicit allocations like `Array{Int}(size(B))` with `similar(Array{Int}, indices(B))`

These are described in more detail below.

### Background

Because unconventional indexing can break implicit assumptions about how arrays work, its use
can be a source of bugs, of which the most serious occur in conjunction with `@inbounds`.
For example, consider the following function:

```julia
function mycopy!(dest::AbstractVector, src::AbstractVector)
    length(dest) == length(src) || throw(DimensionMismatch("vectors must match"))
    # OK, now we're safe to use @inbounds, right? Wrong!
    for i = 1:length(src)
        @inbounds dest[i] = src[i]
    end
    dest
end
```

This code implicitly assumes that vectors are indexed from 1, or rather, from the same starting index.
When that is not the case, this function might address invalid memory and lead to a segfault.
If you do get segfaults, to help locate
the cause try running julia with the option `--check-bounds=yes`.

Below, a more general API is described that allows you to support arbitrary indexing safely and efficiently.

### Using `indices` for bounds checks and loop iteration

`indices(A)` (reminiscent of `size(A)`) returns a tuple of `AbstractUnitRange` objects, specifying
the range of valid indices along each dimension of `A`.  When `A` has unconventional indexing,
the ranges may not start at 1.  If you just want the range for a particular dimension `d`, there
is `indices(A, d)`.

If you want to iterate over every index of an array, the easiest and most efficient
approach is to use [`eachindex`](@ref). If you want
to be certain that you'll be using a multidimensional ([`CartesianIndex`](@ref)) indexing object,
then

```julia
for I in CartesianRange(indices(A))
    # body
end
```

is recommended.

Base implements a custom range type, `OneTo`, where `OneTo(n)` means the same thing as `1:n` but
in a form that guarantees (via the type system) that the lower index is 1. For any new [`AbstractArray`](@ref)
type, this is the default returned by `indices`, and it indicates that this array type uses "conventional"
1-based indexing.

### Linear indexing (`linearindices`)

Some algorithms are most conveniently (or efficiently) written in terms of a single linear index, `A[i]` even if `A` is multi-dimensional. Regardless of the array's native indices, "true" linear indices always range from `1:length(A)`. However, this raises an ambiguity for one-dimensional arrays (a.k.a., [`AbstractVector`](@ref)): does `v[i]` mean linear indexing , or Cartesian indexing with the array's native indices? In Julia, "native" indices take precedence. Consequently, there is a convenience function,`linearindices(A)`, that will return `indices(A, 1)` if A is an `AbstractVector`, and the equivalent of `1:length(A)` otherwise.

Using `indices` and `linearindices`, here is one way you could safely rewrite `mycopy!`:

```julia
function mycopy!(dest::AbstractVector, src::AbstractVector)
    indices(dest) == indices(src) || throw(DimensionMismatch("vectors must match"))
    for i in linearindices(src)
        @inbounds dest[i] = src[i]
    end
    dest
end
```

### Allocating storage using generalizations of `similar`

Storage is often allocated with `Array{Int}(uninitialized, dims)` or `similar(A, args...)`. When the result needs
to match the indices of some other array, this may not always suffice. The generic replacement
for such patterns is to use `similar(storagetype, shape)`.  `storagetype` indicates the kind of
underlying "conventional" behavior you'd like, e.g., `Array{Int}` or `BitArray` or even `dims->zeros(Float32, dims)`
(which would allocate an all-zeros array). `shape` is a tuple of [`Integer`](@ref) or
`AbstractUnitRange` values, specifying the indices that you want the result to use. Note that
a convenient way of producing an all-zeros array that matches the indices of A is simply `zeros(A)`.

Let's walk through a couple of explicit examples. First, if `A` has conventional indices, then
`similar(Array{Int}, indices(A))` would end up calling the equivalent of `Array{Int}(size(A))`,
and thus return an array.  If `A` is an `AbstractArray` type with unconventional indexing,
then `similar(Array{Int}, indices(A))`
should return something that "behaves like" an `Array{Int}` but with a shape (including indices)
that matches `A`.
(The most obvious implementation is to allocate an `Array{Int}(uninitialized, length.(inds))` and
then "wrap" it in a type that shifts the indices.)

Note also that `similar(Array{Int}, (indices(A, 2),))` would allocate an `AbstractVector{Int}`
(i.e., 1-dimensional array) that matches the indices of the columns of `A`.

## Writing custom array types with non-1 indexing

Most of the methods you'll need to define are standard for any `AbstractArray` type, see [Abstract Arrays](@ref man-interface-array).
This page focuses on the steps needed to define unconventional indexing.

### Custom `AbstractUnitRange` types

If you're writing a non-1 indexed array type, you will want to specialize `indices` so it returns
a custom `AbstractUnitRange`.  The advantage of a custom type
is that it "signals" the allocation type for functions like `similar`. If we're writing an array
type for which indexing will start at 0, we likely want to begin by creating a new `AbstractUnitRange`,
`ZeroRange`, where `ZeroRange(n)` is equivalent to `0:n-1`.

In general, you should probably *not* export `ZeroRange` from your package: there may be other
packages that implement their own `ZeroRange`, and having multiple distinct `ZeroRange` types
is (perhaps counterintuitively) an advantage: `ModuleA.ZeroRange` indicates that `similar` should
create a `ModuleA.ZeroArray`, whereas `ModuleB.ZeroRange` indicates a `ModuleB.ZeroArray` type.
This design allows peaceful coexistence among many different custom array types.

Note that the Julia package [CustomUnitRanges.jl](https://github.com/JuliaArrays/CustomUnitRanges.jl)
can sometimes be used to avoid the need to write your own `ZeroRange` type.

### Specializing `indices`

Once you have your `AbstractUnitRange` type, then specialize `indices`. For example:

```julia
Base.indices(A::ZeroArray) = map(ZeroRange, A.size)
```

where here we imagine that `ZeroArray` has a field called `size` (there would be other ways to
implement this).

In some cases, the fallback definition for `indices(A, d)`:

```julia
indices(A::AbstractArray{T,N}, d) where {T,N} = d <= N ? indices(A)[d] : OneTo(1)
```

may not be what you want: you may need to specialize it to return something other than `OneTo(1)`
when `d > ndims(A)`.  Likewise, in `Base` there is a dedicated function `indices1` which is equivalent
to `indices(A, 1)` but which avoids checking (at runtime) whether `ndims(A) > 0`. (This is purely
a performance optimization.)  It is defined as:

```julia
indices1(A::AbstractArray{T,0}) where {T} = OneTo(1)
indices1(A::AbstractArray) = indices(A)[1]
```

If the first of these (the zero-dimensional case) is problematic for your custom array type, be
sure to specialize it appropriately.

### Specializing `similar`

Given your custom `ZeroRange` type, then you should also add the following two specializations
for `similar`:

```julia
function Base.similar(A::AbstractArray, T::Type, shape::Tuple{ZeroRange,Vararg{ZeroRange}})
    # body
end

function Base.similar(f::Union{Function,DataType}, shape::Tuple{ZeroRange,Vararg{ZeroRange}})
    # body
end
```

Both of these should allocate your custom array type.

### Specializing `reshape`

Optionally, define a method

```
Base.reshape(A::AbstractArray, shape::Tuple{ZeroRange,Vararg{ZeroRange}}) = ...
```

and you can `reshape` an array so that the result has custom indices.

## Summary

Writing code that doesn't make assumptions about indexing requires a few extra abstractions, but
hopefully the necessary changes are relatively straightforward.
