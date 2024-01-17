using Statistics

"""
    create_fdr(spectrogram, Nr::Integer)

Create an uninitialized `Matrix` suitable for use with `intfdr!`, `fftfdr!`, or
`zdtfdr!` and the given `spectrogram` and `Nr` (number of rates).  The returned
`Matrix` will be similar to `spectrogram` in type and size of first dimension,
but its second dimension will be `Nr`.
"""
function create_fdr(spectrogram, Nr::Integer)
    Nf = size(spectrogram, 1)
    similar(spectrogram, Nf, Nr)
end

"""
    create_fdr(spectrogram, rates)

Create an uninitialized `Matrix` suitable for use with `intfdr!`, `fftfdr!`, or
`zdtfdr!` and the given `spectrogram` and `rates`.  The returned `Matrix` will
be similar to `spectrogram` in type and size of first dimension, but its second
dimension will be sized by the length of `rates`.
"""
function create_fdr(spectrogram, rates)
    create_fdr(spectrogram, length(rates))
end

"""
    fdrstats(fdr) -> (mean, std)
    fdrstats(fdrs) -> (mean, std)

Compute the mean and standard deviation of the Frequency-Drift-Rate (FDR) Matrix
`fdr` or matrices `fdrs`.  The mean is calculated as the mean of the first
column, i.e. along the frequency axis for the first drift rate of `fdr` or the
first matrix of `fdrs`).  The standard deviation (aka sigma) value used is the
minimum standard deviation of all columns of `fdr` or all columns of all
matrices in `fdrs`.
"""
function fdrstats(fdr)
    m = mean(@view fdr[:,1])
    s = minimum(std(fdr, dims=1))
    (m, s)
end

function fdrstats(fdrs::AbstractVector)
    fdr1 = fdrs[1]
    m = mean(@view fdr1[:,1])
    s = minimum(minimum.(std.(fdrs, dims=1)))
    (m, s)
end

"""
    fdrnormalize!(fdr[, m, s]) -> same fdr (normalized in place)
    fdrnormalize!(fdrs[, m, s]) -> same fdrs (normalized in place)

Normalize the Frequency-Drift-Rate (FDR) Matrix `fdr` or matrices `fdrs`
in-place by subtracting the mean and dividing by the standard deviation.  The
mean is calculated as the mean of the first column, i.e. along the frequency
axis for the first drift rate of `fdr` or the first matrix of `fdrs`).  The
standard deviation (aka sigma) value used is the minimum standard deviation of
all columns of `fdr` or all columns of all matrices in `fdrs`.  The mean and
standard deviation may also be given explicitly as `m` and `s`, respectively.
"""
function fdrnormalize!(fdr, m, s)
    fdr .= (fdr .- m) ./ s
    return fdr
end

function fdrnormalize!(fdr)
    m, s = fdrstats(fdr)
    return fdrnormalize!(fdr, m, s)
end

function fdrnormalize!(fdrs::AbstractVector, m, s)
    for fdr in fdrs
        fdr .= (fdr .- m) ./ s
    end
    return fdrs
end

function fdrnormalize!(fdrs::AbstractVector)
    m, s = fdrstats(fdrs)
    return fdrnormalize!(fdrs, m, s)
end

"""
    fdrsynchronize(::Type{<:AbstractArray})

The default implementaion of this function does nothing, but it should be called
whenever synchronization might be needed.  Methods can be defined for types that
are more specific than `AbstractArray` when they have synchroniztion
requirements and mechanisms (e.g. `CuArray`).
"""
function fdrsynchronize(::Type{<:AbstractArray})
end

"""
    clamprange(r::CartesianIndices, array::AbstractArray) -> CartesianIndices

Clamp the CartesianIndices range `r` to the dimensions of `a`.
"""
function clamprange(r::CartesianIndices, array::AbstractArray)
    lo, hi = extrema(r)
    lolim, hilim = extrema(CartesianIndices(array))
    clamp(lo, lolim):step(r):clamp(hi, hilim)
end

"""
    clusterrange(cijs, array, border=0) -> CartesianIndices

Compute the rectangular region of `array` that contains CartesianIndex values in
`cijs`.  The `border` argument may be used to add a border to the region.  When
given, `border` must be an `Int` (same border size in all directions) or a
`CartesianIndex` or `Tuple` of dimensionality/length of `ndims(array)` giving
the border size for both sides of each dimension.  The returned
`CartesianIndices` region will be clipped to the size of `array` if necessary.
"""
function clusterrange(
    cijs::AbstractVector{CartesianIndex{N}},
    array::AbstractArray{T,N},
    border::CartesianIndex{N}=CartesianIndex(ntuple(_->0, N))
) where {T,N}
    isempty(cijs) && error("no points in range")
    lo, hi = extrema(cijs)
    lo -= border
    hi += border
    lolim, hilim = extrema(CartesianIndices(array))
    clamp(lo, lolim):clamp(hi, hilim)
end

function clusterrange(cijs, array, border)
    clusterrange(cijs, array, CartesianIndex(border))
end

function clusterrange(cijs, array::AbstractArray{T,N}, border::Int) where {T,N}
    clusterrange(cijs, array, ntuple(_->border, N))
end
