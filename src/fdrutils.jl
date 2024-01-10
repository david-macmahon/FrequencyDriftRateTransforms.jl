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
    fdrnormalize!(fdr) -> same fdr (normalized in place)

Normalize the Frequency-Drift-Rate (FDR) Matrix `fdr` in-place by subtracting
the mean and dividing by the standard deviation.  The mean is calculated as the
mean of the first column (i.e. along the frequency axis for the first drift
rate).  The standard deviation (aka sigma) value used is the minimum standard
deviation of all columns of `fdr`.
"""
function fdrnormalize!(fdr)
    m = mean(@view fdr[:,1])
    s = minimum(std(fdr, dims=1))
    fdr .= (fdr .- m) ./ s
end

"""
    fdrnormalize!(fdrs::AbstractVector) -> same fdr (normalized in place)

Normalize the Frequency-Drift-Rate (FDR) Matrices `fdrs` in-place by subtracting
the mean and dividing by the standard deviation.  The mean is calculated as the
mean of the first column (i.e. along the frequency axis for the first drift
rate) of the first Matrix.  The standard deviation (aka sigma) value used is the
minimum standard deviation of all columns of all `fdrs`.  This is intended for
use when the FDR has been computed in pieces (e.g. to fit into GPU memory).
"""
function fdrnormalize!(fdrs::AbstractVector)
    fdr1 = fdrs[1]
    m = mean(@view fdr1[:,1])
    s = minimum(minimum.(std.(fdrs, dims=1)))
    for fdr in fdrs
        fdr .= (fdr .- m) ./ s
    end
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
