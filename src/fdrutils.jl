"""
    create_fdr(spectrogram, Nr::Integer) -> Matrix

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
    create_fdr(spectrogram, rates) -> Matrix

Create an uninitialized `Matrix` suitable for use with `intfdr!`, `fftfdr!`, or
`zdtfdr!` and the given `spectrogram` and `rates`.  The returned `Matrix` will
be similar to `spectrogram` in type and size of first dimension, but its second
dimension will be sized by the length of `rates`.
"""
function create_fdr(spectrogram, rates)
    create_fdr(spectrogram, length(rates))
end

"""
    fdrstats(fdr; robust=false, kwargs...) -> (mean = ..., std = ...)
    fdrstats(fdrs; robust=false, kwargs...) -> (mean = ..., std = ...)

Compute the mean and standard deviation of the Frequency-Drift-Rate (FDR)
Matrix `fdr` or matrices `fdrs` using estimators that rely on the FDR's
wrap-around redundancy: the sum of every drift-rate column is essentially
the sum of the entire input spectrogram, so any drift-rate column gives
the same mean, and the drift-rate column with the minimum σ is the one
least contaminated by RFI, making its σ the least RFI-skewed plain σ
estimate.  For `robust = false` (the default) the mean is calculated as
the mean of the first column that has a non-zero standard deviation
(normally the first column), i.e. along the frequency axis for the first
drift rate of `fdr` or the first matrix of `fdrs` with data, and the
standard deviation (aka sigma) value used is the minimum non-zero standard
deviation of all columns of `fdr` or all columns of all matrices in
`fdrs`.  Columns with a standard deviation of zero (e.g. the all-zero
columns produced for out-of-band drift blocks by `taylorfdr`) are ignored;
if all columns have a standard deviation of zero, the standard deviation
is `Inf`, so that normalizing by it produces zeros and denormalizing with
it produces an `Inf` threshold (i.e. no hits).

For `robust = true` this is identical to `noisestats` with
`robust = true`: its quantile anchoring is RFI-robust without column
assumptions and is the recommended mode.  These column-based estimators
are only valid for FDR-like data; generic data should use
`noisestats`, whose plain mode uses ensemble statistics.

The returned values are a `NamedTuple` with fields `mean` and `std`, so
both `m, s = fdrstats(fdr)` and `fdrstats(fdr).std` work.
"""
function fdrstats(fdr::AbstractMatrix; robust::Bool=false, kwargs...)
    if robust
        return noisestats(fdr; robust, kwargs...)
    end
    stds = vec(std(fdr, dims=1))
    s = minimum(filter(>(0), stds); init=Inf)
    i = findfirst(>(0), stds)
    m = i === nothing ? mean(@view fdr[:, 1]) : mean(@view fdr[:, i])
    (mean = m, std = s)
end

function fdrstats(fdrs; robust::Bool=false, kwargs...)
    stats = [fdrstats(fdr; robust, kwargs...) for fdr in fdrs]
    s = minimum(st.std for st in stats; init=Inf)
    i = findfirst(st -> st.std < Inf, stats)
    m = i === nothing ? first(stats).mean : stats[i].mean
    (mean = m, std = s)
end

# Deprecated (soft) aliases for the noise* names provided by
# NoiseEstimators.jl.  They forward with the old `robust = false` default
# of the pre-rename `fdrstats` (unless passed explicitly), so deprecated
# calls preserve pre-rename behavior; their no-statistics forms route
# through `fdrstats` for the same reason.

function fdrnormalize(fdr::Number, m, s)
    Base.depwarn("`fdrnormalize` is deprecated, use `noisenormalize` instead",
                 :fdrnormalize)
    noisenormalize(fdr, m, s)
end

function fdrnormalize!(fdr::AbstractMatrix, m, s)
    Base.depwarn("`fdrnormalize!` is deprecated, use `noisenormalize!` instead",
                 :fdrnormalize!)
    noisenormalize!(fdr, m, s)
end

function fdrnormalize!(fdr::AbstractMatrix)
    Base.depwarn("`fdrnormalize!` is deprecated, use `noisenormalize!` instead",
                 :fdrnormalize!)
    m, s = fdrstats(fdr)  # preserve old default (FDR plain statistics)
    return noisenormalize!(fdr, m, s)
end

function fdrnormalize!(fdrs, m, s)
    Base.depwarn("`fdrnormalize!` is deprecated, use `noisenormalize!` instead",
                 :fdrnormalize!)
    noisenormalize!(fdrs, m, s)
end

function fdrnormalize!(fdrs)
    Base.depwarn("`fdrnormalize!` is deprecated, use `noisenormalize!` instead",
                 :fdrnormalize!)
    m, s = fdrstats(fdrs)  # preserve old default (FDR plain statistics)
    return noisenormalize!(fdrs, m, s)
end

function fdrdenormalize(snr, m, s)
    Base.depwarn("`fdrdenormalize` is deprecated, use `noisedenormalize` instead",
                 :fdrdenormalize)
    noisedenormalize(snr, m, s)
end

function fdrdenormalize(snr, fdr::AbstractMatrix)
    Base.depwarn("`fdrdenormalize` is deprecated, use `noisedenormalize` instead",
                 :fdrdenormalize)
    m, s = fdrstats(fdr)  # preserve old default (FDR plain statistics)
    return noisedenormalize(snr, m, s)
end

function fdrdenormalize(snr, fdrs)
    Base.depwarn("`fdrdenormalize` is deprecated, use `noisedenormalize` instead",
                 :fdrdenormalize)
    m, s = fdrstats(fdrs)  # preserve old default (FDR plain statistics)
    return noisedenormalize(snr, m, s)
end

"""
    fdrsynchronize(::Type{<:AbstractArray})

The default implementation of this function does nothing, but it should be
called whenever synchronization might be needed.  Methods can be defined for
types that are more specific than `AbstractArray` when they have
synchronization requirements and mechanisms (e.g. `CuArray`).
"""
function fdrsynchronize(::Type{<:AbstractArray})
end

"""
    findprotohits(fdr::AbstractMatrix, threshold; snr=false) -> hijs
    findprotohits(fdrs, threshold; snr=false) -> hijs

Given a Frequency-Drift-Rate matrix `fdr`, or an iterable of FDR matrices,
find all points that are greater than or equal to `threshold`.  For a single FDR
matrix, this is essentially a simple wrapper around `findall`.  When an iterable
of FDR matrices is passed, they are treated as drift rate adjacent portions of a
larger FDR Matrix (as if they had been `hcat`'d together).  If `snr` is true,
the threshold is denormalized based on `fdr` or `fdrs` before the comparison.

The resultant *proto-hits* are returned as `Vector{CartesianIndex}`, often
referred to as `hijs` for "hit (i,j) coordinates".  The `hijs` at this stage are
called proto-hits (as opposed to "real" hits) since no clustering of neighboring
proto-hits has been performed.
"""
function findprotohits(fdr::AbstractMatrix, threshold; snr::Bool=false)
    if snr
        threshold = noisedenormalize(threshold, fdr)
    end

    findall(>=(threshold), fdr)
end

function findprotohits(fdrs, threshold; snr::Bool=false)
    if snr
        threshold = noisedenormalize(threshold, fdrs)
    end

    Nrb = size(first(fdrs), 2)
    offset = CartesianIndex(0, Nrb)
    hijs = CartesianIndex{2}[]
    for (i, fdr) in enumerate(fdrs)
        append!(hijs, findprotohits(fdr, threshold) .+ ((i-1)*offset))
    end

    return hijs
end
