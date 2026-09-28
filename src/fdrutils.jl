using Statistics

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

# Select the thresholding statistics from per-column standard deviations:
# the sigma is the minimum positive column std (`Inf` when every column is
# degenerate, e.g. all-zero out-of-band drift blocks), and `i` is the first
# non-degenerate column whose mean the caller uses (nothing when
# degenerate).  Shared by the scalar, banded-buffer, and banded-reduce
# non-robust paths.
function _select_colstd(colstds::AbstractVector)
    i = findfirst(>(0), colstds)
    s = mapreduce(x -> x > 0 ? x : Inf, min, colstds; init = Inf)
    return i, s
end

# A few divisors of `Nf` nearest to `cpb`, for the divisibility error
# message.
function _nearest_divisors(Nf::Int, cpb::Int)
    out = Int[]
    for d in 0:max(cpb, Nf)
        for c in (d == 0 ? (cpb,) : (cpb - d, cpb + d))
            if 1 <= c <= Nf && Nf % c == 0 && c ∉ out
                push!(out, c)
                length(out) == 3 && return out
            end
        end
    end
    return out
end

"""
    noisestats(fdr; robust=true, chans_per_band=nothing, kwargs...)
        -> (mean = ..., std = ...)
    noisestats(fdrs; robust=true, kwargs...) -> (mean = ..., std = ...)

Compute the mean and standard deviation used for thresholding the
Frequency-Drift-Rate (FDR) Matrix `fdr` or matrices `fdrs` (the same
estimation applies to any frequency-by-something matrix of power-like data).
The mean is calculated as the mean of the first column that has a non-zero
standard deviation (normally the first column), i.e. along the frequency axis
for the first drift rate of `fdr` or the first matrix of `fdrs` with data.
The standard deviation (aka sigma) value used is the minimum non-zero
standard deviation of all columns of `fdr` or all columns of all matrices in
`fdrs`.  Columns with a standard deviation of zero (e.g. the all-zero columns
produced for out-of-band drift blocks by `taylorfdr`) are ignored; if all
columns have a standard deviation of zero, the standard deviation is `Inf`,
so that normalizing by it produces zeros and denormalizing with it produces
an `Inf` threshold (i.e. no hits).

If `robust` is true (the default), the mean and standard deviation are
instead estimated with [`noisefloor`](@ref), which is robust to signal
contamination (see its docstring).  The optional `k` keyword
(per-polarization Gamma shape) is passed through to `noisefloor` and improves
the accuracy of the standard deviation estimate when the integration factor
of the data is known.  The `qlo`, `clip`, and `refine` keywords of
`noisefloor` are also passed through.

When the integer `chans_per_band` is given, the statistics are instead
estimated per *band* of `chans_per_band` frequency channels (each band's
statistics are estimated from its `chans_per_band`-by-`Nr` block of values,
pooling the channels of the band) and returned as vectors of length
`size(fdr, 1)`, with each band's estimate repeated for every channel of the
band.  This enables per-channel thresholding (e.g. in [`findhits`](@ref))
while pooling enough samples per band for well-sampled estimates;
`chans_per_band = 1` estimates each channel independently (note that the
non-robust estimator's sigma is degenerate, `Inf`, for single-channel
bands).  `chans_per_band` must evenly divide the number of frequency
channels.  Banding is not supported for iterables of matrices.

Banding matters when the noise power varies along the frequency axis, e.g.
from power-level variations across a coarse channel's passband that survive
the analytic filter-response correction: one estimate for the whole channel
is then biased wherever the power differs, while per-band estimates track
the variation.  Pick the band width small enough that the variation within
a band is negligible, and large enough to pool many samples (e.g. 16K
channels, ≈ 48 kHz for a ~2.9 Hz channelization).  On CUDA, the robust
per-band estimates are computed in a fixed handful of batched device
passes, at a cost independent of the number of bands; on the host, each
band is estimated independently.

The returned values are a `NamedTuple` with fields `mean` and `std`, so both
`m, s = noisestats(fdr)` and `noisestats(fdr).std` work (scalars, or vectors
of length `size(fdr, 1)` when `chans_per_band` is given).
"""
function noisestats(fdr::AbstractMatrix; robust::Bool = true,
                    chans_per_band::Union{Nothing, Integer} = nothing,
                    kwargs...)
    if chans_per_band !== nothing
        cpb = Int(chans_per_band)
        cpb >= 1 ||
            throw(ArgumentError("chans_per_band must be at least 1 (got $cpb)"))
        Nf = size(fdr, 1)
        Nf % cpb == 0 || throw(ArgumentError(
            "chans_per_band (= $cpb) must evenly divide the number of " *
            "frequency channels ($Nf); nearest: " *
            join(_nearest_divisors(Nf, cpb), ", ")))
        return _banded_stats(fdr, cpb; robust, kwargs...)
    end
    if robust
        return _noisefloor_stats(fdr; kwargs...)
    end
    stds = vec(std(fdr, dims=1))
    i, s = _select_colstd(stds)
    m = i === nothing ? mean(@view fdr[:, 1]) : mean(@view fdr[:, i])
    (mean = m, std = s)
end

function noisestats(fdrs; robust::Bool = true, chans_per_band = nothing,
                    kwargs...)
    chans_per_band === nothing || throw(ArgumentError(
        "chans_per_band is only supported for a single matrix"))
    stats = [noisestats(fdr; robust, kwargs...) for fdr in fdrs]
    s = minimum(st.std for st in stats; init=Inf)
    i = findfirst(st -> st.std < Inf, stats)
    m = i === nothing ? first(stats).mean : stats[i].mean
    (mean = m, std = s)
end

# Per-band (mean, std) statistics for `chans_per_band`-sized bands of
# frequency channels (`cpb` is guaranteed to evenly divide the channel
# count), returned as vectors of length `size(fdr, 1)` with each band's
# estimate repeated over the channels of the band.
#
# Robust (quantile-based) statistics are not reductions; they are delegated
# to `_noisefloor_banded`, which the CUDA extension implements as a batched
# device computation whose cost is independent of the number of bands.  The
# host fallback below estimates each band independently.  Non-robust
# statistics are plain reductions computed directly on the matrix by
# `_banded_colstats!` — no bands are materialized (a single fused pass on
# CUDA).
function _banded_stats(fdr::AbstractMatrix, cpb::Int; robust::Bool, kwargs...)
    if !robust
        return _banded_stats_reduce(fdr, cpb)
    end
    return _noisefloor_banded(fdr, cpb; kwargs...)
end

# Host fallback for `_noisefloor_banded`: robust per-band statistics via an
# independent `noisefloor` estimate per band.  Each band is broadcast into a
# single reusable buffer, so repeated calls do not churn a full-matrix worth
# of allocations.  (The buffer also sidesteps `std` of a 2D SubArray, which
# falls back to scalar iteration and is disallowed for GPU arrays.)
function _noisefloor_banded(fdr::AbstractMatrix, cpb::Int; kwargs...)
    Nf, Nr = size(fdr)
    nbands = Nf ÷ cpb
    buf = similar(fdr, cpb, Nr)
    bstats = map(1:nbands) do bi
        r = ((bi - 1) * cpb + 1):(bi * cpb)
        buf .= @view fdr[r, :]
        _noisefloor_stats(buf; kwargs...)
    end
    means = [st.mean for st in bstats]
    stds = [st.std for st in bstats]
    T = promote_type(eltype(means), eltype(stds))
    mean_v = Vector{T}(undef, Nf)
    std_v = Vector{T}(undef, Nf)
    for (bi, st) in enumerate(bstats)
        r = ((bi - 1) * cpb + 1):(bi * cpb)
        mean_v[r] .= st.mean
        std_v[r] .= st.std
    end
    (mean = mean_v, std = std_v)
end

# Per-(band, column) means and (corrected) standard deviations of the
# `cpb`-sized bands of `fdr`'s rows, written into the preallocated
# `(Nf ÷ cpb, Nr)` outputs.  Host fallback: reshape reductions (see
# `_banded_stats` for why the bands are not viewed).
function _banded_colstats!(colmean::AbstractMatrix, colstd::AbstractMatrix,
                           fdr::AbstractMatrix, cpb::Int)
    nbands = size(colmean, 1)
    R = reshape(fdr, cpb, nbands, size(fdr, 2))
    colmean .= dropdims(mean(R, dims=1), dims=1)
    colstd .= dropdims(std(R, dims=1), dims=1)
    return colmean, colstd
end

# Non-robust banded statistics via `_banded_colstats!` (see
# `_banded_stats`); only the per-band selection (first non-degenerate
# column's mean, minimum positive column sigma) runs on the host.
function _banded_stats_reduce(fdr::AbstractMatrix, cpb::Int)
    Nf, Nr = size(fdr)
    nbands = Nf ÷ cpb
    colmean = similar(fdr, nbands, Nr)
    colstd = similar(fdr, nbands, Nr)
    _banded_colstats!(colmean, colstd, fdr, cpb)
    hmean = Array(colmean)
    hstd = Array(colstd)               # small (nbands, Nr) download
    T = promote_type(eltype(hmean), eltype(hstd))
    mean_v = Vector{T}(undef, Nf)
    std_v = Vector{T}(undef, Nf)
    for b in 1:nbands
        i, s_b = _select_colstd(@view hstd[b, :])
        m_b = i === nothing ? hmean[b, 1] : hmean[b, i]
        r = ((b - 1) * cpb + 1):(b * cpb)
        mean_v[r] .= m_b
        std_v[r] .= s_b
    end
    (mean = mean_v, std = std_v)
end

"""
    noisenormalize(scalar, m, s) -> normalized_scalar
    noisenormalize!(fdr[, m, s]) -> same fdr (normalized in place)
    noisenormalize!(fdrs[, m, s]) -> same fdrs (normalized in place)

Normalize the Frequency-Drift-Rate (FDR) Matrix `fdr` or matrices `fdrs`
in-place by subtracting the mean and dividing by the standard deviation.
If not given, the statistics are computed with [`noisestats`](@ref) (which
is robust to signal contamination by default).  The mean and standard
deviation may also be given explicitly as `m` and `s`, respectively — as
scalars, or as equal-length vectors for per-channel normalization (which
broadcasts along the frequency axis).
"""
function noisenormalize(fdr::Number, m, s)
    fdr = (fdr - m) / s
    return fdr
end

function noisenormalize!(fdr::AbstractMatrix, m, s)
    fdr .= (fdr .- m) ./ s
    return fdr
end

function noisenormalize!(fdr::AbstractMatrix)
    m, s = noisestats(fdr)
    return noisenormalize!(fdr, m, s)
end

function noisenormalize!(fdrs, m, s)
    for fdr in fdrs
        fdr .= (fdr .- m) ./ s
    end
    return fdrs
end

function noisenormalize!(fdrs)
    m, s = noisestats(fdrs)
    return noisenormalize!(fdrs, m, s)
end

"""
    noisedenormalize(snr, m, s) -> threshold
    noisedenormalize(snr, fdr) -> threshold
    noisedenormalize(snr, fdrs) -> threshold

Compute the denormalized value of `snr` using the mean `m` and standard
deviation `s`.  If frequency drift rate matrix `fdr` (or matrices `fdrs`) is
passed instead of `m` and `s` the statistics will be computed from the given
data with [`noisestats`](@ref) (which is robust to signal contamination by
default).  The denormalized value can be used as the threshold when detecting
proto-hits in `fdr` rather than normalizing `fdr` and using the `snr` value
directly.  `m` and `s` may be equal-length vectors, in which case a vector of
per-channel thresholds is returned.
"""
function noisedenormalize(snr, m, s)
    threshold = snr * s + m
    return threshold
end

function noisedenormalize(snr, fdr::AbstractMatrix)
    m, s = noisestats(fdr)
    return noisedenormalize(snr, m, s)
end

function noisedenormalize(snr, fdrs)
    m, s = noisestats(fdrs)
    return noisedenormalize(snr, m, s)
end

# Deprecated (soft) aliases for the noise* names above.  They forward with
# the old `robust = false` default of `fdrstats` (unless passed explicitly),
# so deprecated calls preserve pre-rename behavior.

function fdrstats(fdr::AbstractMatrix; robust::Bool = false, kwargs...)
    Base.depwarn("`fdrstats` is deprecated, use `noisestats` instead",
                 :fdrstats)
    noisestats(fdr; robust, kwargs...)
end

function fdrstats(fdrs; robust::Bool = false, kwargs...)
    Base.depwarn("`fdrstats` is deprecated, use `noisestats` instead",
                 :fdrstats)
    noisestats(fdrs; robust, kwargs...)
end

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
    m, s = noisestats(fdr; robust = false)  # preserve old default
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
    m, s = noisestats(fdrs; robust = false)  # preserve old default
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
    m, s = noisestats(fdr; robust = false)  # preserve old default
    return noisedenormalize(snr, m, s)
end

function fdrdenormalize(snr, fdrs)
    Base.depwarn("`fdrdenormalize` is deprecated, use `noisedenormalize` instead",
                 :fdrdenormalize)
    m, s = noisestats(fdrs; robust = false)  # preserve old default
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
