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
    fdrnormalize(scalar, m, s) -> normalized_scalar
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
function fdrnormalize(fdr::Number, m, s)
    fdr = (fdr - m) / s
    return fdr
end

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
    fdrdenormalize(snr, m, s) -> threshold
    fdrdenormalize(snr, fdr) -> threshold
    fdrdenormalize(snr, fdrs) -> threshold

Compute the denormalized value of `snr` using the mean `m` and standard
deviation `s`.  If frequency drift rate matrix `fdr` (or matrices `fdrs`) is
passed instead of `m` and `s` the statistics will be computed from the given
data.  The denormalized value can be used as the threshold when detecting
proto-hits in `fdr` rather than normalizing `fdr` and using the `snr` value
directly.
"""
function fdrdenormalize(snr, m, s)
    threshold = snr * s + m
    return threshold
end

function fdrdenormalize(snr, fdr)
    m, s = fdrstats(fdr)
    threshold = fdrdenormalize(snr, m, s)
    return threshold
end

function fdrdenormalize(snr, fdrs::AbstractVector)
    m, s = fdrstats(fdrs)
    threshold = fdrdenormalize(snr, m, s)
    return threshold
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
