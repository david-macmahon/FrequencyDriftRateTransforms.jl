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
