module DopplerDriftSearch

export create_fdr

# intfdr.jl
export intshift, intshift!
export intfdr, intfdr!

# fftfdr.jl
export fftfdr_workspace, fftfdr_workspace!
export fdshiftsum, fdshiftsum!
export fdshift, fdshift!
export fftfdr, fftfdr!

# zdtfdr.jl
export calcNl, growNr
export ZDTWorkspace
export initialize!
export zdtfdr, zdtfdr!

# Type piracy to workaround CUDA.jl issue #1559.  For more details, see:
# https://github.com/JuliaGPU/CUDA.jl/issues/1559
import AbstractFFTs: plan_brfft
import CUDA: CuArray
plan_brfft(A::CuArray, d::Integer, region; kwargs...) = plan_brfft(A, d, region)

# Import CUDA functions for optimizing workarea usage
import CUDA: unsafe_free!, CUFFT.cufftSetWorkArea

"""
Create an uninitialized `Matrix` suitable for use with `intfdr!` or `fftfdr!`
and the given `spectrogram` and `Nr` (number of rates).  The returned `Matrix`
will be similar to `spectrogram` in type and size of first dimension, but its
second dimension will be `Nr`.
"""
function create_fdr(spectrogram, Nr::Integer)
    Nf = size(spectrogram, 1)
    similar(spectrogram, Nf, Nr)
end

"""
Create an uninitialized `Matrix` suitable for use with `intfdr!` or `fftfdr!`
and the given `spectrogram` and `rates`.  The returned `Matrix` will be similar
to `spectrogram` in type and size of first dimension, but its second dimension
will be sized by the length of `rates`.
"""
function create_fdr(spectrogram, rates)
    create_fdr(spectrogram, length(rates))
end

include("intfdr.jl")
include("fftfdr.jl")
include("zdtfdr.jl")

end # module DopplerDriftSearch
