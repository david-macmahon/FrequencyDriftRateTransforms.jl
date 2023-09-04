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
export input!, output!
export preprocess!, convolve!, postprocess!
export zdtfdr, zdtfdr!

# For Julia < 1.9.0
if !isdefined(Base, :get_extension)
    # Add DataFrame constructor for Vector{Filterbank.Header} if/when DataFrames
    # is imported.
    using Requires
end
@static if !isdefined(Base, :get_extension)
    function __init__()
        @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" begin
            include("../ext/CUDADopplerDriftSearchExt.jl")
        end
    end
end

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
