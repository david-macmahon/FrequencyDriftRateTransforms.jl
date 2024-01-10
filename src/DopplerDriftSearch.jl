module DopplerDriftSearch

# fdrutils.jl
export create_fdr, fdrnormalize!

# intfdr.jl
export intshift, intshift!
export intfdr, intfdr!

# fftfdr.jl
export fftfdr_workspace, fftfdr_workspace!
export fdshiftsum, fdshiftsum!
export fdshift, fdshift!
export fftfdr, fftfdr!

# zdtfdr.jl
export ZDTWorkspace
export input!, output!
export preprocess!, convolve!, postprocess!
export zdtfdr, zdtfdr!

# zdtutils.jl
export calcNl, growNr, estimate_memory, driftrates

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

include("extstubs.jl")
include("fdrutils.jl")
include("intfdr.jl")
include("fftfdr.jl")
include("zdtfdr.jl")
include("zdtutils.jl")

end # module DopplerDriftSearch
