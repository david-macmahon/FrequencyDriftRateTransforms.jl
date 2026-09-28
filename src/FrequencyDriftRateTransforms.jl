module FrequencyDriftRateTransforms

# fastquantile.jl
export fast_quantile

# noisefloor.jl
export noisefloor

# fdrutils.jl
export create_fdr, noisestats, noisenormalize!, noisenormalize
export noisedenormalize
export findprotohits
# deprecated (soft) aliases for the noise* names
export fdrstats, fdrnormalize!, fdrnormalize, fdrdenormalize

# findhits.jl
export findhits

# intfdr.jl
export intshift, intshift!
export intfdr, intfdr!

# fftfdr.jl
export FFTWorkspace, FFTWorkspace!
export fftfdr_workspace, fftfdr_workspace!  # deprecated
export fdshiftsum, fdshiftsum!
export fdshift, fdshift!
export fftfdr, fftfdr!

# taylorfdr.jl
export TaylorWorkspace
export taylorrates, taylortree!, taylortree
export taylorfdr, taylorfdr!

# zdtfdr.jl
export ZDTWorkspace
export zdtinput!, zdtoutput!
export zdtpreprocess!, zdtconvolve!, zdtpostprocess!
export zdtfdr, zdtfdr!
export input!, output!  # deprecated
export preprocess!, convolve!, postprocess!  # deprecated

# zdtutils.jl, batchrates.jl
export calcNl, growNr, estimate_memory, driftrates, batchrates

include("batchrates.jl")
include("fastquantile.jl")
include("noisefloor.jl")
include("fdrutils.jl")
include("findhits.jl")
include("intfdr.jl")
include("fftfdr.jl")
include("taylorfdr.jl")
include("zdtfdr.jl")
include("zdtutils.jl")

end # module FrequencyDriftRateTransforms
