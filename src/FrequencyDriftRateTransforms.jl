module FrequencyDriftRateTransforms

# The exact-quantile and noise-statistics machinery lives in FastQuantiles.jl
# and NoiseEstimators.jl, re-exported here.
using FastQuantiles: fast_quantile
using NoiseEstimators: noisefloor, noisestats, noisenormalize!, noisenormalize,
                       noisedenormalize
using Statistics

# fast_quantile — provided by FastQuantiles.jl and re-exported
export fast_quantile

# noisefloor — provided by NoiseEstimators.jl and re-exported
export noisefloor

# fdrutils.jl — noise statistics provided by NoiseEstimators.jl, re-exported
export create_fdr
export noisestats, noisenormalize!, noisenormalize
export noisedenormalize
export findprotohits
export fdrstats  # FDRT's FDR-specific statistics (see its docstring)
# deprecated (soft) aliases for the noise* names
export fdrnormalize!, fdrnormalize, fdrdenormalize

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
export fftw_set_num_threads, fftw_get_num_threads
export zdtinput!, zdtoutput!
export zdtpreprocess!, zdtconvolve!, zdtpostprocess!
export zdtfdr, zdtfdr!
export input!, output!  # deprecated
export preprocess!, convolve!, postprocess!  # deprecated

# zdtutils.jl, batchrates.jl
export calcNl, growNr, estimate_memory, driftrates, batchrates

include("batchrates.jl")
include("fdrutils.jl")
include("findhits.jl")
include("intfdr.jl")
include("fftfdr.jl")
include("taylorfdr.jl")
include("zdtfdr.jl")
include("zdtutils.jl")

end # module FrequencyDriftRateTransforms
