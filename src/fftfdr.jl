using AbstractFFTs
using LinearAlgebra

# We need to depend on FFTW for access to its flags constants because
# AbstractFFTs defines the planning flags to be "a bitwise-or of FFTW planner
# flags".  For more details, see:
# https://github.com/JuliaMath/AbstractFFTs.jl/issues/71
import FFTW

# Type piracy to workaround CUDA.jl issue #1559.  For more details, see:
# https://github.com/JuliaGPU/CUDA.jl/issues/1559
import CUDA: CuArray
AbstractFFTs.plan_brfft(A::CuArray, d::Integer, region; kwargs...) = plan_brfft(A, d, region)

function phasor(k::Integer, n::Integer, r::Float32, N::Integer)
    cispi(-2*k*n*r/N)
end

function phasor(ij::CartesianIndex, r, N)
    phasor(ij[1]-1, ij[2]-1, r, N)
end

"""
    fftfdr_workspace(specrogram[; bunaligned=true]) -> workspace

Create a *workspace* suitable for use with `fdshift!` and `fftfdr!`.  The
workspace includes all the required intermediate storage buffers and FFT plan
objects needed for creating a frequency drift rate matrix for `spectrogram`
using `fftfdr`.  One of these intermediate buffers is an `AbstractMatrix` that
holds the FFT of `spectrogram` along the frequency dimension.  This FFT is
performed as part of creating the workspace.  The first (fastest changing)
dimension of `spectrogram` is frequency and the second dimension (slowest
changing) is time. 

By default, the spectra in columns of `spectrogram` have no alignment
constraints, but if the columns of `fftfdr`'s output buffer will be suitably
aligned for the FFT implementation, then `bunalingned` may be passed as `false`.
This will usually be the case if the number of frequency channels has several
factors of 2, but it depends on the specifics of the FFT implementation.

The workspace also includes an FFT plan suitable for generating a *de-dopplered*
spectrogram for a given rate.  This functionality is provided by `fdshift!`.
"""
function fftfdr_workspace(spectrogram::AbstractMatrix{<:Real}; bunaligned=true)
    Nf, Nt = size(spectrogram)
    dest_rfft = similar(spectrogram, Complex{eltype(spectrogram)}, Nf÷2+1, Nt)
    dest_phasor = similar(dest_rfft)
    dest_sum = similar(dest_rfft, Nf÷2+1)
    fplan = plan_rfft(spectrogram, 1)
    bplan1d = plan_brfft(dest_sum, Nf; flags=FFTW.ESTIMATE|(bunaligned ? FFTW.UNALIGNED : 0))
    # The `2d` in `bplan2d` refers to the input/output arrays.
    # The dimensionality of the FFT is still 1D along the first dimension.
    bplan2d = plan_brfft(dest_rfft, Nf, 1) # Assume unaligned output for now
    mul!(dest_rfft, fplan, spectrogram)
    (
        Nf=Nf,
        dest_rfft=dest_rfft, 
        dest_phasor=dest_phasor, 
        dest_sum=dest_sum, 
        fplan=fplan, 
        bplan1d=bplan1d,
        bplan2d=bplan2d
    )
end

"""
    fftfdr_workspace!(workspace, specrogram) -> workspace

Reinitialize `workspace` buffers using `spectrogram`.  An exception is thrown if
`spectrogram` is type and/or size incompatible with `workspace`.  If `workspace`
is `nothing`, then a new workspace is created.  The `bunaligned` keyword
argument is ignored unless `workspace` is `nothing`.
"""
function fftfdr_workspace!(workspace, spectrogram::AbstractMatrix{<:Real}; bunaligned=true)
    Nf, Nt = size(spectrogram)
    if size(workspace.dest_rfft) !== (Nf÷2+1, Nt)
        error("input spectrogram has unexpected size")
    end
    if eltype(workspace.dest_rfft) !== Complex{eltype(spectrogram)}
        error("input spectrogram has unexpected element type")
    end
    # Size and eltype matches, FFT spectrogram into workspace
    mul!(workspace.dest_rfft, workspace.fplan, spectrogram)
    return workspace
end

function fftfdr_workspace!(::Nothing, spectrogram::AbstractMatrix{<:Real}; bunaligned=true)
    fftfdr_workspace(spectrogram; bunaligned=bunaligned)
end

function fdshiftsum!(dest::AbstractVector, workspace, rate)
    # Multiply the Fourier domain spectra by the doppler rate phasors.
    workspace.dest_phasor .= workspace.dest_rfft .*
        phasor.(CartesianIndices(workspace.dest_rfft), Float32(rate), workspace.Nf)

    # Sum over time in the post-phasor Fourier domain
    sum!(workspace.dest_sum, workspace.dest_phasor)

    # Store the backwards FFT of `workspace.dest_sum` into `dest`.
    mul!(dest, workspace.bplan1d, workspace.dest_sum)
end

function fdshiftsum(workspace, rate)
    Nf = workspace.Nf
    dest = similar(workspace.dest_sum, real(eltype(workspace.dest_sum)), Nf)
    fdshiftsum!(dest, workspace, rate)
end

function fdshift!(dest::AbstractMatrix, workspace, rate)
    # Multiply the Fourier domain spectra by the doppler rate phasors.
    workspace.dest_phasor .= workspace.dest_rfft .*
        phasor.(CartesianIndices(workspace.dest_rfft), Float32(rate), workspace.Nf)

    # Store the backwards FFT of `workspace.dest_phasor` into `dest`.
    mul!(dest, workspace.bplan2d, workspace.dest_phasor)
end

function fdshift(workspace, rate)
    Nf = workspace.Nf
    Nt = size(workspace.dest_rfft, 2)
    dest = similar(workspace.dest_rfft, real(eltype(workspace.dest_rfft)), Nf, Nt)
    fdshift!(dest, workspace, rate)
end

"""
Same as the `fftfdr` function, but store the results in `fdr`, which is also
returned.  The size of `fdr` must be `(workspace.Nf, length(rates))`.
"""
function fftfdr!(fdr, workspace, rates)
    Nf = workspace.Nf
    Nr = length(rates)
    @assert size(fdr) == (Nf, Nr)
    for (col, rate) in zip(eachcol(fdr), rates)
        fdshiftsum!(col, workspace, rate)
    end
    return fdr
end

"""
Compute the frequency drift rate matrix for the given `workspace` and `rates`
values using FFT shifting of each frequency spectrum. The size of the returned
frequency drift rate matrix will be `(workspace.Nf, length(rates))`.
"""
function fftfdr(workspace, rates)
    Nf = workspace.Nf
    Nr = length(rates)
    fdr = similar(workspace.dest_sum, real(eltype(workspace.dest_sum)), Nf, Nr)
    fftfdr!(fdr, workspace, rates)
end
