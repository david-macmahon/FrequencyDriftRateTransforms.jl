using AbstractFFTs
using LinearAlgebra

# We need to depend on FFTW for access to its flags constants because
# AbstractFFTs defines the planning flags to be "a bitwise-or of FFTW planner
# flags".  For more details, see:
# https://github.com/JuliaMath/AbstractFFTs.jl/issues/71
import FFTW

function phasor(k::Integer, n::Integer, r::Float32, N::Integer)
    cispi(2*k*n*r/N)
end

function phasor(ij::CartesianIndex, r, N)
    phasor(ij[1]-1, ij[2]-1, r, N)
end

"""
    FFTWorkspace(spectrogram[; bunaligned=true]) -> workspace

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
aligned for the FFT implementation, then `bunaligned` may be passed as `false`.
This will usually be the case if the number of frequency channels has several
factors of 2, but it depends on the specifics of the FFT implementation.

The workspace also includes an FFT plan suitable for generating a *de-dopplered*
spectrogram for a given rate.  This functionality is provided by `fdshift!`.
"""
function FFTWorkspace(spectrogram::AbstractMatrix{<:Real}; bunaligned=true)
    Nf, Nt = size(spectrogram)
    dest_rfft = similar(spectrogram, complex(eltype(spectrogram)), Nf÷2+1, Nt)
    dest_phasor = similar(dest_rfft)
    dest_sum = similar(dest_rfft, Nf÷2+1)
    fplan = plan_rfft(spectrogram, 1)
    bplan1d = plan_irfft(dest_sum, Nf; flags=FFTW.ESTIMATE|(bunaligned ? FFTW.UNALIGNED : 0))
    # The `2d` in `bplan2d` refers to the input/output arrays.
    # The dimensionality of the FFT is still 1D along the first dimension.
    bplan2d = plan_irfft(dest_rfft, Nf, 1) # Assume unaligned output for now
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
    FFTWorkspace!(workspace, spectrogram) -> workspace

Reinitialize `workspace` buffers using `spectrogram`.  An `ArgumentError` is
thrown if `spectrogram` is type and/or size incompatible with `workspace`.
If `workspace` is `nothing`, then a new workspace is created.  The
`bunaligned` keyword argument is ignored unless `workspace` is `nothing`.
"""
function FFTWorkspace!(workspace, spectrogram::AbstractMatrix{<:Real}; bunaligned=true)
    Nf, Nt = size(spectrogram)
    if size(workspace.dest_rfft) != (Nf÷2+1, Nt)
        throw(ArgumentError("input spectrogram has unexpected size (got ($Nf, $Nt))"))
    end
    if eltype(workspace.dest_rfft) !== complex(eltype(spectrogram))
        throw(ArgumentError("input spectrogram has unexpected element type ($(eltype(spectrogram)))"))
    end
    # Size and eltype matches, FFT spectrogram into workspace
    mul!(workspace.dest_rfft, workspace.fplan, spectrogram)
    return workspace
end

function FFTWorkspace!(::Nothing, spectrogram::AbstractMatrix{<:Real}; bunaligned=true)
    FFTWorkspace(spectrogram; bunaligned=bunaligned)
end

"""
    fftfdr_workspace(spectrogram[; bunaligned=true])

Deprecated alias of [`FFTWorkspace`](@ref).
"""
function fftfdr_workspace(args...; kwargs...)
    Base.depwarn("`fftfdr_workspace` is deprecated, use `FFTWorkspace`", :fftfdr_workspace)
    FFTWorkspace(args...; kwargs...)
end

"""
    fftfdr_workspace!(workspace, spectrogram)

Deprecated alias of [`FFTWorkspace!`](@ref).
"""
function fftfdr_workspace!(args...; kwargs...)
    Base.depwarn("`fftfdr_workspace!` is deprecated, use `FFTWorkspace!`",
                 :fftfdr_workspace!)
    FFTWorkspace!(args...; kwargs...)
end

"""
    fdshiftsum!(dest, workspace, rate) -> dest

Same as the `fdshiftsum` function, but store the result in `dest`, which is
also returned.  `dest` must be a `Vector` of length `workspace.Nf`.
"""
function fdshiftsum!(dest::AbstractVector, workspace, rate)
    # Multiply the Fourier domain spectra by the doppler rate phasors.
    workspace.dest_phasor .= workspace.dest_rfft .*
        phasor.(CartesianIndices(workspace.dest_rfft), Float32(rate), workspace.Nf)

    # Sum over time in the post-phasor Fourier domain
    sum!(workspace.dest_sum, workspace.dest_phasor)

    # Store the backwards FFT of `workspace.dest_sum` into `dest`.
    mul!(dest, workspace.bplan1d, workspace.dest_sum)
end

"""
    fdshiftsum(workspace, rate) -> dest

Compute one column of the frequency drift rate matrix for the given
`workspace` and `rate` values (see [`fftfdr`](@ref)), i.e. the sum over time
of the Fourier domain spectra in `workspace` after de-doppler shifting them
for `rate`.  The returned `dest` will be a `Vector` of length
`workspace.Nf`.
"""
function fdshiftsum(workspace, rate)
    Nf = workspace.Nf
    dest = similar(workspace.dest_sum, real(eltype(workspace.dest_sum)), Nf)
    fdshiftsum!(dest, workspace, rate)
end

"""
    fdshift!(dest, workspace, rate) -> dest

Same as the `fdshift` function, but store the result in `dest`, which is also
returned.  `dest` must have size `(workspace.Nf, Nt)` where `Nt` is the
number of time samples in the spectrogram used to create `workspace`.
"""
function fdshift!(dest::AbstractMatrix, workspace, rate)
    # Multiply the Fourier domain spectra by the doppler rate phasors.
    workspace.dest_phasor .= workspace.dest_rfft .*
        phasor.(CartesianIndices(workspace.dest_rfft), Float32(rate), workspace.Nf)

    # Store the backwards FFT of `workspace.dest_phasor` into `dest`.
    mul!(dest, workspace.bplan2d, workspace.dest_phasor)
end

"""
    fdshift(workspace, rate) -> dest

Compute the *de-dopplered* spectrogram for the given `workspace` and `rate`
values by de-doppler shifting the Fourier domain spectra in `workspace` for
`rate` and transforming each time sample back to frequency.  The returned
`dest` will be a `Matrix` with the same size as the spectrogram used to
create `workspace`.
"""
function fdshift(workspace, rate)
    Nf = workspace.Nf
    Nt = size(workspace.dest_rfft, 2)
    dest = similar(workspace.dest_rfft, real(eltype(workspace.dest_rfft)), Nf, Nt)
    fdshift!(dest, workspace, rate)
end

"""
    fftfdr!(fdr, workspace, rates) -> fdr

Same as the `fftfdr` function, but store the results in `fdr`, which is also
returned.  The size of `fdr` must be `(workspace.Nf, length(rates))`.
"""
function fftfdr!(fdr, workspace, rates)
    Nf = workspace.Nf
    Nr = length(rates)
    size(fdr) == (Nf, Nr) ||
        throw(ArgumentError("fdr must have size ($Nf, $Nr) (got $(size(fdr)))"))
    for (col, rate) in zip(eachcol(fdr), rates)
        fdshiftsum!(col, workspace, rate)
    end
    return fdr
end

"""
    fftfdr(workspace, rates) -> fdr

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

"""
    fftfdr(spectrogram, rates) -> fdr

One-shot form of `fftfdr`: construct an [`FFTWorkspace`](@ref) for
`spectrogram` and return `fftfdr(workspace, rates)`.  Constructing a workspace
for every call is relatively expensive; reuse a workspace when computing
multiple transforms of the same size.
"""
function fftfdr(spectrogram::AbstractMatrix{<:Real}, rates)
    fftfdr(FFTWorkspace(spectrogram), rates)
end

"""
    fftfdr(workspace, spectrogram, rates) -> fdr

One-shot form that first reinputs `spectrogram` into `workspace` (as
[`FFTWorkspace!`](@ref) does) and then returns `fftfdr(workspace, rates)`.
This avoids the stale-data hazard of reusing a workspace without reinput.
"""
function fftfdr(workspace, spectrogram::AbstractMatrix{<:Real}, rates)
    FFTWorkspace!(workspace, spectrogram)
    fftfdr(workspace, rates)
end

"""
    fftfdr!(fdr, workspace, spectrogram, rates) -> fdr

Same as `fftfdr!(fdr, workspace, rates)`, but first reinputs `spectrogram`
into `workspace` (as [`FFTWorkspace!`](@ref) does).
"""
function fftfdr!(fdr, workspace, spectrogram::AbstractMatrix{<:Real}, rates)
    FFTWorkspace!(workspace, spectrogram)
    fftfdr!(fdr, workspace, rates)
end
