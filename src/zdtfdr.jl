using AbstractFFTs
using LinearAlgebra

# We need to depend on FFTW for access to its flags constants because
# AbstractFFTs defines the planning flags to be "a bitwise-or of FFTW planner
# flags".  For more details, see:
# https://github.com/JuliaMath/AbstractFFTs.jl/issues/71
import FFTW

"""
Calculate the smallest FFT size greater than `Nt + Nr - 1` and having only
factors from the given `factors`.
"""
function calcNl(Nt, Nr, factors::Union{Tuple,AbstractVector}=(2,3,5))
    nextprod(factors, Nt + Nr - 1)
end

"""
Compute the largest `Nr` that fits in the smallest FFT size greater than `Nt +
Nr - 1` and having only factors from the given `factors`.
"""
function growNr(Nt::Integer, Nr::Integer, factors::Union{Tuple,AbstractVector}=(2,3,5))
    return calcNl(Nt, Nr, factors) - Nt + 1
end

mutable struct ZDTWorkspace
    Nf::Int
    Nt::Int
    r0::Float32
    δr::Float32
    Nr::Int
    Nl::Int
    factors::Union{Tuple,AbstractVector}

    F::AbstractMatrix{<:Complex}
    Yf::AbstractMatrix{<:Complex}
    Y::AbstractMatrix{<:Complex}
    Ys::AbstractMatrix{<:Complex}
    V::AbstractMatrix{<:Complex}

    # Input/output FFT plans
    rfft_plan::AbstractFFTs.Plan
    brfft_plan::AbstractFFTs.Plan

    # CZT FFT plans (in-place)
    fft_plan::AbstractFFTs.Plan
    bfft_plan::AbstractFFTs.Plan
    
    function ZDTWorkspace(spectrogram::AbstractMatrix{<:Real},
                          r0::Real, δr::Real, Nr::Integer,
                          factors::Union{Tuple,AbstractVector}=(2,3,5);
                          output_aligned=false)
        Nf, Nt = size(spectrogram)
        Nl = calcNl(Nt, Nr, factors)

        F = similar(spectrogram, complex(eltype(spectrogram)), Nf÷2+1, Nt)
        Y = similar(spectrogram, complex(eltype(spectrogram)), Nf÷2+1, Nl)
        V = similar(Y)

        Yf = @view Y[:, 1:Nt]
        Ys = @view Y[:, 1:Nr]

        ws = new(
            Nf, Nt, r0, δr, Nr, Nl, factors,
            F, Yf, Y, Ys, V
        )

        # Call function to plan FFTs.  This allows for specialization based on
        # the Array type of spectrogram.
        plan_ffts!(ws, spectrogram; output_aligned=output_aligned)

        # Initialize F with FFT of spectrogram
        initialize!(ws, spectrogram)
        # Precompute V
        computeV!(ws)

        return ws
    end
end # mutable struct ZDTWorkspace

function ZDTWorkspace(spectrogram::AbstractMatrix{<:Real},
                      rates::AbstractRange,
                      factors::Union{Tuple,AbstractVector}=(2,3,5);
                      output_aligned=false)
    r0 = first(rates)
    δr = step(rates)
    Nr = length(rates)
    ZDTWorkspace(spectrogram, r0, δr, Nr, factors;
                 output_aligned=output_aligned)
end

function Base.show(io::IO, ws::ZDTWorkspace)
    print(io, typeof(ws))
    print(io, "(Nf=", ws.Nf)
    print(io, ",Nt=", ws.Nt)
    print(io, ",r0=", ws.r0)
    print(io, ",δr=", ws.δr)
    print(io, ",Nr=", ws.Nr)
    print(io, ",Nl=", ws.Nl)
    print(io, ",", ws.factors)
    print(io, ")")
end

"""
Make the ZDT's FFT plans for `spectrogram::AbstractArray` for which a more
specialied method is not available.
"""
function plan_ffts!(workspace::ZDTWorkspace,
                    spectrogram::AbstractMatrix{<:Real};
                    output_aligned::Bool=false)
    Nf = workspace.Nf
    Y = workspace.Y
    Ys = workspace.Ys
    brfft_flags = FFTW.ESTIMATE | (output_aligned ? 0 : FFTW.UNALIGNED)

    workspace.rfft_plan = plan_rfft(spectrogram, 1)
    workspace.brfft_plan = plan_brfft(Ys, Nf, 1; flags=brfft_flags)

    workspace.fft_plan = plan_fft!(Y, 2)
    workspace.bfft_plan = plan_bfft!(Y, 2)

    return nothing
end

"""
Make the ZDT's FFT plans for `spectrogram.CuArray`.  CUFFT requires workareas
for FFT plans, which can be as large as the inputs.  The complex-to-complex FFTs
(both forwards and backwards) used in the ZDT require a larger workspace than
the somewhat smaller real-to-complex forwards FFT and the complex-to-real
backards FFT.  By creating the forward complex-to-complex FFT plan first, we can
then replace the workareas of the other FFT plans, as they are created, with the
workarea of the first created plan.  This allows all FFT plans to share one
common workarea.
    
Once this package no longer depends on CUDA.jl we might consider switching to
using `Requires.jl` so that this method only gets defined if the user has chosen
to use CUDA (rather than this package depending directly on CUDA).
"""
function plan_ffts!(workspace::ZDTWorkspace,
                    spectrogram::CuArray{<:Real};
                    output_aligned::Bool=false)
    # Inner function to replace workarea of `dest_plan` with that of `src_plan`.
    function replace_workarea(dest_plan, src_plan)
        new_workarea = @view src_plan.workarea[axes(src_plan.workarea)...]
        cufftSetWorkArea(dest_plan, new_workarea)
        unsafe_free!(dest_plan.workarea)
        dest_plan.workarea = new_workarea
    end

    Nf = workspace.Nf
    Y = workspace.Y
    Ys = workspace.Ys
    brfft_flags = FFTW.ESTIMATE | (output_aligned ? 0 : FFTW.UNALIGNED)

    workspace.fft_plan = plan_fft!(Y, 2)

    workspace.bfft_plan = plan_bfft!(Y, 2)
    replace_workarea(workspace.bfft_plan, workspace.fft_plan)

    workspace.rfft_plan = plan_rfft(spectrogram, 1)
    replace_workarea(workspace.rfft_plan, workspace.fft_plan)

    workspace.brfft_plan = plan_brfft(Ys, Nf, 1; flags=brfft_flags)
    replace_workarea(workspace.brfft_plan, workspace.fft_plan)

    return nothing
end

"""
Used to generate values for the `(begin+k,begin+l)` element of `ZDTWorkspace.V`
(before its FFT is computed).  `k` and `l` are zero-based offsets.  This
function should be used to populate elements in the first `Nr` columns of `V`.
`δr` is the drift rate step size.  `Nf` is the number of frequency channels in
the input spectrogram.
"""
function vlow(k::Integer, l::Integer, δr::Float32, Nf::Integer)
    cispi(k * l^2 * δr / Nf)
end

"""
Used to generate values for the `(begin+k,begin+l)` element of `ZDTWorkspace.V`
(before its FFT is computed).  `k` and `l` are zero-based offsets.  This
function should be used to populate elements in the last `Nt-1` columns of `V`.
`δr` is the drift rate step size.  `Nf` is the number of frequency channels in
the input spectrogram.  `Nl` is the size of the second dimension of `V`.
"""
function vhigh(k::Integer, l::Integer, δr::Float32, Nf::Integer, Nl::Integer)
    vlow(k, Nl-l, δr, Nf)
end

"""
Used to generate values for the `kl` element of `ZDTWorkspace.V` (before its FFT
is computed).  `kl` is a `CartesianIndex`.  This function should be used to
populate elements in the first `Nr` columns of `V`.  `δr` is the drift rate step
size.  `Nf` is the number of frequency channels in the input spectrogram.
"""
function vlow(kl::CartesianIndex, δr::Float32, Nf::Integer)
    vlow(kl[1]-1, kl[2]-1, δr, Nf)
end

"""
Used to generate values for the `kl` element of `ZDTWorkspace.V` (before its FFT
is computed).  `kl` is a `CartesianIndex`.  This function should be used to
populate elements in the last `Nt-1` columns of `V`.  `δr` is the drift rate step
size.  `Nf` is the number of frequency channels in the input spectrogram.  `Nl`
is the sie of the second dimension of `V`.
"""
function vhigh(kl::CartesianIndex, δr::Float32, Nf::Integer, Nl::Integer)
    vlow(kl[1]-1, Nl-(kl[2]-1), δr, Nf)
end

"""
Compute the contents of `workspace.V`, which is the FFT of the convolving
function of the CZT for the Zdop parameters contained in `workspace`.
"""
function computeV!(workspace::ZDTWorkspace)
    Nf = workspace.Nf
    Nt = workspace.Nt
    δr = workspace.δr
    Nr = workspace.Nr
    Nl = workspace.Nl
    V  = workspace.V
    # Populate the low portion of V
    V[:, 1:Nr] .= vlow.(CartesianIndices((axes(V,1), 1:Nr)), δr, Nf)
    # Populate the high portion of V
    lastl = lastindex(V, 2)
    V[:, end-Nt+2:end] .= vhigh.(CartesianIndices((axes(V,1), lastl-Nt+2:lastl)), δr, Nf, Nl)
    # FFT V in-place
    mul!(V, workspace.fft_plan, V)
end

"""
Generate phase factors used in the *pre-multiply* step of the CZT.  `k` and `l`
are zero-based offsets.
"""
function prephase(k::Integer, l::Integer, r0::Float32, δr::Float32, Nf::Integer)
    cispi(-k * l * (l * δr + 2 * r0) / Nf)
end

"""
Generate phase factors used in the *pre-multiply* step of the CZT.  `kl` is a
`CartesianIndex`.
"""
function prephase(kl::CartesianIndex, r0::Float32, δr::Float32, Nf::Integer)
    prephase(kl[1]-1, kl[2]-1, r0, δr, Nf)
end

"""
FFT `spectrogram` into `workspace.F`.
"""
function initialize!(workspace, spectrogram)
    # FFT `spectrogram` into `workspace.F`
    mul!(workspace.F, workspace.rfft_plan, spectrogram)
end

"""
Multiply `workspace.F` by `prephase` as per the parameters in `workspace`,
storing results in `workspace.Yf`, then zero-pad the rest of `workspace.Y`.
`r0` can be optionally specified to override `workspace.r0`.
"""
function preprocess!(workspace, r0::Float32=workspace.r0)
    Nf = workspace.Nf
    Nt = workspace.Nt
    δr = workspace.δr
    F  = workspace.F
    Yf = workspace.Yf
    Y  = workspace.Y

    # Multiply `F` by `prephase` as per the parameters from `workspace`
    Yf .= F .* prephase.(CartesianIndices(F), r0, δr, Nf)

    # Zero-pad the rest of `Y`
    # TODO: Add Yz field to ZDTWorkspace for this view?
    fill!(@view(Y[:, Nt+1:end]), zero(eltype(Y)))
end

"""
Perform CZT convolution step for data in `workspace` by doing:
1. In-place FFT `workspace.Y`
2. In-place multiply of `workspace.Y` by `workspace.V`
3. In-place backwards multiply of `Workspace.Y`
"""
function convolve!(workspace)
    mul!(workspace.Y, workspace.fft_plan, workspace.Y)
    workspace.Y .*= workspace.V
    mul!(workspace.Y, workspace.bfft_plan, workspace.Y)
end

"""
Generate phase factors used in the *post-multiply* step of the CZT.  `k` and `l`
are zero-based offsets.
"""
function postphase(k::Integer, l::Integer, δr::Float32, Nf::Integer)
    cispi(-k * l * l * δr / Nf)
end

"""
Generate phase factors used in the *post-multiply* step of the CZT.  `kl` is a
`CartesianIndex`.
"""
function postphase(kl::CartesianIndex, δr::Float32, Nf::Integer)
    postphase(kl[1]-1, kl[2]-1, δr, Nf)
end

"""
Multiply `workspace.Ys` by `postphase` as per the parameters in `workspace` and
backwards FFT `workspace.Ys` into `dest`
"""
function postprocess!(dest, workspace)
    # Multiply `workspace.Ys` by `postphase` as per the parameters in `workspace`
    workspace.Ys .*= postphase.(CartesianIndices(workspace.Ys),
                                workspace.δr, workspace.Nf)
    # Backwards FFT `workspace.Ys` into `dest`
    # TODO Make this a separate function?
    mul!(dest, workspace.brfft_plan, workspace.Ys)
end

"""
Compute the frequency drift rate matrix via the ZDop algorithm as specified in
`workspace` and output results into `dest`.  `r0` can be optionally specified to
override `workspace.r0`.
"""
function zdtfdr!(dest, workspace; r0::Real=workspace.r0)
    preprocess!(workspace, Float32(r0))
    convolve!(workspace)
    postprocess!(dest, workspace)
end

"""
Compute the frequency drift rate matrix for `spectrogram` via the ZDop algorithm
as specified in `workspace` and output results into `dest`.  `r0` can be
optionally specified to override `workspace.r0`.
"""
function zdtfdr!(dest, workspace, spectrogram; r0::Real=workspace.r0)
    initialize!(workspace, spectrogram)
    zdtfdr!(dest, workspace; r0=r0)
end

"""
Compute the frequency drift rate matrix via the ZDop algorithm as specified in
`workspace` and return it as newly allocated matrix.  `r0` can be optionally
specified to override `workspace.r0`.
"""
function zdtfdr(workspace; r0::Real=workspace.r0)
    Nf = workspace.Nf
    Nr = workspace.Nr
    dest = similar(workspace.Ys, real(eltype(workspace.Ys)), Nf, Nr)
    zdtfdr!(dest, workspace; r0=r0)
end

"""
Compute the frequency drift rate matrix for `spectrogram` via the ZDop algorithm
as specified in `workspace` and return it as newly allocated matrix.  `r0` can
be optionally specified to override `workspace.r0`.
"""
function zdtfdr(workspace, spectrogram; r0::Real=workspace.r0)
    initialize!(workspace, spectrogram)
    zdtfdr(workspace; r0=r0)
end
