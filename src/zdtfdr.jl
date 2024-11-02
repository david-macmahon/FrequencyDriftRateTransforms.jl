using AbstractFFTs
using LinearAlgebra

# We need to depend on FFTW for access to its flags constants because
# AbstractFFTs defines the planning flags to be "a bitwise-or of FFTW planner
# flags".  For more details, see:
# https://github.com/JuliaMath/AbstractFFTs.jl/issues/71
import FFTW

mutable struct ZDTWorkspace{T}
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
    irfft_plan::AbstractFFTs.Plan

    # CZT FFT plans (in-place)
    fft_plan::AbstractFFTs.Plan
    ifft_plan::AbstractFFTs.Plan

    # Some specializations (e.g. for CUDA) need to allocate a work area for
    # FFTs, which can be stored here.
    fft_workarea::Union{Nothing,AbstractArray{UInt8}}
    
    function ZDTWorkspace(spectrogram::T,
                          r0::Real, δr::Real, Nr::Integer,
                          factors::Union{Tuple,AbstractVector}=(2,3,5);
                          output_aligned=false) where {T<:AbstractMatrix{<:Real}}
        Nf, Nt = size(spectrogram)
        Nl = calcNl(Nt, Nr, factors)

        F = similar(spectrogram, complex(eltype(spectrogram)), Nf÷2+1, Nt)
        Y = similar(spectrogram, complex(eltype(spectrogram)), Nf÷2+1, Nl)

        Yf = @view Y[:, 1:Nt]
        Ys = @view Y[:, 1:Nr]

        ws = new{T}(
            Nf, Nt, r0, δr, Nr, Nl, factors,
            F, Yf, Y, Ys
        )

        # Call function to plan FFTs.  This allows for specialization based on
        # the Array type of spectrogram.
        plan_ffts!(ws, spectrogram; output_aligned=output_aligned)

        # Initialize F with FFT of spectrogram
        input!(ws, spectrogram)

        # Any V locations that fall between vlow and vhigh are described as
        # "don't care" values, but actually they are "don't care so long as they
        # are not NaN" values, so we need to initialize them to non-NaN values.
        # The easiest way to do that is to initialize all of V to zero.
        ws.V = zero(Y)

        # Precompute V
        computeV!(ws)

        # Wait for input! and computeV! or not, depending on typeof(spectrogram)
        fdrsynchronize(typeof(spectrogram))

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

function Base.sizeof(ws::ZDTWorkspace)
    mapreduce(p->sizeof(getproperty(ws, p)), +, propertynames(ws))
end

"""
    plan_ffts!(workspace::ZDTWorkspace, spectrogram; output_aligned=false)

Make the ZDT's FFT plans for `spectrogram::AbstractArray` for which a more
specialized method is not available.
"""
function plan_ffts!(workspace::ZDTWorkspace,
                    spectrogram::AbstractMatrix{<:Real};
                    output_aligned::Bool=false)
    Nf = workspace.Nf
    Y = workspace.Y
    Ys = workspace.Ys
    irfft_flags = FFTW.ESTIMATE | (output_aligned ? 0 : FFTW.UNALIGNED)

    workspace.rfft_plan = plan_rfft(spectrogram, 1)
    workspace.irfft_plan = plan_irfft(Ys, Nf, 1; flags=irfft_flags)

    workspace.fft_plan = plan_fft!(Y, 2)
    workspace.ifft_plan = plan_ifft!(Y, 2)

    workspace.fft_workarea = nothing

    return nothing
end

"""
    vlow(k, l, δr::Float32, Nf) -> Complex{Float32}
    vlow(kl::CartesianIndex, δr::Float32, Nf::Integer) -> ComplexF32

Used to generate values for the `(begin+k,begin+l)` element of `ZDTWorkspace.V`
(before its FFT is computed).  `k` and `l` are zero-based offsets.  `kl` is a
one-based `CartesianIndex`.  This function should be used to populate elements
in the first `Nr` columns of `V`.  `δr` is the drift rate step size.  `Nf` is
the number of frequency channels in the input spectrogram.
"""
function vlow(k::Integer, l::Integer, δr::Float32, Nf::Integer)
    cispi(-k * l^2 * δr / Nf)
end

function vlow(kl::CartesianIndex, δr::Float32, Nf::Integer)
    vlow(kl[1]-1, kl[2]-1, δr, Nf)
end

"""
    vhigh(k, l, δr::Float32, Nf, Nl) -> ComplexF32
    vhigh(kl::CartesianIndex, δr::Float32, Nf::Integer, Nl::Integer)

Used to generate values for the `(begin+k,begin+l)` element of `ZDTWorkspace.V`
(before its FFT is computed).  `k` and `l` are zero-based offsets.  `kl` is a
one-based `CartesianIndex`.  This function should be used to populate elements
in the last `Nt-1` columns of `V`.  `δr` is the drift rate step size.  `Nf` is
the number of frequency channels in the input spectrogram.  `Nl` is the size of
the second dimension of `V`.
"""
function vhigh(k::Integer, l::Integer, δr::Float32, Nf::Integer, Nl::Integer)
    vlow(k, Nl-l, δr, Nf)
end

function vhigh(kl::CartesianIndex, δr::Float32, Nf::Integer, Nl::Integer)
    vlow(kl[1]-1, Nl-(kl[2]-1), δr, Nf)
end

"""
    computeV!(workspace::ZDTWorkspace)

Compute and update the contents of `workspace.V`, which is the FFT of the
convolving function of the CZT for the ZDT parameters contained in `workspace`.
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
    prephase(k, l, r0::Float32, δr, Nf) -> ComplexF32
    prephase(kl::CartesianIndex, r0::Float32, δr::Float32, Nf::Integer)

Generate phase factors used in the *pre-multiply* step of the CZT.  `k` and `l`
are zero-based offsets. `kl` is a one-based `CartesianIndex`.
"""
function prephase(k::Integer, l::Integer, r0::Float32, δr::Float32, Nf::Integer)
    cispi(k * l * (l * δr + 2 * r0) / Nf)
end

function prephase(kl::CartesianIndex, r0::Float32, δr::Float32, Nf::Integer)
    prephase(kl[1]-1, kl[2]-1, r0, δr, Nf)
end

"""
    input!(workspace, spectrogram)

Input FFT of `spectrogram` into `workspace.F`.
"""
function input!(workspace, spectrogram)
    # FFT `spectrogram` into `workspace.F`
    mul!(workspace.F, workspace.rfft_plan, spectrogram)
end

"""
    preprocess!(workspace, r0=workspace.r0)

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

function preprocess!(workspace, r0::Real)
    preprocess!(workspace, Float32(r0))
end

"""
    convolve!(workspace)

Perform CZT convolution step for data in `workspace` by doing:
1. In-place FFT `workspace.Y`
2. In-place multiply of `workspace.Y` by `workspace.V`
3. In-place backwards FFT of `Workspace.Y`
"""
function convolve!(workspace)
    mul!(workspace.Y, workspace.fft_plan, workspace.Y)
    workspace.Y .*= workspace.V
    mul!(workspace.Y, workspace.ifft_plan, workspace.Y)
end

"""
    postphase([w,] k::Integer, l::Integer, δr::Float32, Nf::Integer) -> ComplexF32
    postphase([w,] kl::CartesianIndex, δr::Float32, Nf::Integer)

Generate phase factors used in the *post-multiply* step of the CZT.  `k` and `l`
are zero-based offsets.  `kl` is a one-based `CartesianIndex`.

`w` specifies the windowing function to apply prior to the final output inverse
FFT.  It may be given as `:rect` to use a rectangular window (the default),
`:hamming` to use a Hamming window, or a two-arg function that will be passed
the zero-indexed channel number and the total number of channels and should
return the window value for that channel number.
"""
function postphase(w::Function, k::Integer, l::Integer, δr::Float32, Nf::Integer)
    cispi(k * l * l * δr / Nf) * w(k,Nf)
end

function postphase(k::Integer, l::Integer, δr::Float32, Nf::Integer)
    postphase((n,N)->1, k, l, δr, Nf) # Default to rectangular window
end

# postphase CartesianIndex methods

function postphase(w::Function, kl::CartesianIndex, δr::Float32, Nf::Integer)
    postphase(w, kl[1]-1, kl[2]-1, δr, Nf)
end

function postphase(kl::CartesianIndex, δr::Float32, Nf::Integer)
    postphase((n,N)->1, kl, δr, Nf, w) # Default to rectangular window
end

"""
    postprocess!([w,] workspace)

Multiply `workspace.Ys` by `postphase` as per the parameters in `workspace`.

`w` specifies the windowing function to apply prior to the final output inverse
FFT.  It may be given as `:rect` to use a rectangular window (the default),
`:hamming` to use a Hamming window, or a two-arg function that will be passed
the zero-indexed channel number and the total number of channels and should
return the window value for that channel number.
"""
function postprocess!(w::Function, workspace)
    # Multiply `workspace.Ys` by `postphase` as per the parameters in `workspace`
    workspace.Ys .*= postphase.(w, CartesianIndices(workspace.Ys),
                                workspace.δr, workspace.Nf)
end

function postprocess!(::Val{:hamming}, workspace)
    postprocess!((n,N)->(0.53836 + 0.46164 * cospi(2n/N)), workspace)
end

function postprocess!(::Val{:rect}, workspace)
    postprocess!((n,N)->1, workspace)
end

function postprocess!(::Val{S}, workspace) where S
    error("unsupported window type ($S)")
end

function postprocess!(w::Symbol, workspace)
    postprocess!(Val(w), workspace)
end

function postprocess!(workspace)
    postprocess!(Val(:rect), workspace)
end

"""
    output!(dest, workspace) -> dest

Output ZDT results into `dest`, which should have size `(Nf, Nr)`.
"""
function output!(dest, workspace)
    # Backwards FFT `workspace.Ys` into `dest`
    mul!(dest, workspace.irfft_plan, workspace.Ys)
end

"""
    zdtfdr!([dest,] workspace[, spectrogram]; r0=workspace.r0)

If `spectrogram` is given, `input!` it into `workspace.F`.  Perform the ZDT
algorithm as specified in `workspace`.  If `dest` is given, `output!` frequency
drift rate matrix into `dest` and return `dest`, otherwise return `nothing`.  An
alternate `r0` may be given to override `workspace.r0`.  `dest` and `r0` may
also be iterators to compute multiple ZDTs from the same input for different r0
values.

`w` specifies the windowing function to apply prior to the final output inverse
FFT.  It may be given as `:rect` to use a rectangular window (the default),
`:hamming` to use a Hamming window, or a two-arg function that will be passed
the zero-indexed channel number and the total number of channels and should
return the window value for that channel number.
"""
function zdtfdr!(w::Union{Function,Symbol,Val}, dests, workspace, spectrogram=nothing; r0=workspace.r0)
    if spectrogram !== nothing
        input!(workspace, spectrogram)
    end

    for (dest, rate) in zip(dests, Iterators.cycle(r0))
        preprocess!(workspace, rate)
        convolve!(workspace)
        postprocess!(w, workspace)
        output!(dest, workspace)
    end

    return dests
end

function zdtfdr!(dests, workspace, spectrogram=nothing; r0=workspace.r0)
    zdtfdr!(Val(:rect), dests, workspace, spectrogram; r0)
end

# dest as standalone Matrix

function zdtfdr!(w::Union{Function,Symbol,Val}, dest::AbstractMatrix{<:Real}, workspace, spectrogram=nothing; r0::Real=workspace.r0)
    zdtfdr!(w, (dest,), workspace, spectrogram; r0)
    return dest
end

function zdtfdr!(dest::AbstractMatrix{<:Real}, workspace, spectrogram=nothing; r0::Real=workspace.r0)
    zdtfdr!(Val(:rect), (dest,), workspace, spectrogram; r0)
end

# No dest

function zdtfdr!(w::Union{Function,Symbol,Val}, workspace::ZDTWorkspace, spectrogram=nothing; r0::Real=workspace.r0)
    if spectrogram !== nothing
        input!(workspace, spectrogram)
    end

    preprocess!(workspace, r0)
    convolve!(workspace)
    postprocess!(w, workspace)

    return nothing
end

function zdtfdr!(workspace::ZDTWorkspace, spectrogram=nothing; r0::Real=workspace.r0)
    zdtfdr!(Val(:rect), workspace::ZDTWorkspace, spectrogram; r0)
end

"""
    zdtfdr([w,] workspace[, spectrogram]; r0=workspace.r0)

If `spectrogram` is given, `input!` it into `workspace.F`.  Perform the ZDT
algorithm as specified in `workspace`, `output!` frequency drift rate matrix to
a newly allocated `Matrix` and return it.  An alternate `r0` may be given to
override `workspace.r0`.

`w` specifies the windowing function to apply prior to the final output inverse
FFT.  It may be given as `:rect` to use a rectangular window (the default),
`:hamming` to use a Hamming window, or a two-arg function that will be passed
the zero-indexed channel number and the total number of channels and should
return the window value for that channel number.
"""
function zdtfdr(w::Union{Function,Symbol,Val}, workspace, spectrogram=nothing; r0::Real=workspace.r0)
    Nf = workspace.Nf
    Nr = workspace.Nr
    dest = similar(workspace.Ys, real(eltype(workspace.Ys)), Nf, Nr)
    zdtfdr!(w, dest, workspace, spectrogram; r0=r0)
end

function zdtfdr(workspace, spectrogram=nothing; r0::Real=workspace.r0)
    zdtfdr(Val(:rect), workspace, spectrogram; r0)
end
