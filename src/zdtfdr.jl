using AbstractFFTs
using LinearAlgebra

# We need to depend on FFTW for access to its flags constants because
# AbstractFFTs defines the planning flags to be "a bitwise-or of FFTW planner
# flags".  For more details, see:
# https://github.com/JuliaMath/AbstractFFTs.jl/issues/71
import FFTW

"""
    ZDTWorkspace(spectrogram, rates[, factors=(2, 3, 5)]; output_aligned=false) -> workspace

Create a *workspace* suitable for use with `zdtfdr` and `zdtfdr!`.  The
workspace includes the intermediate storage buffers and FFT plans needed for
creating a frequency drift rate matrix for `spectrogram` using the ZDT
algorithm for the `length(rates)` normalized drift rates in `rates` (see
[`driftrates`](@ref)).  `factors` restricts the internal FFT sizes (see
[`calcNl`](@ref)).  A workspace can be reused for any spectrogram of the same
size and element type; pass a new `spectrogram` to `zdtfdr!` to reload it.
`output_aligned` may be set to `true` if the output buffers will be suitably
aligned for the FFT implementation.
"""
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
    Ys2::AbstractMatrix{<:Complex}
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

        # The convolve buffers (`Y` and `V`) are stored *rate-major* (drift
        # rate along the first, contiguous dimension) so that the CZT FFTs
        # run on contiguous data; `F` and `Ys2` stay freq-major for the
        # input/output real FFTs.  The layout changes are fused into
        # `zdtpreprocess!` and `zdtpostprocess!`, which already rewrite the
        # whole array.
        F = similar(spectrogram, complex(eltype(spectrogram)), Nf÷2+1, Nt)
        Y = similar(spectrogram, complex(eltype(spectrogram)), Nl, Nf÷2+1)
        Ys2 = similar(spectrogram, complex(eltype(spectrogram)), Nf÷2+1, Nr)

        Yf = @view Y[1:Nt, :]
        Ys = @view Y[1:Nr, :]

        ws = new{T}(
            Nf, Nt, r0, δr, Nr, Nl, factors,
            F, Yf, Y, Ys, Ys2
        )

        # Call function to plan FFTs.  This allows for specialization based on
        # the Array type of spectrogram.
        plan_ffts!(ws, spectrogram; output_aligned=output_aligned)

        # Initialize F with FFT of spectrogram
        zdtinput!(ws, spectrogram)

        # Any V locations that fall between vlow and vhigh are described as
        # "don't care" values, but actually they are "don't care so long as they
        # are not NaN" values, so we need to initialize them to non-NaN values.
        # The easiest way to do that is to initialize all of V to zero.
        ws.V = zero(Y)

        # Precompute V
        computeV!(ws)

        # Wait for zdtinput! and computeV! or not, depending on typeof(spectrogram)
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

NB: the CPU FFT plans are created without enabling FFTW's multithreading
because `FFTW.set_num_threads` sets process-global state, which a package
should not silently change.  To use multiple threads for the CPU FFTs, call
`FFTW.set_num_threads(Threads.nthreads())` before constructing the workspace;
plans pick up the thread count at planning time.
"""
function plan_ffts!(workspace::ZDTWorkspace,
                    spectrogram::AbstractMatrix{<:Real};
                    output_aligned::Bool=false)
    Nf = workspace.Nf
    Y = workspace.Y
    Ys2 = workspace.Ys2
    irfft_flags = FFTW.ESTIMATE | (output_aligned ? 0 : FFTW.UNALIGNED)

    workspace.rfft_plan = plan_rfft(spectrogram, 1)
    workspace.irfft_plan = plan_irfft(Ys2, Nf, 1; flags=irfft_flags)

    workspace.fft_plan = plan_fft!(Y, 1)
    workspace.ifft_plan = plan_ifft!(Y, 1)

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
    # `V` is stored rate-major, so broadcast the zero-based channel index
    # along the rows and the zero-based rate index down the columns to keep
    # the (k, l) argument order of `vlow`/`vhigh`.
    K = Nf÷2 + 1
    # Populate the low portion of V (zero-based drift rates 0:Nr-1)
    V[1:Nr, :] .= vlow.((0:K-1)', 0:Nr-1, δr, Nf)
    # Populate the high portion of V (zero-based drift rates Nl-Nt+1:Nl-1)
    V[end-Nt+2:end, :] .= vhigh.((0:K-1)', (Nl-Nt+1):(Nl-1), δr, Nf, Nl)
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
    zdtinput!(workspace, spectrogram) -> workspace

Input FFT of `spectrogram` into `workspace.F`.
"""
function zdtinput!(workspace, spectrogram)
    # FFT `spectrogram` into `workspace.F`
    mul!(workspace.F, workspace.rfft_plan, spectrogram)
    return workspace
end

# Multithreaded, cache-blocked transpose: `dest[j, i] = src[i, j]`.  Blocking
# keeps both sides' cache lines fully utilized (a naive transpose touches a
# whole cache line per element); the innermost loop over the contiguous
# dimension vectorizes.
function blocked_transpose!(dest, src)
    K, N = size(src)
    T = 64
    @sync for j0 in 1:T:N
        Threads.@spawn for i0 in 1:T:K
            @inbounds for j in j0:min(j0 + T - 1, N)
                @simd for i in i0:min(i0 + T - 1, K)
                    dest[j, i] = src[i, j]
                end
            end
        end
    end
    return dest
end

"""
    zdtpreprocess!(workspace[, r0]) -> workspace

Multiply `workspace.F` by `prephase` as per the parameters in `workspace`,
storing results in `workspace.Yf`, then zero-pad the rest of `workspace.Y`.
`r0` can be optionally specified to override `workspace.r0`.
"""
function zdtpreprocess!(workspace, r0::Float32=workspace.r0)
    Nf = workspace.Nf
    Nt = workspace.Nt
    δr = workspace.δr
    F  = workspace.F
    Yf = workspace.Yf
    Y  = workspace.Y

    # Multiply `workspace.F` by `prephase` as per the parameters from
    # `workspace`, storing the result transposed into the rate-major `Yf`
    # view of `workspace.Y` (this fuses the layout change required by
    # `zdtconvolve!` into this pass).
    Yft = PermutedDimsArray(Yf, (2, 1))
    Yft .= F .* prephase.(CartesianIndices(F), r0, δr, Nf)

    # Zero-pad the rest of `Y` (the rows beyond `Nt`)
    # TODO: Add Yz field to ZDTWorkspace for this view?
    fill!(@view(Y[Nt+1:end, :]), zero(eltype(Y)))
    return workspace
end

# CPU-optimized variant: a fused transposed broadcast is slow for CPU arrays
# (strided scalar access), so transpose with blocked multithreaded loops
# first and then multiply by `prephase` in place (contiguous).  Dispatch on
# `Array` (not `StridedArray`, which in recent Julia versions also matches
# GPU arrays); other CPU wrappers fall back to the generic method.
function zdtpreprocess!(workspace::ZDTWorkspace{<:Array}, r0::Float32)
    Nf = workspace.Nf
    Nt = workspace.Nt
    δr = workspace.δr
    F  = workspace.F
    Yf = workspace.Yf
    Y  = workspace.Y

    blocked_transpose!(Yf, F)
    Yf .*= prephase.((0:Nf÷2)', 0:Nt-1, r0, δr, Nf)
    fill!(@view(Y[Nt+1:end, :]), zero(eltype(Y)))
    return workspace
end

function zdtpreprocess!(workspace, r0::Real)
    zdtpreprocess!(workspace, Float32(r0))
end

"""
    zdtconvolve!(workspace) -> workspace

Perform CZT convolution step for data in `workspace` by doing:
1. In-place FFT `workspace.Y`
2. In-place multiply of `workspace.Y` by `workspace.V`
3. In-place backwards FFT of `Workspace.Y`
"""
function zdtconvolve!(workspace)
    mul!(workspace.Y, workspace.fft_plan, workspace.Y)
    workspace.Y .*= workspace.V
    mul!(workspace.Y, workspace.ifft_plan, workspace.Y)
    return workspace
end

"""
    postphase([w,] k::Integer, l::Integer, δr::Float32, Nf::Integer) -> ComplexF32
    postphase([w,] kl::CartesianIndex, δr::Float32, Nf::Integer)

Generate phase factors used in the *post-multiply* step of the CZT.  `k` and `l`
are zero-based offsets.  `kl` is a one-based `CartesianIndex`.

`w` specifies the windowing function to apply prior to the final output inverse
FFT.  It may be given as `:rect` to use a rectangular window (the default),
`:hamming` or `:binom5` to use Hamming or 5-point-binomial smoothing
windows, or a two-arg function that will be
passed the zero-indexed channel number and the total number of channels and
should return the window value for that channel number.  For kernel recipes
(including generalized binomial windows) and Gibbs-ringing/apodization
guidance, see the extended help of [`zdtfdr`](@ref).
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
    postphase((n,N)->1, kl, δr, Nf) # Default to rectangular window
end

"""
    zdtpostprocess!([w,] workspace) -> workspace

Read `workspace.Ys`, multiply it by `postphase` as per the parameters in
`workspace`, and store the result transposed in `workspace.Ys2` (which
[`zdtoutput!`](@ref) consumes).

`w` specifies the windowing function to apply prior to the final output inverse
FFT.  It may be given as `:rect` to use a rectangular window (the default),
`:hamming` or `:binom5` to use Hamming or 5-point-binomial smoothing
windows, or a two-arg function that will be
passed the zero-indexed channel number and the total number of channels and
should return the window value for that channel number.  For details about the
window function, see the extended help of [`zdtfdr`](@ref).
"""
function zdtpostprocess!(w::Function, workspace)
    # Multiply `workspace.Ys` by `postphase` as per the parameters in
    # `workspace`, storing the result transposed in the freq-major `Ys2`
    # (this fuses the layout change required by `zdtoutput!` into this
    # pass).  `Ys` is rate-major, so broadcast the zero-based channel index
    # along the rows and the zero-based rate index down the columns to keep
    # the (k, l) argument order of `postphase`.
    Yst = PermutedDimsArray(workspace.Ys, (2, 1))
    workspace.Ys2 .= postphase.(w, 0:workspace.Nf÷2, (0:workspace.Nr-1)',
                                workspace.δr, workspace.Nf) .* Yst
    return workspace
end

# CPU-optimized variant (see the `zdtpreprocess!` note): blocked transpose
# followed by an in-place contiguous phase multiply.
function zdtpostprocess!(w::Function, workspace::ZDTWorkspace{<:Array})
    blocked_transpose!(workspace.Ys2, workspace.Ys)
    workspace.Ys2 .*= postphase.(w, CartesianIndices(workspace.Ys2),
                                 workspace.δr, workspace.Nf)
    return workspace
end

function zdtpostprocess!(::Val{:hamming}, workspace)
    zdtpostprocess!((n,N)->(0.53836 + 0.46164 * cospi(2n/N)), workspace)
end

function zdtpostprocess!(::Val{:binom5}, workspace)
    zdtpostprocess!((n, N) -> (6 + 8cospi(2n / N) + 2cospi(4n / N)) / 16,
                    workspace)
end

function zdtpostprocess!(::Val{:rect}, workspace)
    zdtpostprocess!((n,N)->1, workspace)
end

function zdtpostprocess!(::Val{S}, workspace) where S
    error("unsupported window type ($S)")
end

function zdtpostprocess!(w::Symbol, workspace)
    zdtpostprocess!(Val(w), workspace)
end

function zdtpostprocess!(workspace)
    zdtpostprocess!(Val(:rect), workspace)
end

"""
    zdtoutput!(dest, workspace) -> dest

Output ZDT results into `dest`, which should have size `(Nf, Nr)`.
"""
function zdtoutput!(dest, workspace)
    # Backwards FFT `workspace.Ys2` into `dest`
    mul!(dest, workspace.irfft_plan, workspace.Ys2)
end

"""
    zdtfdr!([w,] [dest,] workspace[, spectrogram]; r0=workspace.r0)

If `spectrogram` is given, `zdtinput!` it into `workspace.F`.  Perform the ZDT
algorithm as specified in `workspace`.  If `dest` is given, `zdtoutput!` frequency
drift rate matrix into `dest` and return `dest`, otherwise return `nothing`.  An
alternate `r0` may be given to override `workspace.r0`.  `dest` and `r0` may
also be iterators to compute multiple ZDTs from the same input for different r0
values.

`w` specifies the windowing function to apply prior to the final output inverse
FFT.  It may be given as `:rect` to use a rectangular window (the default),
`:hamming` or `:binom5` to use Hamming or 5-point-binomial smoothing
windows, or a two-arg function that will be
passed the zero-indexed channel number and the total number of channels and
should return the window value for that channel number.  For details about the
window function, see the extended help of [`zdtfdr`](@ref).
"""
function zdtfdr!(w::Union{Function,Symbol,Val}, dests, workspace, spectrogram=nothing; r0=workspace.r0)
    if spectrogram !== nothing
        zdtinput!(workspace, spectrogram)
    end

    for (dest, rate) in zip(dests, Iterators.cycle(r0))
        zdtpreprocess!(workspace, rate)
        zdtconvolve!(workspace)
        zdtpostprocess!(w, workspace)
        zdtoutput!(dest, workspace)
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
    return dest
end

# No dest

function zdtfdr!(w::Union{Function,Symbol,Val}, workspace::ZDTWorkspace, spectrogram=nothing; r0::Real=workspace.r0)
    if spectrogram !== nothing
        zdtinput!(workspace, spectrogram)
    end

    zdtpreprocess!(workspace, r0)
    zdtconvolve!(workspace)
    zdtpostprocess!(w, workspace)

    return nothing
end

function zdtfdr!(workspace::ZDTWorkspace, spectrogram=nothing; r0::Real=workspace.r0)
    zdtfdr!(Val(:rect), workspace::ZDTWorkspace, spectrogram; r0)
end

"""
    zdtfdr([w,] workspace[, spectrogram]; r0=workspace.r0)

If `spectrogram` is given, `zdtinput!` it into `workspace.F`.  Perform the ZDT
algorithm as specified in `workspace`, `zdtoutput!` frequency drift rate matrix to
a newly allocated `Matrix` and return it.  An alternate `r0` may be given to
override `workspace.r0`.

`w` specifies the windowing function to apply prior to the final output inverse
FFT.  It may be given as `:rect` to use a rectangular window (the default),
`:hamming` or `:binom5` to use Hamming or 5-point-binomial smoothing
windows, or a two-arg function that will be
passed the zero-indexed channel number and the total number of channels and
should return the window value for that channel number.  For details about the
window function, see the extended help below.

# Extended help

To implement a desired kernel, supply its forward FFT as the window, centered
at channel 0; e.g. the window `a + b*cospi(2n/N)` with `a + b = 1` yields the
3-point smoothing kernel `[b/2, a, b/2]`.  (Real, even-symmetric windows
produce real, symmetric kernels.)  NB: for smoothing, the peak should be at
channel 0 rather than N/2 because the data are in FFT order.  The `:hamming`
window uses `a = 0.53836` and `b = 0.46164` to provide the classic 3-point
kernel `[0.23082, 0.53836, 0.23082]`.  The `:binom5` window provides the
5-point binomial smoothing kernel `[1, 4, 6, 4, 1]/16`, with spectral form
`(6 + 8cospi(2n/N) + 2cospi(4n/N))/16`.

The binomial family generalizes directly: the symmetric `(2m + 1)`-point
binomial smoothing kernel (Pascal's triangle row `2m`, normalized, e.g.
`[1, 6, 15, 20, 15, 6, 1]/64` for `m = 3`) has the spectral form
`cospi(n/N)^(2m)`, so passing `w = (n, N) -> cospi(n / N)^(2m)` as the
window yields the `2m + 1`-point binomial window (`:binom5` is the
`m = 2` case).  Each increment of `m` adds a higher-order zero of the
spectral response at the band edge (Nyquist), further suppressing
channel-scale structure such as interpolation ringing, at the cost of a
proportionally wider smoothing kernel.

## Gibbs ringing and apodization

The ZDT realizes fractional drift rates with pure Fourier phase factors, so
each time column of the spectrogram is effectively band-limited
(periodic-sinc) interpolated along frequency.  Bright, narrow lines ring
under that interpolation (the Gibbs phenomenon), producing spurious
oscillating structure in the FDR near the brightest peaks, which can show
up as false positives at modest signal-to-noise ratios.  Applying a
smoothing window `w` low-pass filters the FDR along the frequency axis;
because a frequency convolution commutes with the shear-and-sum of the
dedoppler transform, this is exactly equivalent to apodizing (smoothing)
the input spectrogram before the transform, which bounds the ringing by
the (designed) sidelobes of the smoothing kernel.  The cost is frequency
resolution: features that span several channels gain signal-to-noise
ratio under smoothing, while a signal confined to a single channel loses
roughly 16% in sigma with `:hamming`, and the FDR noise becomes
correlated across the width of the smoothing kernel.  When the data
contain very bright narrow lines, the `:hamming` or `:binom5` windows
are recommended for suppressing ringing near those lines.
"""
function zdtfdr(w::Union{Function,Symbol,Val}, workspace, spectrogram=nothing; r0::Real=workspace.r0)
    Nf = workspace.Nf
    Nr = workspace.Nr
    dest = similar(workspace.Ys2, real(eltype(workspace.Ys2)), Nf, Nr)
    zdtfdr!(w, dest, workspace, spectrogram; r0=r0)
end

function zdtfdr(workspace, spectrogram=nothing; r0::Real=workspace.r0)
    zdtfdr(Val(:rect), workspace, spectrogram; r0)
end

"""
    zdtfdr([w,] spectrogram, rates[, factors]; output_aligned=false) -> fdr

One-shot form of `zdtfdr`: construct a [`ZDTWorkspace`](@ref) for `spectrogram`
and `rates` (and `factors`) and return `zdtfdr(workspace)` (or `zdtfdr(w,
workspace)` for a windowing function `w`; see [`zdtfdr!`](@ref) for `w`).
Constructing a workspace for every call is relatively expensive; reuse a
workspace when computing multiple ZDTs of the same size.
"""
function zdtfdr(spectrogram::AbstractMatrix{<:Real}, rates::AbstractRange,
                factors=(2, 3, 5); output_aligned=false)
    zdtfdr(ZDTWorkspace(spectrogram, rates, factors; output_aligned=output_aligned))
end

function zdtfdr(w::Union{Function,Symbol,Val}, spectrogram::AbstractMatrix{<:Real},
                rates::AbstractRange, factors=(2, 3, 5); output_aligned=false)
    zdtfdr(w, ZDTWorkspace(spectrogram, rates, factors; output_aligned=output_aligned))
end

# Deprecated aliases of the generic ZDT pipeline stage names, which are too
# collision-prone to export unqualified.

"""
    input!(workspace, spectrogram)

Deprecated alias of [`zdtinput!`](@ref).
"""
function input!(args...; kwargs...)
    Base.depwarn("`input!` is deprecated, use `zdtinput!`", :input!)
    zdtinput!(args...; kwargs...)
end

"""
    output!(dest, workspace)

Deprecated alias of [`zdtoutput!`](@ref).
"""
function output!(args...; kwargs...)
    Base.depwarn("`output!` is deprecated, use `zdtoutput!`", :output!)
    zdtoutput!(args...; kwargs...)
end

"""
    preprocess!(workspace[, r0])

Deprecated alias of [`zdtpreprocess!`](@ref).
"""
function preprocess!(args...; kwargs...)
    Base.depwarn("`preprocess!` is deprecated, use `zdtpreprocess!`",
                 :preprocess!)
    zdtpreprocess!(args...; kwargs...)
end

"""
    convolve!(workspace)

Deprecated alias of [`zdtconvolve!`](@ref).
"""
function convolve!(args...; kwargs...)
    Base.depwarn("`convolve!` is deprecated, use `zdtconvolve!`", :convolve!)
    zdtconvolve!(args...; kwargs...)
end

"""
    postprocess!([w,] workspace)

Deprecated alias of [`zdtpostprocess!`](@ref).
"""
function postprocess!(args...; kwargs...)
    Base.depwarn("`postprocess!` is deprecated, use `zdtpostprocess!`",
                 :postprocess!)
    zdtpostprocess!(args...; kwargs...)
end
