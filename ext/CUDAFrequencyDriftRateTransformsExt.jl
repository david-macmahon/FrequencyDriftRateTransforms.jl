module CUDAFrequencyDriftRateTransformsExt

import FrequencyDriftRateTransforms: plan_ffts!, ZDTWorkspace, zdtoutput!,
                                     fdrsynchronize, taylortree!

if isdefined(Base, :get_extension)
    import FFTW
    using CUDA: CuArray, CuMatrix, CuDeviceMatrix, CuStaticSharedArray,
                synchronize, @cuda, blockIdx, threadIdx, blockDim,
                sync_threads
    using CUDA.CUFFT: plan_fft!, plan_ifft!, plan_rfft, plan_irfft
    # Import CUDA functions for optimizing workarea usage
    import CUDA.CUFFT: cufftGetSize, cufftSetWorkArea,
                       update_stream, cufftExecC2R
else
    import ..FFTW
    import ..CUDA: CuArray, CuMatrix, CuDeviceMatrix, CuStaticSharedArray,
                   synchronize, @cuda, blockIdx, threadIdx, blockDim,
                   sync_threads
    using ..CUDA.CUFFT: plan_fft!, plan_ifft!, plan_rfft, plan_irfft
    # Import CUDA functions for optimizing workarea usage
    import ..CUDA.CUFFT: cufftGetSize, cufftSetWorkArea,
                         update_stream, cufftExecC2R
end

"""
    fdrsynchronize(::Type{<:CuArray})

CUDA-specific implementation of this function that calls CUDA's `synchronize()`.
"""
function fdrsynchronize(::Type{<:CuArray})
    synchronize()
end

"""
    plan_ffts!(workspace::ZDTWorkspace, spectrogram::CuMatrix{<:Real};
               output_aligned::Bool=false)

Make the ZDT's FFT plans for `spectrogram::CuMatrix`.  CUFFT requires work areas
for FFT plans, which can be as large as the inputs.  The complex-to-complex FFT
plans' work areas used in the ZDT tend to be large and therefore require large
work areas.  After creating the forward complex-to-complex FFT plan first, we
can then create our own work area, and replace the work area of the just-created
FFT plan (which will free the auto-allocated work area).  Then we can also
replace the auto-allocated work area of the second complex-to-complex FFT plan,
which will also free that plan's auto-allocated work area.  While this saves
memory, it also imposes the constraint that the two plans must not be used
concurrently.  Since these two plans are only used in the `convolve!` function,
there is no danger of them being used concurrently on different streams (i.e.
Tasks).  The real-to-complex and complex-to-real FFT plans require smaller work
areas so there is not so much savings to be had by trying to share work areas
there and there is flexibility in not being constrained by a shared work area,
so each of those plans gets its own work area.
"""
function plan_ffts!(workspace::ZDTWorkspace,
                    spectrogram::CuMatrix{<:Real};
                    output_aligned::Bool=false)
    Nf = workspace.Nf
    Y = workspace.Y
    Ys = workspace.Ys
    workareasize = Ref{Csize_t}(0)

    # Plan one of biggest FFTs first: Forward FFT of Y
    workspace.fft_plan = plan_fft!(Y, 2)

    # Get size of plan's workarea
    cufftGetSize(workspace.fft_plan.handle, workareasize)
    # Allocate workarea on GPU
    workspace.fft_workarea = CuArray{UInt8}(undef, workareasize[])
    # Set plan's work area
    cufftSetWorkArea(workspace.fft_plan, workspace.fft_workarea)

    # Backward FFT of Y
    workspace.ifft_plan = plan_ifft!(Y, 2)
    # workspace.ifft_plan is an AbstractFFTs.ScaledPlan that wraps a CuFFTPlan.
    # CUDA 5.4.2 and earlier do not properly convert the ScaledPlan to a
    # cufftHandle, so we set the workarea on the contained CuFFTPlan directly.
    cufftSetWorkArea(workspace.ifft_plan.p, workspace.fft_workarea)

    # Forward real FFT for input to F
    workspace.rfft_plan = plan_rfft(spectrogram, 1)

    # Backward real FFT for output from Ys
    workspace.irfft_plan = plan_irfft(Ys, Nf, 1)

    return nothing
end

"""
    zdtoutput!(dest::CuMatrix{<:Real}, workspace) -> dest

Output ZDT results into `dest`, which should have size `(Nf, Nr)`.  This method
exists to avoid an allocating hack that CUDA.jl employs to work around a CUFFT
"known issue" that "cuFFT will always overwrite the input for out-of-place C2R
transform".  In our case, we don't care whether the input, `workspace.Ys`, gets
clobbered, but it does mean that `zdtoutput!` cannot be called more than once per
ZDT operation.
"""
function zdtoutput!(dest::CuMatrix{<:Real}, workspace)
    # Backwards FFT `workspace.Ys` into `dest`
    # workspace.irfft_plan is an AbstractFFTs.ScaledPlan that wraps a CuFFTPlan.
    # CUDA 5.4.2 and earlier do not properly convert the ScaledPlan to a
    # cufftHandle, so we operate on the contained CuFFTPlan directly.
    update_stream(workspace.irfft_plan.p)
    cufftExecC2R(workspace.irfft_plan.p, workspace.Ys, dest)
    dest .*= workspace.irfft_plan.scale
    return dest
end

# Taylor tree kernels.  The tiled kernel (following the tiled Taylor tree
# in seticore's taylor.cu) stages 32 timesteps x 128 channels of the
# spectrogram in shared memory and runs the path length 2..32 rounds there;
# each block computes 96 output channels (tiles overlap by 32 channels so
# that paths of 32 steps stay within the tile), with TILE_THREADS threads
# per channel splitting the per-channel work.  The drift block's integer
# drift is applied when loading the tile (timestep t is read drift_block*t
# channels up), so the in-tile rounds run with zero drift, and entries that
# fall outside the spectrogram (in frequency, or in time, i.e. the zero
# padding) are loaded as zeros.  For Ntp >= 64 the per-entry kernel handles
# the remaining rounds in global memory; for smaller Ntp the tiled kernel
# runs the whole tree.  The final round of each tree zero-fills the entries
# for paths that extend beyond the frequency band (earlier rounds skip
# them, since they are never read), so no separate wedge-zeroing pass is
# needed.  The results are identical to the CPU implementation in
# src/taylorfdr.jl.

const TILE_WIDTH = 128        # shared channels per tile
const TILE_BLOCK_WIDTH = 96   # output channels per block
const TILE_TIMESTEPS = 32     # shared timesteps per tile
const TILE_THREADS = 2        # threadIdx.y threads splitting per-channel work

"""
    taylortree!(buffer1, buffer2, spectrogram::CuMatrix, drift_block) -> result

CUDA implementation of `taylortree!` using a tiled shared-memory kernel
(following seticore's tiled Taylor tree) for the path length 2..32 rounds
and a per-entry kernel for any remaining rounds.
"""
function taylortree!(buffer1::CuMatrix, buffer2::CuMatrix,
                     spectrogram::CuMatrix{<:Real}, drift_block::Integer)
    Nf, Nt = size(spectrogram)
    Ntp = nextpow(2, Nt)
    drift_block = Int(drift_block)
    size(buffer1) == (Nf, Ntp) ||
        throw(ArgumentError("buffer1 must have size ($Nf, $Ntp) (got $(size(buffer1)))"))
    size(buffer2) == (Nf, Ntp) ||
        throw(ArgumentError("buffer2 must have size ($Nf, $Ntp) (got $(size(buffer2)))"))
    buffer1 === buffer2 &&
        throw(ArgumentError("buffer1 and buffer2 must be distinct"))
    (spectrogram === buffer1 || spectrogram === buffer2) &&
        throw(ArgumentError("spectrogram must be distinct from buffer1 and buffer2"))
    Nt >= 2 || throw(ArgumentError("number of time samples ($Nt) must be at least 2"))
    if Ntp >= 2 * TILE_TIMESTEPS
        # Stage 1: tiled kernel computes the path length 2..32 rounds,
        # writing the results to buffer1
        blocks = (cld(Nf, TILE_BLOCK_WIDTH), Ntp ÷ TILE_TIMESTEPS)
        @cuda blocks=blocks threads=(TILE_WIDTH, TILE_THREADS) taylor_tiled_kernel!(
            buffer1, spectrogram, Nf, Nt, Ntp, drift_block, false)
        # Stage 2: per-entry rounds for path lengths 64..Ntp:
        # buffer1 -> buffer2 -> buffer1 -> ...
        source_buffer, target_buffer = buffer1, buffer2
        L = 2 * TILE_TIMESTEPS
        threads = min(Nf, 256)
        blockrows = cld(Nf, threads)
        while L <= Ntp
            # The final round zero-fills the entries for paths that extend
            # beyond the frequency band (earlier rounds skip them; they are
            # never read, and the final round fully overwrites its target)
            @cuda blocks=(Ntp, blockrows) threads=threads taylor_round_kernel!(
                target_buffer, source_buffer, Nf, Ntp, L, drift_block, L == Ntp)
            source_buffer, target_buffer = target_buffer, source_buffer
            L *= 2
        end
    else
        # The tiled kernel runs the whole tree (path lengths 2..Ntp)
        @cuda blocks=(cld(Nf, TILE_BLOCK_WIDTH), 1) threads=(TILE_WIDTH, TILE_THREADS) taylor_tiled_kernel!(
            buffer1, spectrogram, Nf, Nt, Ntp, drift_block, true)
        source_buffer = buffer1
    end
    return source_buffer
end

# One in-tile Taylor tree step; thread (s, ty) of TILE_WIDTH x TILE_THREADS
# threads computes the entries for shared-memory channel `s` and the time
# blocks ty-1, ty-1+TILE_THREADS, ... (a zero-drift variant of taylorstep!
# in src/taylorfdr.jl).  Entries whose reads would fall outside the tile are
# not written; as in the parent implementation, such entries are never read
# by subsequent steps.
function taylor_shared_step!(dst::CuDeviceMatrix{T}, src::CuDeviceMatrix{T},
                             s::Int, ty::Int, nty::Int, L::Int) where T
    half = L ÷ 2
    for tb in (ty - 1):nty:(TILE_TIMESTEPS ÷ L - 1), p in 0:(L - 1)
        half_offset = p >> 1
        shift = (p + 1) >> 1
        if s + shift <= TILE_WIDTH && s + p <= TILE_WIDTH
            @inbounds dst[s, tb*L + p + 1] =
                src[s, tb*L + half_offset + 1] +
                src[s + shift, tb*L + half_offset + half + 1]
        end
    end
    return
end

# Tiled kernel: stages a TILE_TIMESTEPS x TILE_WIDTH tile of the (drift
# shifted) spectrogram in shared memory and runs the path length 2..Lmax
# rounds there, where Lmax = min(Ntp, TILE_TIMESTEPS).  Each block computes
# the TILE_BLOCK_WIDTH output channels starting at
# (blockIdx().x - 1) * TILE_BLOCK_WIDTH; the final round writes global
# memory directly.  For Ntp >= 2 * TILE_TIMESTEPS the output holds the
# path length TILE_TIMESTEPS sums for the remaining rounds (one tile time
# block per blockIdx().y); for smaller Ntp it holds the complete result,
# with `zerofill` writing zeros for paths that extend beyond the band.
function taylor_tiled_kernel!(output::CuDeviceMatrix{T}, input::CuDeviceMatrix{T},
                              Nf::Int, Nt::Int, Ntp::Int, drift_block::Int,
                              zerofill::Bool) where T
    sh1 = CuStaticSharedArray(T, (TILE_WIDTH, TILE_TIMESTEPS))
    sh2 = CuStaticSharedArray(T, (TILE_WIDTH, TILE_TIMESTEPS))
    block_start = (blockIdx().x - 1) * TILE_BLOCK_WIDTH
    time_offset = (blockIdx().y - 1) * TILE_TIMESTEPS
    s = Int(threadIdx().x)
    ty = Int(threadIdx().y)
    nty = Int(blockDim().y)
    # Load the drift-shifted tile into shared memory, zeroing entries that
    # fall outside the spectrogram (in frequency or in time)
    for t in (ty - 1):nty:(TILE_TIMESTEPS - 1)
        g0 = block_start + (s - 1) + drift_block * t
        if time_offset + t < Nt && 0 <= g0 < Nf
            @inbounds sh1[s, t + 1] = input[g0 + 1, time_offset + t + 1]
        else
            @inbounds sh1[s, t + 1] = zero(T)
        end
    end
    sync_threads()
    # In-tile rounds (zero drift; the drift block was applied at load time)
    src, dst = sh1, sh2
    Lmax = min(Ntp, TILE_TIMESTEPS)
    L = 2
    while L < Lmax
        taylor_shared_step!(dst, src, s, ty, nty, L)
        src, dst = dst, src
        L *= 2
        sync_threads()
    end
    # Final round reads shared memory and writes global memory.  Entries
    # for the last TILE_WIDTH - TILE_BLOCK_WIDTH shared channels duplicate
    # the next block's output (identical values, benign writes).  With
    # `zerofill`, entries for paths extending beyond the band (endpoint
    # gchan + drift_block*(Lmax-1) + p outside 1:Nf) are written as zeros.
    gchan = block_start + s  # 1-based global channel for this thread
    if gchan <= Nf
        half = Lmax ÷ 2
        for p in 0:(Lmax - 1)
            half_offset = p >> 1
            shift = (p + 1) >> 1
            if s + shift <= TILE_WIDTH && s + p <= TILE_WIDTH
                inband = 1 <= gchan + drift_block * (Lmax - 1) + p <= Nf
                if inband || zerofill
                    @inbounds output[gchan, time_offset + p + 1] =
                        inband ? src[s, half_offset + 1] +
                                 src[s + shift, half_offset + half + 1] : zero(T)
                end
            end
        end
    end
    return
end

# One round of the Taylor tree; one thread per output entry of the round
# (a power-of-2 padded variant of taylorstep! in src/taylorfdr.jl).
# blockIdx.x selects the target column, threadIdx.x selects the frequency
# channel so that consecutive threads access consecutive memory addresses.
# With `zerofill`, entries for paths that extend beyond the frequency band
# are written as zeros instead of being skipped (used for the final round,
# which fully overwrites its target, so no wedge-zeroing pass is needed).
function taylor_round_kernel!(target::CuDeviceMatrix{T}, source::CuDeviceMatrix{T},
                              Nf::Int, Ntp::Int, L::Int, drift_block::Int,
                              zerofill::Bool) where T
    tcol = blockIdx().x
    chan = (blockIdx().y - 1) * blockDim().x + threadIdx().x
    chan > Nf && return
    path_offset = rem(tcol - 1, L)
    tb = (tcol - 1) ÷ L
    half = L ÷ 2
    half_offset = path_offset >> 1
    shift = ((path_offset + 1) >> 1) + drift_block * half
    drift_channels = path_offset + drift_block * (L - 1)
    # Only write entries for paths that stay within the frequency band
    if 1 <= chan + shift <= Nf && 1 <= chan + drift_channels <= Nf
        @inbounds target[chan, tcol] =
            source[chan, tb*L + half_offset + 1] +
            source[chan + shift, tb*L + half_offset + half + 1]
    elseif zerofill
        @inbounds target[chan, tcol] = zero(T)
    end
    return
end

end # module CUDAFrequencyDriftRateTransformsExt
