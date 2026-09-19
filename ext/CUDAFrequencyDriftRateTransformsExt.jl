module CUDAFrequencyDriftRateTransformsExt

import FrequencyDriftRateTransforms: plan_ffts!, ZDTWorkspace, output!,
                                     fdrsynchronize, taylortree!

if isdefined(Base, :get_extension)
    import FFTW
    using CUDA: CuArray, CuMatrix, CuDeviceMatrix, synchronize, @cuda,
                blockIdx, threadIdx, blockDim
    using CUDA.CUFFT: plan_fft!, plan_ifft!, plan_rfft, plan_irfft
    # Import CUDA functions for optimizing workarea usage
    import CUDA.CUFFT: cufftGetSize, cufftSetWorkArea,
                       update_stream, cufftExecC2R
else
    import ..FFTW
    import ..CUDA: CuArray, CuMatrix, CuDeviceMatrix, synchronize, @cuda,
                   blockIdx, threadIdx, blockDim
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
    output!(dest::CuMatrix{<:Real}, workspace) -> dest

Output ZDT results into `dest`, which should have size `(Nf, Nr)`.  This method
exists to avoid an allocating hack that CUDA.jl employs to work around a CUFFT
"known issue" that "cuFFT will always overwrite the input for out-of-place C2R
transform".  In our case, we don't care whether the input, `workspace.Ys`, gets
clobbered, but it does mean that `output!` cannot be called more than once per
ZDT operation.
"""
function output!(dest::CuMatrix{<:Real}, workspace)
    # Backwards FFT `workspace.Ys` into `dest`
    # workspace.irfft_plan is an AbstractFFTs.ScaledPlan that wraps a CuFFTPlan.
    # CUDA 5.4.2 and earlier do not properly convert the ScaledPlan to a
    # cufftHandle, so we operate on the contained CuFFTPlan directly.
    update_stream(workspace.irfft_plan.p)
    cufftExecC2R(workspace.irfft_plan.p, workspace.Ys, dest)
    dest .*= workspace.irfft_plan.scale
    return dest
end

# Taylor tree kernels.  Unlike the CPU implementation in src/taylorfdr.jl,
# which uses virtual zero padding (never storing or computing on the
# padding), the GPU implementation materializes the zero padding of the
# spectrogram to Ntp = nextpow(2, Nt) time samples so that the kernels can
# assume power-of-2 sizes with no special casing for padding (as in the
# seticore reference implementation).  The results are identical to the
# CPU implementation.

"""
    taylortree!(buffer1, buffer2, spectrogram::CuMatrix, drift_block) -> result

CUDA implementation of `taylortree!` that materializes the zero padding of
`spectrogram` to `Ntp = nextpow(2, Nt)` time samples and uses a CUDA kernel
(one thread per output entry) for each Taylor tree step.
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
    # Materialize the zero padding in buffer1
    copyto!(view(buffer1, :, 1:Nt), spectrogram)
    Nt < Ntp && fill!(view(buffer1, :, Nt+1:Ntp), zero(eltype(buffer1)))
    # Run all rounds of the tree: buffer1 -> buffer2 -> buffer1 -> ...
    source_buffer, target_buffer = buffer1, buffer2
    L = 2
    threads = min(Nf, 256)
    blockrows = cld(Nf, threads)
    while L <= Ntp
        @cuda blocks=(Ntp, blockrows) threads=threads taylor_round_kernel!(
            target_buffer, source_buffer, Nf, Ntp, L, drift_block)
        source_buffer, target_buffer = target_buffer, source_buffer
        L *= 2
    end
    # Zero the entries of the final result for paths that extend beyond the
    # frequency band; they were never written by the rounds above.
    @cuda blocks=cld(Ntp, 256) threads=256 taylor_zero_wedge_kernel!(
        source_buffer, Nf, Ntp, drift_block)
    return source_buffer
end

# One round of the Taylor tree; one thread per output entry of the round
# (a power-of-2 padded variant of taylorstep! in src/taylorfdr.jl).
# blockIdx.x selects the target column, threadIdx.x selects the frequency
# channel so that consecutive threads access consecutive memory addresses.
function taylor_round_kernel!(target::CuDeviceMatrix{T}, source::CuDeviceMatrix{T},
                              Nf::Int, Ntp::Int, L::Int, drift_block::Int) where T
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
    end
    return
end

# Zero the out-of-band wedge of a taylortree! result; one thread per column.
function taylor_zero_wedge_kernel!(buffer::CuDeviceMatrix{T},
                                   Nf::Int, Ntp::Int, drift_block::Int) where T
    col = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    col > Ntp && return
    drift_channels = (Ntp - 1) * drift_block + col - 1
    chan_lo = max(1, 1 - drift_channels)
    chan_hi = min(Nf, Nf - drift_channels)
    for chan in 1:(chan_lo - 1)
        @inbounds buffer[chan, col] = zero(T)
    end
    for chan in (chan_hi + 1):Nf
        @inbounds buffer[chan, col] = zero(T)
    end
    return
end

end # module CUDAFrequencyDriftRateTransformsExt
