module CUDADopplerDriftSearchExt

import DopplerDriftSearch: plan_ffts!, ZDTWorkspace, output!

if isdefined(Base, :get_extension)
    import FFTW
    using CUDA: CuArray
    using CUDA.CUFFT: plan_fft!, plan_bfft!, plan_rfft, plan_brfft
    # Import CUDA functions for optimizing workarea usage
    import CUDA.CUFFT: cufftGetSize, cufftSetWorkArea,
                       update_stream, cufftExecC2R
else
    import ..FFTW
    import ..CUDA: CuArray
    using ..CUDA.CUFFT: plan_fft!, plan_bfft!, plan_rfft, plan_brfft
    # Import CUDA functions for optimizing workarea usage
    import ..CUDA.CUFFT: cufftGetSize, cufftSetWorkArea,
                         update_stream, cufftExecC2R
end

"""
    plan_ffts!(workspace::ZDTWorkspace, spectrogram::CuArray{<:Real};
               output_aligned::Bool=false)

Make the ZDT's FFT plans for `spectrogram::CuArray`.  CUFFT requires work areas
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
                    spectrogram::CuArray{<:Real};
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
    workspace.bfft_plan = plan_bfft!(Y, 2)
    cufftSetWorkArea(workspace.bfft_plan, workspace.fft_workarea)

    # Forward real FFT for input to F
    workspace.rfft_plan = plan_rfft(spectrogram, 1)

    # Backward real FFT for output from Ys
    workspace.brfft_plan = plan_brfft(Ys, Nf, 1)

    return nothing
end

"""
    output!(dest::CuArray{<:Real}, workspace) -> dest

Output ZDT results into `dest`, which should have size `(Nf, Nr)`.  This method
exists to avoid an allocating hack that CUDA.jl employs to work around a CUFFT
"known issue" that "cuFFT will always overwrite the input for out-of-place C2R
transform".  In our case, we don't care whether the input, `workspace.Ys`, gets
clobbered, but it does mean that `output!` cannot be called more than once per
ZDT operation.
"""
function output!(dest::CuArray{<:Real}, workspace)
    # Backwards FFT `workspace.Ys` into `dest`
    update_stream(workspace.brfft_plan)
    cufftExecC2R(workspace.brfft_plan, workspace.Ys, dest)
    return dest
end

end # module CUDADopplerDriftSearchExt
