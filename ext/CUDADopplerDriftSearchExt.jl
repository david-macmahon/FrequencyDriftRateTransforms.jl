module CUDADopplerDriftSearchExt

import DopplerDriftSearch: plan_ffts!, ZDTWorkspace, output!

if isdefined(Base, :get_extension)
    import FFTW
    using CUDA: CuArray
    using CUDA.CUFFT: plan_fft!, plan_bfft!, plan_rfft, plan_brfft
    # Import CUDA functions for optimizing workarea usage
    import CUDA.CUFFT: cufftGetSize1d, cufftSetWorkArea,
                       update_stream, cufftExecC2R,
                       CUFFT_C2C, CUFFT_C2R, CUFFT_R2C
else
    import ..FFTW
    import ..CUDA: CuArray
    using ..CUDA.CUFFT: plan_fft!, plan_bfft!, plan_rfft, plan_brfft
    # Import CUDA functions for optimizing workarea usage
    import ..CUDA.CUFFT: cufftGetSize1d, cufftSetWorkArea,
                         update_stream, cufftExecC2R,
                         CUFFT_C2C, CUFFT_C2R, CUFFT_R2C
end

"""
Make the ZDT's FFT plans for `spectrogram.CuArray`.  CUFFT requires workareas
for FFT plans, which can be as large as the inputs.  The complex-to-complex FFTs
(both forwards and backwards) used in the ZDT require a larger workspace than
the somewhat smaller real-to-complex forwards FFT and the complex-to-real
backwards FFT.  By creating the forward complex-to-complex FFT plan first, we
can then replace the workareas of the other FFT plans, as they are created, with
the workarea of the first created plan.  This allows all FFT plans to share one
common workarea.

Once this package no longer depends on CUDA.jl we might consider switching to
using `Requires.jl` so that this method only gets defined if the user has chosen
to use CUDA (rather than this package depending directly on CUDA).
"""
function plan_ffts!(workspace::ZDTWorkspace,
                    spectrogram::CuArray{<:Real};
                    output_aligned::Bool=false)
    Nf = workspace.Nf
    Y = workspace.Y
    Ys = workspace.Ys
    workareasize = Ref{Csize_t}(0)

    # Plan biggest FFT first
    workspace.fft_plan = plan_fft!(Y, 2)

    # Get size of plan's workarea
    cufftGetSize1d(workspace.fft_plan, size(Y, 2), CUFFT_C2C,
                   size(Y, 1), workareasize)
    # Allocate workarea on GPU
    workspace.fft_workarea = CuArray{UInt8}(undef, workareasize[])
    # Set plan's work area
    cufftSetWorkArea(workspace.fft_plan, workspace.fft_workarea)

    workspace.bfft_plan = plan_bfft!(Y, 2)
    cufftSetWorkArea(workspace.bfft_plan, workspace.fft_workarea)

    workspace.rfft_plan = plan_rfft(spectrogram, 1)
    # Get size of plan's workarea
    cufftGetSize1d(workspace.rfft_plan, size(spectrogram, 1), CUFFT_R2C,
                   size(spectrogram, 2), workareasize)
    #@info "got work area size $(Base.format_bytes(workareasize[])) for rfft of $(size(spectrogram))"
    if workareasize[] <= sizeof(workspace.fft_workarea)
        #@info "replacing workarea for rfft_plan"
        cufftSetWorkArea(workspace.rfft_plan, workspace.fft_workarea)
    end

    workspace.brfft_plan = plan_brfft(Ys, Nf, 1)
    # Get size of plan's workarea
    cufftGetSize1d(workspace.brfft_plan, Nf, CUFFT_C2R,
                   size(Ys, 2), workareasize)
    #@info "got work area size $(Base.format_bytes(workareasize[])) for brfft of ($Nf, $(size(Ys, 2)))"
    if workareasize[] <= sizeof(workspace.fft_workarea)
        #@info "replacing workarea for brfft_plan"
        cufftSetWorkArea(workspace.brfft_plan, workspace.fft_workarea)
    end

    return nothing
end

"""
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
