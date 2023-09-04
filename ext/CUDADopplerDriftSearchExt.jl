module CUDADopplerDriftSearchExt

import DopplerDriftSearch: plan_ffts!, ZDTWorkspace

if isdefined(Base, :get_extension)
    using CUDA: CuArray
    using CUDA.CUFFT: plan_fft!, plan_bfft!, plan_rfft, plan_brfft
    import AbstractFFTs: plan_brfft
    # Import CUDA functions for optimizing workarea usage
    import CUDA: unsafe_free!
    import CUDA.CUFFT: cufftSetWorkArea
else
    import ..CUDA: CuArray
    using ..CUDA.CUFFT: plan_fft!, plan_bfft!, plan_rfft, plan_brfft
    import ..AbstractFFTs: plan_brfft
    # Import CUDA functions for optimizing workarea usage
    import ..CUDA: unsafe_free!
    import ..CUDA.CUFFT: cufftSetWorkArea
end

# Type piracy to workaround CUDA.jl issue #1559.  For more details, see:
# https://github.com/JuliaGPU/CUDA.jl/issues/1559
plan_brfft(A::CuArray, d::Integer, region; kwargs...) = plan_brfft(A, d, region)

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

end # module CUDADopplerDriftSearchExt
