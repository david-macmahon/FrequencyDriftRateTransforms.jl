module CUDADopplerDriftSearchExt

import DopplerDriftSearch: plan_ffts!, ZDTWorkspace, output!

if isdefined(Base, :get_extension)
    import FFTW
    using CUDA: CuArray, context, stream
    using CUDA.CUFFT: plan_fft!, plan_bfft!, plan_rfft, plan_brfft
    # Import CUDA functions for optimizing workarea usage
    import CUDA.CUFFT: cufftGetSize1d, cufftSetWorkArea,
                       update_stream, cufftExecC2R,
                       CUFFT_C2C, CUFFT_C2R, CUFFT_R2C,
                       cufftHandle, cufftCreate, cufftSetAutoAllocation,
                       cufftSetStream, cufftMakePlanMany,
                       cCuFFTPlan, rCuFFTPlan, cufftReal, cufftComplex,
                       CUFFT_FORWARD, CUFFT_INVERSE, idle_handles
else
    import ..FFTW
    import ..CUDA: CuArray, context, stream
    using ..CUDA.CUFFT: plan_fft!, plan_bfft!, plan_rfft, plan_brfft
    # Import CUDA functions for optimizing workarea usage
    import ..CUDA.CUFFT: cufftGetSize1d, cufftSetWorkArea,
                         update_stream, cufftExecC2R,
                         CUFFT_C2C, CUFFT_C2R, CUFFT_R2C,
                         cufftHandle, cufftCreate, cufftSetAutoAllocation,
                         cufftSetStream, cufftMakePlanMany,
                         cCuFFTPlan, rCuFFTPlan, cufftReal, cufftComplex,
                         CUFFT_FORWARD, CUFFT_INVERSE, idle_handles
end

# This is a modified version of CUDA.CUFFT.cufftGetPlan
function getPlanHandle(args...)
    ctx = context()
    handle = pop!(idle_handles, (ctx, args...)) do
        # make the plan handle
        handle_ref = Ref{cufftHandle}()
        cufftCreate(handle_ref)

        # Set auto-allocation to false
        cufftSetAutoAllocation(handle_ref[], 0)

        handle_ref[]
    end

    # assign to the current stream
    cufftSetStream(handle, stream())

    return handle
end

"""
Very specific function to make FFT plans for `workspace` with "auto allocation"
disabled.  This means that we need to allocate and assign the work area
ourselves.  We do this to share one workarea across all of the FFT plans.  The
old approach would create a plan with an auto-allocated workarea and then
replace the plan's workare with a shared workarea, which would free the
auto-allocated work area, but this auto-allocation had the potential to exceed
the memory capacity of the GPU, which throws an error even though the final
memory requirement (after freeing the auto-allocated workarea) would be within
the GPU's capacity.
"""
function plan_ffts!(workspace::ZDTWorkspace,
                    spectrogram::CuArray{<:Real};
                    output_aligned::Bool=false)
    Nf = workspace.Nf
    Nt = workspace.Nt
    Nr = workspace.Nr
    Y = workspace.Y

    # initialize the plans' handles
    Y_handle = getPlanHandle(CUFFT_C2C, size(Y), (ZDTWorkspace, CUFFT_FORWARD))
    Yb_handle = getPlanHandle(CUFFT_C2C, size(Y), (ZDTWorkspace, CUFFT_INVERSE))
    in_handle = getPlanHandle(CUFFT_R2C, (Nf, Nt), ZDTWorkspace)
    out_handle = getPlanHandle(CUFFT_C2R, (Nf, Nr), ZDTWorkspace)

    # Set auto-allocation to false
    cufftSetAutoAllocation(Y_handle, 0)
    cufftSetAutoAllocation(Yb_handle, 0)
    cufftSetAutoAllocation(in_handle, 0)
    cufftSetAutoAllocation(out_handle, 0)

    # Setup for cufftMakePlanMany for Y and Yb
    nrank = 1
    n = Cint[size(Y, 2)]

    inembed = Cint[size(Y, 2)]
    istride = size(Y, 1)
    idist = 1

    ostride = istride
    odist = idist
    onembed = inembed

    xtype = CUFFT_C2C
    batch = istride
    Y_worksize_ref = Ref{Csize_t}(0)

    # arguments are:  plan, rank, transform-sizes,
    #                 inembed, istride, idist, onembed, ostride, odist,
    #                 type, batch, worksize_ref
    cufftMakePlanMany(Y_handle, nrank, n,
                      inembed, istride, idist, onembed, ostride, odist,
                      xtype, batch, Y_worksize_ref)

    cufftMakePlanMany(Yb_handle, nrank, n,
                      inembed, istride, idist, onembed, ostride, odist,
                      xtype, batch, Y_worksize_ref)

    # Setup for cufftMakePlanMany for in_handle
    nrank = 1
    n = Cint[Nf]

    xtype = CUFFT_R2C
    batch = Nt
    in_worksize_ref = Ref{Csize_t}(0)

    cufftMakePlanMany(in_handle, nrank, n, C_NULL, 1, 1, C_NULL, 1, 1,
                      xtype, batch, in_worksize_ref)

    # Setup for cufftMakePlanMany for out_handle
    nrank = 1
    n = Cint[Nf]

    xtype = CUFFT_C2R
    batch = Nr
    out_worksize_ref = Ref{Csize_t}(0)

    cufftMakePlanMany(out_handle, nrank, n, C_NULL, 1, 1, C_NULL, 1, 1,
                      xtype, batch, out_worksize_ref)

    # Make workarea to cover the largest required
    worksize = max(Y_worksize_ref[], in_worksize_ref[], out_worksize_ref[])
    workspace.fft_workarea = CuArray{UInt8}(undef, worksize)

    # Share workarea to all plan handles
    cufftSetWorkArea(Y_handle, workspace.fft_workarea)
    cufftSetWorkArea(Yb_handle, workspace.fft_workarea)
    cufftSetWorkArea(in_handle, workspace.fft_workarea)
    cufftSetWorkArea(out_handle, workspace.fft_workarea)

    # Setup to create cCuFFTPlan instances for Y_handle
    inplace = true
    region = (2,)
    xtype = CUFFT_C2C

    workspace.fft_plan = cCuFFTPlan{cufftComplex,CUFFT_FORWARD,inplace,2}(
        Y_handle, Y, size(Y), region, xtype
    )
    workspace.bfft_plan = cCuFFTPlan{cufftComplex,CUFFT_INVERSE,inplace,2}(
        Yb_handle, Y, size(Y), region, xtype
    )

    # Setup to create rCuFFTPlan instances for in_handle
    inplace = false
    xdims = (Nf, Nt)
    ydims = (div(Nf, 2) + 1, Nt)
    region = (1,)
    xtype = CUFFT_R2C

    workspace.rfft_plan =
        rCuFFTPlan{eltype(spectrogram),CUFFT_FORWARD,inplace,2}(
            in_handle, xdims, ydims, region, xtype
        )

    # Setup to create rCuFFTPlan instances for in_handle
    xdims = (div(Nf, 2) + 1, Nr)
    ydims = (Nf, Nr)
    xtype = CUFFT_C2R

    workspace.brfft_plan =
        rCuFFTPlan{cufftComplex,CUFFT_INVERSE,inplace,2}(
            out_handle, xdims, ydims, region, xtype
        )

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
