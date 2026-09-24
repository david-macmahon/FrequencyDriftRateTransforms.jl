# CUDA method of `fast_quantile` and its supporting kernels.  The shared
# driver `_fast_quantile_impl` and the selection machinery live in
# `src/fastquantile.jl`.

import FrequencyDriftRateTransforms: fast_quantile, _fast_quantile_impl,
                                     _select_eltypes, _monokey, _task_bins,
                                     _SelectTask

"""
    fast_quantile(data::CuArray, ps)

CUDA method of `fast_quantile`: the required order statistics are selected
on the device by iterative histogram refinement (the same algorithm as the
host method in `src/fastquantile.jl`), so the data never leaves the GPU and
no sorted copy is materialized.  Each pass bins the sign-flipped IEEE bit
patterns of the values into 2048 shared-memory histogram bins per open task
and merges the block-local counts into a device histogram; the host drives
the task refinement on the few-KB histograms.  The results match
`Statistics.quantile` exactly.
"""
fast_quantile(data::CuArray{<:_select_eltypes}, ps) =
    _fast_quantile_impl(_cuda_histpass!, data, ps)

# One refinement pass on the device: histogram the keys of every open task
# (block-local shared-memory histograms merged into one global histogram
# per task) and download the small counts to the host.  NaNs are counted in
# a side flag and reported by the host when requested.
function _cuda_histpass!(data::CuArray, tasks::Vector{_SelectTask}, checknan::Bool)
    hists = Vector{Vector{Int}}(undef, length(tasks))
    nanflag = CuVector{Int32}(undef, 1)
    fill!(nanflag, Int32(0))
    threads = 256
    blocks = min(cld(length(data), threads), 1 << 15)
    for (t, task) in enumerate(tasks)
        nbins, shift = _task_bins(task)
        hist = CuVector{Int}(undef, nbins)
        fill!(hist, 0)
        @cuda threads = threads blocks = blocks _khist!(hist, nanflag, data,
                                                        task.klo, task.khi,
                                                        shift, nbins)
        hists[t] = Array(hist)
    end
    if checknan
        Array(nanflag)[1] > 0 && throw(ArgumentError(
            "quantiles are undefined in presence of NaNs or missing values"))
    end
    return hists
end

function _khist!(hist::CuDeviceVector{Int}, nanflag::CuDeviceVector{Int32},
                 data::CuDeviceArray{<:_select_eltypes},
                 klo::UInt64, khi::UInt64, shift::Int, nbins::Int)
    shmem = CuStaticSharedArray(Int32, 2048)
    tid = Int64(threadIdx().x)
    bsz = Int64(blockDim().x)
    n = Int64(length(data))
    i = tid
    while i ≤ nbins
        shmem[i] = Int32(0)
        i += bsz
    end
    sync_threads()
    stride = bsz * Int64(gridDim().x)
    i = (Int64(blockIdx().x) - 1) * bsz + tid
    while i ≤ n
        @inbounds x = data[i]
        if x != x
            @atomic nanflag[1] += Int32(1)
        else
            key = _monokey(x)
            if klo ≤ key ≤ khi
                @atomic shmem[1 + Int((key - klo) >>> shift)] += Int32(1)
            end
        end
        i += stride
    end
    sync_threads()
    i = tid
    while i ≤ nbins
        v = shmem[i]
        if v != 0
            @atomic hist[i] += Int64(v)
        end
        i += bsz
    end
    return nothing
end
