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
# per task) and download the small counts into the reusable host-side
# `hists` workspace.  NaNs are counted in a side flag and reported by the
# host when requested.  (Device histogram buffers are per pass; the CUDA
# memory pool absorbs their allocation.)
function _cuda_histpass!(data::CuArray, tasks::Vector{_SelectTask},
                         checknan::Bool, hists::Vector{Vector{Int}})
    while length(hists) < length(tasks)
        push!(hists, zeros(Int, 2048))
    end
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
        copyto!(hists[t], 1, hist, 1, nbins)
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

# One batched refinement pass for the per-band selection tasks of
# `_noisefloor_banded` (all bands proceed in lockstep, since every task
# starts from the same full-range key interval and the number of passes
# depends only on the eltype's key width): a single kernel launch histograms
# every open task's band of rows — `gridDim.y` = number of open tasks, each
# block histogramming into 2048 shared-memory bins and merging into its
# task's column of the global histogram matrix — and downloads all
# histograms in one transfer into the reusable host-side `hists` workspace.
# Each task's band of rows is recovered from its offset `below` count (see
# `_noisefloor_banded`).
function _cuda_banded_histpass!(fdr::CuMatrix{<:_select_eltypes},
                                tasks::Vector{_SelectTask}, checknan::Bool,
                                hists::Vector{Vector{Int}}, cpb::Int)
    while length(hists) < length(tasks)
        push!(hists, zeros(Int, 2048))
    end
    n_b = cpb * size(fdr, 2)
    bins = [_task_bins(task) for task in tasks]
    klos = CuArray(UInt64[task.klo for task in tasks])
    khis = CuArray(UInt64[task.khi for task in tasks])
    nbins = CuArray(Int[nb for (nb, _) in bins])
    shifts = CuArray(Int[sh for (_, sh) in bins])
    rows0 = CuArray(Int[(task.below ÷ n_b) * cpb + 1 for task in tasks])
    nanflag = CuVector{Int32}(undef, 1)
    fill!(nanflag, Int32(0))
    hist_dev = CuMatrix{Int}(undef, 2048, length(tasks))
    fill!(hist_dev, 0)
    threads = 256
    # Share a fixed total-block budget across the open tasks (like the scalar
    # kernel's per-task cap): more blocks would multiply the global histogram
    # merge atomics with the number of bands, while the grid-stride loop
    # simply gives each thread more elements.
    blocks_x = min(cld(n_b, threads), max(1, (1 << 15) ÷ length(tasks)))
    @cuda threads = threads blocks = (blocks_x, length(tasks)) _kbanded_hist!(
        hist_dev, nanflag, fdr, klos, khis, shifts, nbins, rows0, cpb, n_b)
    hdev = Array(hist_dev)
    for (t, (nb, _)) in enumerate(bins)
        copyto!(hists[t], 1, view(hdev, 1:nb, t), 1, nb)
    end
    if checknan
        Array(nanflag)[1] > 0 && throw(ArgumentError(
            "quantiles are undefined in presence of NaNs or missing values"))
    end
    return hists
end

# Batched variant of `_khist!`: `blockIdx().y` selects the open task, whose
# band of `cpb` matrix rows starting at `rows0[task]` (over all `Nr` columns,
# `n_b = cpb * Nr` elements) is histogrammed within the task's current key
# interval.
function _kbanded_hist!(hists::CuDeviceMatrix{Int},
                        nanflag::CuDeviceVector{Int32},
                        fdr::CuDeviceMatrix{<:_select_eltypes},
                        klos::CuDeviceVector{UInt64},
                        khis::CuDeviceVector{UInt64}, shifts::CuDeviceVector{Int},
                        nbins::CuDeviceVector{Int}, rows0::CuDeviceVector{Int},
                        cpb::Int, n_b::Int)
    shmem = CuStaticSharedArray(Int32, 2048)
    task = Int(blockIdx().y)
    klo = klos[task]
    khi = khis[task]
    shift = shifts[task]
    nb = nbins[task]
    row0 = rows0[task]
    tid = Int64(threadIdx().x)
    bsz = Int64(blockDim().x)
    i = tid
    while i ≤ nb
        shmem[i] = Int32(0)
        i += bsz
    end
    sync_threads()
    stride = bsz * Int64(gridDim().x)
    i = (Int64(blockIdx().x) - 1) * bsz + tid
    while i ≤ n_b
        # Linear index within the band, column-major over (cpb, Nr): the
        # decomposition keeps consecutive threads on consecutive addresses.
        r = (i - 1) % cpb + 1
        c = (i - 1) ÷ cpb + 1
        @inbounds x = fdr[row0 + r - 1, c]
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
    while i ≤ nb
        v = shmem[i]
        if v != 0
            @atomic hists[i, task] += Int64(v)
        end
        i += bsz
    end
    return nothing
end
