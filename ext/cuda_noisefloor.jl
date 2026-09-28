# Batched robust per-band noise statistics on CUDA: `_noisefloor_banded`
# replaces the host fallback's per-band buffer loop (each band its own
# `fast_quantile` and clipped-mean refinement, each with its own kernel
# launches and round trips) with a device computation whose cost is
# independent of the number of bands: at most a handful of batched histogram
# passes over the whole matrix (all bands' selection tasks proceed in
# lockstep), then per-band host estimation from the downloaded quantiles,
# with the optional clipped-mean refinement driven by batched count+sum
# passes with deterministic per-block partials.

import FrequencyDriftRateTransforms: _noisefloor_banded, _noise_moments,
                                     _noise_refine_step, _quantile_ranks,
                                     _quantile_interp, _select_eltypes,
                                     _SelectTask, _refine!, _keybounds

"""
    _noisefloor_banded(fdr::CuMatrix, cpb; k, qlo, clip, refine)
        -> (mean = ..., std = ...)

CUDA method of `_noisefloor_banded`: robust per-band statistics for the
`cpb`-sized bands of rows of `fdr`, matching the host fallback's estimates
(the quantile selection is bit-exact; only the clipped-mean refinement sums
differ, in reduction order).  One open `_SelectTask` per band holds the
ranks of both required quantiles; the tasks proceed through the same
histogram-refinement loop as `_select_ranks`, batched across bands by
`_cuda_banded_histpass!`.

The per-band rank arithmetic uses an offset trick so that the shared
`_refine!` needs no modification: band `b`'s ranks and `below` count are
offset by `(b - 1) * n_b` (with `n_b = cpb * Nr` the band size).  The
offset cancels in every comparison inside `_refine!` (which only involves
`below`, the per-band histogram counts, and the ranks), the resolved rank
keys become globally unique per (band, rank) pair, and the band index is
recoverable from a task as `below ÷ n_b`.
"""
function _noisefloor_banded(fdr::CuMatrix{T}, cpb::Int; k = nothing, qlo = 0.1,
                            clip = 4.0, refine = true) where {T <: _select_eltypes}
    k !== nothing && k <= 0 && throw(ArgumentError("k must be positive"))
    !(0 < qlo < 0.5) && throw(ArgumentError("qlo must be between 0 and 0.5"))
    Nf, Nr = size(fdr)
    nbands = Nf ÷ cpb
    n_b = cpb * Nr
    js, γs, ranks = _quantile_ranks(n_b, [qlo, 0.5])
    klo0, khi0 = _keybounds(T)
    # `_refine!` walks the ranks in ascending order, so deduplicate and sort
    # once (all bands share the same local ranks).
    rr = sort!(unique(ranks))
    tasks = [_SelectTask(klo0, khi0, off, rr .+ off)
             for off in ((b - 1) * n_b for b in 1:nbands)]
    hists = Vector{Vector{Int}}()
    resolved = Dict{Int, T}()
    checknan = true
    while !isempty(tasks)
        _cuda_banded_histpass!(fdr, tasks, checknan, hists, cpb)
        checknan = false
        tasks = _refine!(resolved, tasks, hists)
    end
    # Per-band quantiles: host-side interpolation identical to
    # `fast_quantile`'s.
    qlo_vals = Vector{Float64}(undef, nbands)
    q50s = Vector{Float64}(undef, nbands)
    for b in 1:nbands
        off = (b - 1) * n_b
        if n_b == 1
            v = Float64(resolved[off + 1])
            qlo_vals[b] = q50s[b] = v
        else
            qlo_vals[b] = Float64(_quantile_interp(resolved[off + js[1]],
                                                   resolved[off + js[1] + 1],
                                                   γs[1]))
            q50s[b] = Float64(_quantile_interp(resolved[off + js[2]],
                                               resolved[off + js[2] + 1],
                                               γs[2]))
        end
    end
    # Per-band moments and degenerate handling.  Degenerate bands (non-positive
    # median, or lower quantile at the median) fall back to the band's plain
    # mean and `Inf` sigma like `noisefloor`; the mean comes from one batched
    # full-range count+sum pass.  (The host fallback's `Float64(mean(buf))`
    # accumulates in the band's eltype; for the zeros and constants that
    # trigger this path both agree exactly.)
    means = Vector{Float64}(undef, nbands)
    stds = Vector{Float64}(undef, nbands)
    shapes = Vector{Float64}(undef, nbands)
    degenerate = Int[]
    for b in 1:nbands
        if !(q50s[b] > 0) || !(qlo_vals[b] < q50s[b])
            push!(degenerate, b)
            means[b] = NaN  # placeholder until the batched band mean below
            stds[b] = Inf
        else
            mom = _noise_moments(qlo_vals[b], q50s[b]; qlo)
            means[b] = mom.mean
            stds[b] = mom.std
            shapes[b] = mom.shape
        end
    end
    if !isempty(degenerate)
        bands = degenerate
        thresholds = fill(Inf, length(bands))
        counts, sums = _cuda_banded_countsum(fdr, cpb, bands, thresholds)
        for (i, b) in enumerate(bands)
            means[b] = sums[i] / counts[i]
        end
    end
    if refine
        # Batched iterated clipped mean: every round thresholds each active
        # band at `clip * mean`, counts and sums the survivors in one fused
        # device pass, and advances each band independently until it
        # converges (or yields no survivors), mirroring the host loop.
        active = [b for b in 1:nbands if !(b in degenerate)]
        for _ in 1:5
            isempty(active) && break
            thresholds = [clip * means[b] for b in active]
            counts, sums = _cuda_banded_countsum(fdr, cpb, active, thresholds)
            still = Int[]
            for (i, b) in enumerate(active)
                n = counts[i]
                n == 0 && continue
                mean_new, done = _noise_refine_step(means[b], sums[i] / n,
                                                    shapes[b], clip)
                means[b] = mean_new
                done || push!(still, b)
            end
            active = still
        end
    end
    mean_v = Vector{Float64}(undef, Nf)
    std_v = Vector{Float64}(undef, Nf)
    for b in 1:nbands
        r = ((b - 1) * cpb + 1):(b * cpb)
        mean_v[r] .= means[b]
        std_v[r] .= stds[b]
    end
    (mean = mean_v, std = std_v)
end

# Per-band count and Float64 sum of the elements below per-band value
# thresholds, in one fused pass over each band's rows.  The comparison is
# `Float64(x) < s` so that the threshold decisions match the host's
# `x < s` (which promotes to Float64) bit for bit; NaN compares false and is
# excluded, like `count(<(s), data)`.  Each block of the fixed
# `(blocks_per_band, length(bands))` grid accumulates its own partial into a
# distinct output slot (no atomics), and the host merges the partials in
# fixed block order, so repeated calls are bitwise reproducible.
function _cuda_banded_countsum(fdr::CuMatrix{<:_select_eltypes}, cpb::Int,
                               bands::Vector{Int}, thresholds::Vector{Float64};
                               blocks_per_band::Int = 64)
    n_b = cpb * size(fdr, 2)
    nt = length(bands)
    sdev = CuArray(thresholds)
    bdev = CuArray(bands)
    counts = CuMatrix{Int64}(undef, blocks_per_band, nt)
    sums = CuMatrix{Float64}(undef, blocks_per_band, nt)
    fill!(counts, Int64(0))
    fill!(sums, 0.0)
    threads = 256
    @cuda threads = threads blocks = (blocks_per_band, nt) _kbanded_countsum!(
        counts, sums, fdr, sdev, bdev, cpb, n_b)
    hc = Array(counts)
    hs = Array(sums)
    out_counts = Vector{Int64}(undef, nt)
    out_sums = Vector{Float64}(undef, nt)
    for i in 1:nt
        c = Int64(0)
        s = 0.0
        for blk in 1:blocks_per_band
            c += hc[blk, i]
            s += hs[blk, i]
        end
        out_counts[i] = c
        out_sums[i] = s
    end
    return out_counts, out_sums
end

function _kbanded_countsum!(counts::CuDeviceMatrix{Int64},
                            sums::CuDeviceMatrix{Float64},
                            fdr::CuDeviceMatrix{<:_select_eltypes},
                            thresholds::CuDeviceVector{Float64},
                            bands::CuDeviceVector{Int}, cpb::Int, n_b::Int)
    bi = Int(blockIdx().y)
    row0 = (bands[bi] - 1) * cpb + 1
    s = thresholds[bi]
    blk = Int(blockIdx().x)
    nblk = Int(gridDim().x)
    bsz = Int(blockDim().x)
    tid = Int(threadIdx().x)
    c = Int64(0)
    acc = 0.0
    i = Int64(tid) + (Int64(blk) - 1) * Int64(bsz)
    stride = Int64(bsz) * Int64(nblk)
    while i ≤ n_b
        r = (i - 1) % cpb + 1
        col = (i - 1) ÷ cpb + 1
        @inbounds x = fdr[row0 + r - 1, col]
        if Float64(x) < s
            c += Int64(1)
            acc += Float64(x)
        end
        i += stride
    end
    # Reduce the threads' partials through shared memory in a fixed tree and
    # let thread 1 write the block's slot (the launcher uses a power-of-two
    # `threads` of at most 256).
    sh_c = CuStaticSharedArray(Int64, 256)
    sh_s = CuStaticSharedArray(Float64, 256)
    sh_c[tid] = c
    sh_s[tid] = acc
    sync_threads()
    h = bsz ÷ 2
    while h > 0
        if tid <= h
            sh_c[tid] += sh_c[tid + h]
            sh_s[tid] += sh_s[tid + h]
        end
        sync_threads()
        h ÷= 2
    end
    if tid == 1
        counts[blk, bi] = sh_c[1]
        sums[blk, bi] = sh_s[1]
    end
    return nothing
end
