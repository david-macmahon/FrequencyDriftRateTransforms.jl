# CUDA method of `findhits` and its supporting kernels.  The thresholding
# runs on the device via 32x32 tile maxima (so only tiles containing
# proto-hits are transferred) and the merge-tree clustering runs on the
# host (see `src/findhits.jl`).

import FrequencyDriftRateTransforms: findhits, _findhits_stats, _findhits_result,
                                     _findhits_empty, noisestats

const _findhits_tile = 32

"""
    findhits(fdr::CuMatrix, threshold::Real,
             stats = noisestats(fdr);
             min_prominence = nothing, dist = 2)

CUDA method of [`findhits`](@ref): a kernel computes the maximum of each
32x32 tile of `fdr` on the device along with the minimum per-channel
threshold over the tile's channels, only the tiles whose maximum reaches
that (conservative) tile threshold are transferred (such tiles provably
contain all proto-hits, since a tile's maximum upper-bounds every element
in it), and a second kernel gathers those tiles' elements into one compact
download.  The merge-tree clustering itself runs on the host, exactly as in
the CPU method.
"""
function findhits(fdr::CuMatrix, threshold::Real,
                  stats = noisestats(fdr);
                  min_prominence = nothing, dist = 2)
    m, s = _findhits_stats(stats)
    dist >= 1 || throw(ArgumentError("dist must be at least 1"))
    Nf, Nr = size(fdr)
    if m isa AbstractVector
        length(m) == Nf || throw(ArgumentError(
            "stats vector length ($(length(m))) must match the number of " *
            "frequency channels ($Nf)"))
        thr = m .+ threshold .* s  # per-channel (raw) thresholds
        minprom = min_prominence === nothing ? nothing : min_prominence .* s
    else
        thr = fill(threshold * s + m, Nf)
        minprom = min_prominence === nothing ? nothing : min_prominence * s
    end
    Ntx = cld(Nf, _findhits_tile)
    Nty = cld(Nr, _findhits_tile)
    tilemax = CuMatrix{eltype(fdr)}(undef, Ntx, Nty)
    tilethr = CuMatrix{eltype(thr)}(undef, Ntx, Nty)
    @cuda threads = 256 blocks = Ntx * Nty _tilemax!(tilemax, tilethr, fdr,
                                                    CuArray(thr), Nf, Nr, Ntx)
    candt = findall(Array(tilemax) .>= Array(tilethr))
    isempty(candt) && return _findhits_empty()
    candlin = [LinearIndices(tilemax)[c] for c in candt]
    cand = CuVector{Int}(candlin)
    out = CuMatrix{eltype(fdr)}(undef, _findhits_tile^2, length(candlin))
    @cuda threads = 256 blocks = length(candlin) _gather!(out, fdr, Nf, Nr, Ntx, cand)
    outvals = Array(out)
    protohijs = CartesianIndex{2}[]
    vals = Float64[]
    for (ct, c) in enumerate(candlin)
        t = (c - 1) % Ntx + 1
        u = (c - 1) ÷ Ntx + 1
        i0 = (t - 1) * _findhits_tile + 1
        i1 = min(i0 + _findhits_tile - 1, Nf)
        h = i1 - i0 + 1
        j0 = (u - 1) * _findhits_tile + 1
        j1 = min(j0 + _findhits_tile - 1, Nr)
        for k in 1:(h * (j1 - j0 + 1))
            row = i0 + (k - 1) % h
            v = outvals[k, ct]
            v >= thr[row] || continue
            push!(protohijs, CartesianIndex(row, j0 + (k - 1) ÷ h))
            push!(vals, Float64(v))
        end
    end
    return _findhits_result(vals, protohijs, Nf, Nr, m, s, minprom, dist)
end

# Maximum of each 32x32 tile of `fdr` plus the minimum per-channel threshold
# `thr` over the tile's channels (block per tile, shared-memory reductions);
# tiles are indexed column-major as (freq-tile, rate-tile).  Comparing the
# tile maximum against the tile's MINIMUM threshold is conservative for any
# per-channel threshold profile: a tile may only be skipped if no element in
# it can clear its own channel's threshold.
function _tilemax!(tilemax::CuDeviceMatrix{T}, tilethr::CuDeviceMatrix{S},
                   fdr::CuDeviceMatrix{T}, thr::CuDeviceVector{S},
                   Nf::Int, Nr::Int, Ntx::Int) where {T, S}
    bsz = Int(blockDim().x)
    tx = Int(threadIdx().x)
    b = Int(blockIdx().x)
    t = (b - 1) % Ntx + 1
    u = (b - 1) ÷ Ntx + 1
    i0 = (t - 1) * _findhits_tile + 1
    i1 = min(i0 + _findhits_tile - 1, Nf)
    j0 = (u - 1) * _findhits_tile + 1
    j1 = min(j0 + _findhits_tile - 1, Nr)
    h = i1 - i0 + 1
    n = h * (j1 - j0 + 1)
    sh = CuStaticSharedArray(T, 256)
    shthr = CuStaticSharedArray(S, 256)
    localmax = typemin(T)
    localthr = typemax(S)
    k = tx
    while k <= n
        @inbounds i = i0 + (k - 1) % h
        @inbounds x = fdr[i, j0 + (k - 1) ÷ h]
        x > localmax && (localmax = x)
        @inbounds ti = thr[i]
        ti < localthr && (localthr = ti)
        k += bsz
    end
    sh[tx] = localmax
    shthr[tx] = localthr
    sync_threads()
    s = bsz ÷ 2
    while s > 0
        if tx <= s
            sh[tx] = max(sh[tx], sh[tx + s])
            shthr[tx] = min(shthr[tx], shthr[tx + s])
        end
        sync_threads()
        s ÷= 2
    end
    tx == 1 && (tilemax[t, u] = sh[1])
    tx == 1 && (tilethr[t, u] = shthr[1])
    return nothing
end

# Gather the elements of each candidate tile (one block per tile, column
# `b` of `out`) so they can be downloaded in one transfer.
function _gather!(out::CuDeviceMatrix{T}, fdr::CuDeviceMatrix{T},
                  Nf::Int, Nr::Int, Ntx::Int, candt::CuDeviceVector{Int}) where T
    bsz = Int(blockDim().x)
    tx = Int(threadIdx().x)
    b = Int(blockIdx().x)
    @inbounds tile = candt[b]
    t = (tile - 1) % Ntx + 1
    u = (tile - 1) ÷ Ntx + 1
    i0 = (t - 1) * _findhits_tile + 1
    i1 = min(i0 + _findhits_tile - 1, Nf)
    j0 = (u - 1) * _findhits_tile + 1
    j1 = min(j0 + _findhits_tile - 1, Nr)
    h = i1 - i0 + 1
    n = h * (j1 - j0 + 1)
    k = tx
    while k <= n
        @inbounds out[k, b] = fdr[i0 + (k - 1) % h, j0 + (k - 1) ÷ h]
        k += bsz
    end
    return nothing
end
