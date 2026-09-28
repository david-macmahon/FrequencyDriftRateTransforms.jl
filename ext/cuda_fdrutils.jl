# CUDA banded (per-channel-group) statistics for `noisestats`.

import FrequencyDriftRateTransforms: _banded_colstats!

# One-pass banded mean/std on the device: one thread per (band, column)
# pair accumulates the band's `cpb` contiguous values in Float64 and writes
# the corrected mean and standard deviation into the preallocated outputs —
# no intermediate allocations; the matrix is read exactly once.  (A warp
# per (band, column) pair with coalesced loads and shuffle reduction was
# measured *slower* — one load in flight per warp versus one per thread —
# so this simpler mapping is kept.)  The one-pass sum/sum-of-squares
# formula differs from `Statistics.std`'s two-pass algorithm only at
# floating-point rounding level (the accumulation is done in Float64).
function _banded_colstats!(colmean::CuMatrix{T}, colstd::CuMatrix{T},
                           fdr::CuMatrix{T}, cpb::Int) where T
    n = length(colmean)
    if n > 0
        threads = 256
        blocks = cld(n, threads)
        @cuda threads = threads blocks = blocks _banded_colstats_kernel!(
            colmean, colstd, fdr, cpb)
    end
    return colmean, colstd
end

function _banded_colstats_kernel!(colmean::CuDeviceMatrix{T},
                                  colstd::CuDeviceMatrix{T},
                                  fdr::CuDeviceMatrix{T}, cpb::Int) where T
    lid = (Int64(blockIdx().x) - 1) * Int64(blockDim().x) + Int64(threadIdx().x)
    lid ≤ length(colmean) || return nothing
    nbands = Int(size(colmean, 1))
    b = Int((lid - 1) % nbands) + 1
    j = Int((lid - 1) ÷ nbands) + 1
    i0 = (b - 1) * cpb
    s1 = 0.0
    s2 = 0.0
    @inbounds for i in 1:cpb
        x = Float64(fdr[i0 + i, j])
        s1 += x
        s2 += x * x
    end
    m = s1 / cpb
    var = cpb == 1 ? 0.0 : max((s2 - s1 * m) / (cpb - 1), 0.0)
    @inbounds colmean[lid] = T(m)
    @inbounds colstd[lid] = T(sqrt(var))
    return nothing
end
