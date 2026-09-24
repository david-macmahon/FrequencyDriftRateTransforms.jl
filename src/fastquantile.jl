# Exact quantile selection without copying or sorting the data, used by
# `noisefloor` for its signal-free quantiles and available directly as
# `fast_quantile` (see its docstring for the algorithm and guarantees).
using Statistics

# Generic fallback for eltypes without an order-preserving unsigned key.
fast_quantile(data, ps) = quantile(vec(data), ps)

# Eltypes with an exactly order-preserving unsigned key.
const _select_eltypes = Union{Bool, Int8, Int16, Int32, Int64, UInt8, UInt16,
                              UInt32, UInt64, Float16, Float32, Float64}

# Order-preserving map to `UInt64`: sign-biased bits.
_monokey(x::Bool) = UInt64(x)
_monokey(x::Union{Int8, Int16, Int32, Int64}) =
    reinterpret(UInt64, Int64(x)) ⊻ 0x8000000000000000
_monokey(x::Union{UInt8, UInt16, UInt32, UInt64}) = UInt64(x)
_monokey(x::Float64) =
    ifelse(signbit(x), ~reinterpret(UInt64, x),
           reinterpret(UInt64, x) | 0x8000000000000000)
_monokey(x::Float32) = UInt64(
    ifelse(signbit(x), ~reinterpret(UInt32, x), reinterpret(UInt32, x) | 0x80000000))
_monokey(x::Float16) = _monokey(Float32(x))

# The exact value a key represents (inverse of `_monokey`).
_unmonokey(::Type{Bool}, k::UInt64) = k == 1
_unmonokey(::Type{T}, k::UInt64) where {T <: Union{Int8, Int16, Int32, Int64}} =
    reinterpret(Int64, k ⊻ 0x8000000000000000) % T
_unmonokey(::Type{T}, k::UInt64) where {T <: Union{UInt8, UInt16, UInt32, UInt64}} = k % T
function _unmonokey(::Type{Float64}, k::UInt64)
    u = ifelse(k >>> 63 == one(UInt64), k & 0x7fffffffffffffff, ~k)
    reinterpret(Float64, u)
end
function _unmonokey(::Type{Float32}, k::UInt64)
    u = UInt32(k)
    reinterpret(Float32, ifelse(u >>> 31 == one(UInt32), u & 0x7fffffff, ~u))
end
_unmonokey(::Type{Float16}, k::UInt64) = Float16(_unmonokey(Float32, k))

# Range of keys a type's values occupy.
_keybounds(::Type{<:Union{Float16, Float32}}) = (UInt64(0), UInt64(0xffffffff))
_keybounds(::Type{Bool}) = (UInt64(0), UInt64(1))
_keybounds(::Type{<:_select_eltypes}) = (UInt64(0), typemax(UInt64))

# One open binning problem: a key interval known to contain the elements of
# ranks `ranks` (1-based, global), with `below` elements keyed below `klo`.
mutable struct _SelectTask
    klo::UInt64
    khi::UInt64
    below::Int
    ranks::Vector{Int}
end

# Bin count and bit shift that partition a task's key interval; with at
# most 2048 keys left, every key gets its own bin and the next refinement
# pass resolves the task.
function _task_bins(task::_SelectTask)
    width = task.khi - task.klo
    width < 2048 ? (Int(width) + 1, 0) : (2048, 64 - leading_zeros(width) - 11)
end

# Advance the selection by one pass: resolve ranks whose bin covers a
# single key, and spawn tasks for the remaining bins.
function _refine!(resolved::Dict{Int, T}, tasks::Vector{_SelectTask},
                  hists::Vector{Vector{Int}}) where T
    newtasks = _SelectTask[]
    for (task, hist) in zip(tasks, hists)
        nbins, shift = _task_bins(task)
        width = task.khi - task.klo
        bin = 1
        below = task.below
        ranks = task.ranks
        i = 1
        while i ≤ length(ranks)
            while below + hist[bin] < ranks[i]
                below += hist[bin]
                bin += 1
            end
            top = below + hist[bin]
            j = i
            while j ≤ length(ranks) && ranks[j] ≤ top
                j += 1
            end
            binlo = task.klo + (UInt64(bin - 1) << shift)
            binhi = task.klo + min((UInt64(bin) << shift) - one(UInt64), width)
            if binlo == binhi
                value = _unmonokey(T, binlo)
                for r in i:j - 1
                    resolved[ranks[r]] = value
                end
            else
                push!(newtasks, _SelectTask(binlo, binhi, below, ranks[i:j - 1]))
            end
            below = top
            bin += 1
            i = j
        end
    end
    return newtasks
end

# Driver: repeatedly histogram the data within the open tasks via
# `histpass(data, tasks, checknan) -> Vector{Vector{Int}}` (one histogram
# per task), then refine.  Returns rank => value for every requested rank.
function _select_ranks(histpass, data::AbstractArray{<:_select_eltypes},
                       ranks::Vector{Int})
    T = eltype(data)
    resolved = Dict{Int, T}()
    klo, khi = _keybounds(T)
    tasks = [_SelectTask(klo, khi, 0, sort!(unique(ranks)))]
    checknan = true
    while !isempty(tasks)
        hists = histpass(data, tasks, checknan)
        checknan = false
        tasks = _refine!(resolved, tasks, hists)
    end
    return resolved
end

# Histogram one index range into per-task histograms; returns whether the
# range contains a NaN (only checked when `checknan`).
function _hist_range!(hists::Vector{Vector{Int}}, data::AbstractArray,
                      tasks::Vector{_SelectTask}, lo::Int, hi::Int,
                      checknan::Bool)
    klos = UInt64[task.klo for task in tasks]
    khis = UInt64[task.khi for task in tasks]
    shifts = Int[last(_task_bins(task)) for task in tasks]
    @inbounds for i in lo:hi
        x = data[i]
        checknan && x != x && return true
        key = _monokey(x)
        for t in eachindex(tasks)
            if klos[t] ≤ key ≤ khis[t]
                hists[t][1 + Int((key - klos[t]) >>> shifts[t])] += 1
                break
            end
        end
    end
    return false
end

function _select_histpass!(data::AbstractArray{<:_select_eltypes}, tasks, checknan::Bool)
    nbins = [first(_task_bins(task)) for task in tasks]
    hists = [zeros(Int, nb) for nb in nbins]
    n = length(data)
    nthreads = Threads.nthreads()
    if nthreads == 1 || n < 1 << 20
        _hist_range!(hists, data, tasks, 1, n, checknan) && throw(ArgumentError(
            "quantiles are undefined in presence of NaNs or missing values"))
    else
        chunk = cld(n, nthreads)
        localhists = [[zeros(Int, nb) for nb in nbins] for _ in 1:nthreads]
        localnan = falses(nthreads)
        Threads.@threads for c in 1:nthreads
            lo = (c - 1) * chunk + 1
            lo > n && continue
            localnan[c] = _hist_range!(localhists[c], data, tasks, lo,
                                       min(n, c * chunk), checknan)
        end
        any(localnan) && throw(ArgumentError(
            "quantiles are undefined in presence of NaNs or missing values"))
        for c in 1:nthreads, t in eachindex(tasks)
            hists[t] .+= localhists[c][t]
        end
    end
    return hists
end

# Julia's quantile interpolation for the default `alpha = beta = 1`,
# replicated from `Statistics._quantile`.
@inline function _quantile_interp(a, b, γ)
    if isfinite(a) && isfinite(b) && a ≈ b
        a + γ * (b - a)
    else
        (1 - γ) * a + γ * b
    end
end

function _fast_quantile_impl(histpass, data::AbstractArray{<:_select_eltypes},
                             ps::AbstractVector{P}) where P
    n = length(data)
    n == 0 && throw(ArgumentError("empty data vector"))
    isempty(ps) && return zeros(promote_type(eltype(data), P), 0)
    for p in ps
        0 <= p <= 1 || throw(ArgumentError("input probability out of [0,1] range"))
    end
    γs = Vector{promote_type(P, Int)}(undef, length(ps))
    js = Vector{Int}(undef, length(ps))
    ranks = Int[]
    for (i, p) in enumerate(ps)
        alpha = beta = 1.0
        m = alpha + p * (one(alpha) - alpha - beta)
        aleph = fma(n, p, oftype(p, m))
        j = clamp(trunc(Int, aleph), 1, n - 1)
        js[i] = j
        γs[i] = clamp(aleph - j, 0, 1)
        if n == 1
            push!(ranks, 1)
        else
            push!(ranks, j, j + 1)
        end
    end
    resolved = _select_ranks(histpass, data, ranks)
    if n == 1
        v = resolved[1]
        return [_quantile_interp(v, v, γ) for γ in γs]
    end
    return [_quantile_interp(resolved[j], resolved[j + 1], γs[i])
            for (i, j) in enumerate(js)]
end

_fast_quantile_impl(histpass, data, ps::Real) =
    _fast_quantile_impl(histpass, data, [ps])[1]
_fast_quantile_impl(histpass, data, ps::Tuple) =
    Tuple(_fast_quantile_impl(histpass, data, collect(ps)))
_fast_quantile_impl(histpass, data, ps) =
    _fast_quantile_impl(histpass, data, collect(ps))

"""
    fast_quantile(data, ps) -> Vector (scalar for `ps::Real`, tuple for `ps::Tuple`)

Exact quantiles of `data`, matching `Statistics.quantile(vec(data), ps)`
bit-for-bit: the same rank arithmetic, interpolation (`alpha = beta = 1`),
NaN handling, and result types, which the test suite verifies.  Unlike
`quantile`, the data is neither copied nor sorted, which makes this much
faster for large arrays: `Statistics.quantile` single-threadedly partially
sorts a copy (about 95 s for 4 GiB), while `fast_quantile` selects the
handful of required order statistics by iterative histogram refinement.
Values are remapped to unsigned integer keys that preserve their total
order (the sign-flipped IEEE bit patterns for floats), the key interval
containing each required rank is narrowed by binning into 2048 bins per
pass, and a bin holding a single key resolves every rank it contains.  Ties
resolve naturally, the data is touched by a fixed handful of sequential
passes (three for `Float32`), and no sorted copy is ever materialized.  The
histogram passes are multithreaded when `Threads.nthreads()` is greater
than one and the data is large.

For `CuArray`s, the CUDA extension runs the same algorithm on the device,
so the data never leaves the GPU.  Inputs with other eltypes fall back to
`quantile`.
"""
fast_quantile(data::AbstractArray{<:_select_eltypes}, ps) =
    _fast_quantile_impl(_select_histpass!, data, ps)
