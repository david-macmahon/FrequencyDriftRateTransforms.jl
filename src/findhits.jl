# Hit extraction: cluster proto-hits (points of an FDR matrix above a
# threshold) into hits, each represented by a local maximum of the
# thresholded region.  Clustering is a decreasing-value union-find sweep
# (a merge-tree construction) over the proto-hits with Chebyshev-`dist`
# linking; `min_prominence` optionally applies a persistence filter that
# keeps secondary peaks rising far enough above the saddle at which they
# merge into a higher peak (recovering sources bridged by above-threshold
# arms, and suppressing low-contrast wiggles on such arms).
#
# Everything is in SNR (sigma) units by default: `threshold` and
# `min_prominence` are denormalized with the `stats` (m, s) pair, which is
# also used to normalize the returned columns, so the identities
# `hits.value .>= threshold` and `hits.prominence .>= min_prominence`
# reproduce the internal criteria exactly.  Pass `stats = (0, 1)` to work
# in raw FDR value units instead.

"""
    findhits(fdr::AbstractMatrix, threshold::Real,
             stats = fdrstats(fdr; robust = true);
             min_prominence = nothing, dist = 2)
        -> (index = ..., value = ..., prominence = ...)

Cluster the proto-hits of the Frequency-Drift-Rate (FDR) matrix `fdr` (the
points with a value of at least `threshold`) into *hits*: the local maxima
of the thresholded region(s), one per unique signal candidate.  Returns a
NamedTuple with (columnar) fields:

- `index`: `Vector{CartesianIndex{2}}` location of each hit's peak.
- `value`: peak value of each hit, normalized as a z-score
  `(peak - m)/s` from the `stats` pair.
- `prominence`: prominence of each hit in sigma units, `(peak - saddle)/s`.
  The maximum of each connected above-threshold region is always reported,
  with `prominence = Inf` (it never merges into a higher peak).  Secondary
  peaks are reported only when they rise at least `min_prominence` above
  the saddle at which they merge into a higher peak; their prominence is
  the height of that rise, so `hits.prominence .>= min_prominence`
  reproduces the reported set exactly.

Thresholding and normalization are expressed in *sigma* (SNR) units via the
`stats` pair `(m, s)` (default: `fdrstats(fdr; robust = true)`, the
noise-floor-based statistics recommended for thresholding; see
[`fdrstats`](@ref) and [`noisefloor`](@ref)).  `threshold` is a level and is
denormalized as `threshold*s + m` (as by [`fdrdenormalize`](@ref)), whereas
`min_prominence` is a difference and is denormalized as `min_prominence*s`
(the mean cancels in any difference of levels).  The same `(m, s)` pair
normalizes the returned columns, so `hits.value .>= threshold` and
`hits.prominence .>= min_prominence` reproduce the internal criteria.

The special pair `stats = (0, 1)` disables normalization entirely:
`threshold` and `min_prominence` are interpreted as *raw FDR values* (not
sigma units), and the returned `value` and `prominence` are likewise raw
FDR values (`value` is identical to `fdr[index]`, `prominence` is
`peak - saddle`).  This is the escape hatch for workflows that precompute
raw thresholds (e.g. via `fdrdenormalize` with robust statistics):

```julia
hits = findhits(fdr, threshold_raw, (0, 1); min_prominence = prominence_raw)
```

Keyword arguments:

- `min_prominence`: `nothing` (default) reports only the maximum of each
  connected above-threshold region (cluster-peak semantics; `Inf`
  prominence for every hit).  A `Real` value additionally reports secondary
  peaks whose persistence reaches it, which recovers distinct signals
  bridged by an above-threshold arm (e.g. the X-shaped pattern a strong
  drifting signal leaves in an FDR matrix) and suppresses low-contrast
  wiggles along such arms.
- `dist`: linking distance in Chebyshev metric; proto-hits within `dist`
  (in both frequency and drift-rate index) belong to the same region.
  The default of 2 bridges single-pixel gaps.

For `CuArray`s, the CUDA extension runs the thresholding on the device
(via 32x32 tile maxima, so only tiles containing proto-hits are transferred)
and the clustering on the host.  An iterable of FDR matrices is treated as
drift-rate adjacent portions of a larger FDR matrix (as if they had been
`hcat`'d together), with `stats` pooled across them by default.

For a single FDR matrix the hits are sorted by descending `value` (ties by
index); for an iterable they are in batch order (each batch sorted).
"""
function findhits(fdr::AbstractMatrix, threshold::Real,
                  stats = fdrstats(fdr; robust = true);
                  min_prominence = nothing, dist = 2)
    m, s = _findhits_stats(stats)
    dist >= 1 || throw(ArgumentError("dist must be at least 1"))
    threshold_raw = threshold * s + m
    min_prom_raw = min_prominence === nothing ? nothing : min_prominence * s
    protohijs = findall(>=(threshold_raw), fdr)
    vals = [Float64(fdr[hij]) for hij in protohijs]
    nrows, ncols = size(fdr)
    _findhits_result(vals, protohijs, nrows, ncols, m, s, min_prom_raw, dist)
end

function findhits(fdrs, threshold::Real,
                  stats = fdrstats(fdrs; robust = true);
                  min_prominence = nothing, dist = 2)
    m, s = _findhits_stats(stats)
    dist >= 1 || throw(ArgumentError("dist must be at least 1"))
    Nrb = size(first(fdrs), 2)
    offset = CartesianIndex(0, Nrb)
    index = CartesianIndex{2}[]
    value = Float64[]
    prominence = Float64[]
    for (i, fdr) in enumerate(fdrs)
        hits = findhits(fdr, threshold, (m, s); min_prominence, dist)
        append!(index, hits.index .+ ((i - 1) * offset))
        append!(value, hits.value)
        append!(prominence, hits.prominence)
    end
    (index = index, value = value, prominence = prominence)
end

# Validate and unpack the (m, s) normalization pair (any 2-iterable, e.g.
# the NamedTuple returned by `fdrstats`).
function _findhits_stats(stats)
    m, s = stats
    s > 0 || throw(ArgumentError("stats standard deviation must be positive (got $s)"))
    return Float64(m), Float64(s)
end

# Assemble the sigma-domain result from the proto-hits and their (raw)
# values: run the merge-tree sweep and normalize the columns.
function _findhits_result(vals::Vector{Float64}, protohijs::Vector{CartesianIndex{2}},
                          nrows::Int, ncols::Int, m::Float64, s::Float64,
                          min_prom_raw::Union{Nothing, Float64}, dist::Int)
    isempty(protohijs) && return (index = CartesianIndex{2}[], value = Float64[],
                                  prominence = Float64[])
    lijs = [hij[1] + (hij[2] - 1) * nrows for hij in protohijs]
    idxs, proms_raw = _findhits_sweep(vals, lijs, nrows, ncols, dist, min_prom_raw)
    index = protohijs[idxs]
    value = [(vals[i] - m) / s for i in idxs]
    prominence = [prom / s for prom in proms_raw]
    ord = sortperm(eachindex(idxs), by = k -> (-value[k], lijs[idxs[k]]))
    return (index = index[ord], value = value[ord], prominence = prominence[ord])
end

# Decreasing-value union-find sweep over the proto-hits (a merge-tree
# construction).  `vals`/`lijs` are the (raw) values and linear indices of
# the proto-hits; returns the proto-hit indices of the surviving peaks and
# their raw prominences (`Inf` for component maxima, i.e. roots).
function _findhits_sweep(vals::Vector{Float64}, lijs::Vector{Int}, nrows::Int,
                         ncols::Int, dist::Int, min_prom::Union{Nothing, Float64})
    n = length(vals)
    order = sortperm(1:n, by = i -> (-vals[i], lijs[i]))
    parent = collect(1:n)
    peakord = collect(1:n)             # proto-hit index of each component's peak
    peakval = copy(vals)               # value of each component's peak
    active = Dict{Int, Int}()          # linear index => proto-hit index
    sizehint!(active, n)
    retired_site = Int[]
    retired_prom = Float64[]

    for oi in order
        v = vals[oi]
        active[lijs[oi]] = oi
        i = (lijs[oi] - 1) % nrows + 1
        j = (lijs[oi] - 1) ÷ nrows + 1
        for dj in -dist:dist, di in -dist:dist
            (di == 0 && dj == 0) && continue
            ii = i + di
            jj = j + dj
            (1 <= ii <= nrows && 1 <= jj <= ncols) || continue
            other = get(active, ii + (jj - 1) * nrows, 0)
            other == 0 && continue
            # Union the components, retiring the lower peak at saddle `v`
            ra = oi
            while parent[ra] != ra
                parent[ra] = parent[parent[ra]]
                ra = parent[ra]
            end
            rb = other
            while parent[rb] != rb
                parent[rb] = parent[parent[rb]]
                rb = parent[rb]
            end
            ra == rb && continue
            # The lower peak retires at saddle `v`; ties favor the
            # earlier-activated peak (smaller linear index)
            if peakval[ra] > peakval[rb] ||
               (peakval[ra] == peakval[rb] && lijs[peakord[ra]] < lijs[peakord[rb]])
                lo, hi = rb, ra
            else
                lo, hi = ra, rb
            end
            persist = peakval[lo] - v
            if min_prom !== nothing && persist > 0 && persist >= min_prom
                push!(retired_site, peakord[lo])
                push!(retired_prom, persist)
            end
            parent[lo] = hi
        end
    end

    idxs = Int[]
    proms = Float64[]
    roots = Set{Int}()
    for i in 1:n
        ra = i
        while parent[ra] != ra
            ra = parent[ra]
        end
        (ra in roots) && continue
        push!(roots, ra)
        push!(idxs, peakord[ra])
        push!(proms, Inf)
    end
    append!(idxs, retired_site)
    append!(proms, retired_prom)
    return idxs, proms
end
