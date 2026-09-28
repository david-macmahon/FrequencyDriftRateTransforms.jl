# Hit extraction: cluster proto-hits (points of an FDR matrix above a
# threshold) into hits, each represented by a local maximum of the
# thresholded region.  Clustering is a decreasing-value union-find sweep
# (a merge-tree construction) over the proto-hits with Chebyshev-`dist`
# linking; `min_prominence` optionally applies a persistence filter that
# keeps secondary peaks rising far enough above the saddle at which they
# merge into a higher peak (recovering sources bridged by above-threshold
# arms, and suppressing low-contrast wiggles on such arms).  Each hit also
# reports footprint info: the proto-hits its peak dominates (count and
# row/column extrema) and the extent of its contiguous above-threshold run
# in its own drift-rate column (`hitwidth`).
#
# Everything is in SNR (sigma) units by default: `threshold` and
# `min_prominence` are denormalized with the `stats` (m, s) pair, which is
# also used to normalize the returned columns, so the identities
# `hits.value .>= threshold` and `hits.prominence .>= min_prominence`
# reproduce the internal criteria exactly.  Pass `stats = (0, 1)` to work
# in raw FDR value units instead.

"""
    findhits(fdr::AbstractMatrix, threshold::Real,
             stats = noisestats(fdr);
             min_prominence = nothing, dist = 2)
        -> (; index, value, prominence, nhits, lochan, hichan,
            lorateidx, hirateidx, hitwidth)

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
- `nhits`: number of proto-hits in each hit's *footprint*, i.e. the set of
  above-threshold points its peak dominates: the whole connected region for
  region maxima; for secondary peaks, the portion merged away at the saddle
  where they retire.
- `lochan`, `hichan`: lowest/highest frequency row (channel) of the footprint.
- `lorateidx`, `hirateidx`: lowest/highest drift-rate column index of the
  footprint.
- `hitwidth`: extent (in frequency rows) of the maximal chain of the
  footprint's proto-hits in the hit's own drift-rate column, anchored at the
  hit's row, where consecutive chain members are within `dist` rows (so the
  chain never leaves the footprint); bridged gap rows count toward the
  extent.  This measures how localized the hit is along the drift axis at
  its own frequency (the "waist" of the butterfly pattern); an isolated hit
  has `hitwidth = 1`.

Thresholding and normalization are expressed in *sigma* (SNR) units via the
`stats` pair `(m, s)` (default: `noisestats(fdr)`, the noise-floor-based
statistics recommended for thresholding; see [`noisestats`](@ref) and
[`noisefloor`](@ref)).  `threshold` is a level and is denormalized as
`threshold*s + m` (as by [`noisedenormalize`](@ref)), whereas
`min_prominence` is a difference and is denormalized as `min_prominence*s`
(the mean cancels in any difference of levels).  The same `(m, s)` pair
normalizes the returned columns, so `hits.value .>= threshold` and
`hits.prominence .>= min_prominence` reproduce the internal criteria.

The `stats` pair may also be given as a pair of *vectors*, both of length
`size(fdr, 1)` (one entry per frequency channel; e.g. from
`noisestats(fdr; chans_per_band = ...)`).  Thresholding then compares each
channel `i` against `threshold*s[i] + m[i]` (per-channel thresholds), the
returned `value` and `prominence` columns are normalized with the hit's own
channel's `(m, s)`, and `min_prominence` filters each hit against
`min_prominence*s[i]` of its own channel.  Scalar and vector forms cannot be
mixed.  For CuArrays, per-channel thresholds make the GPU tile filtering
conservative (tiles are compared against the minimum threshold of their
channels), so results match the CPU method exactly.

The special pair `stats = (0, 1)` disables normalization entirely:
`threshold` and `min_prominence` are interpreted as *raw FDR values* (not
sigma units), and the returned `value` and `prominence` are likewise raw
FDR values (`value` is identical to `fdr[index]`, `prominence` is
`peak - saddle`).  This is the escape hatch for workflows that precompute
raw thresholds (e.g. via `noisedenormalize` with robust statistics):

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
                  stats = noisestats(fdr);
                  min_prominence = nothing, dist = 2)
    m, s = _findhits_stats(stats)
    dist >= 1 || throw(ArgumentError("dist must be at least 1"))
    nrows = size(fdr, 1)
    if m isa AbstractVector
        length(m) == nrows || throw(ArgumentError(
            "stats vector length ($(length(m))) must match the number of " *
            "frequency channels ($nrows)"))
        # Per-channel (raw) thresholds; row-wise comparison avoids a
        # full-matrix Boolean temporary.
        thr = m .+ threshold .* s
        protohijs = CartesianIndex{2}[]
        for i in 1:nrows
            append!(protohijs, (CartesianIndex(i, j) for j in
                                findall(>=(thr[i]), @view fdr[i, :])))
        end
    else
        threshold_raw = threshold * s + m
        protohijs = findall(>=(threshold_raw), fdr)
    end
    vals = [Float64(fdr[hij]) for hij in protohijs]
    minprom = min_prominence === nothing ? nothing :
              m isa AbstractVector ? min_prominence .* s : min_prominence * s
    _findhits_result(vals, protohijs, nrows, size(fdr, 2), m, s, minprom, dist)
end

function findhits(fdrs, threshold::Real,
                  stats = noisestats(fdrs);
                  min_prominence = nothing, dist = 2)
    m, s = _findhits_stats(stats)
    dist >= 1 || throw(ArgumentError("dist must be at least 1"))
    Nrb = size(first(fdrs), 2)
    offset = CartesianIndex(0, Nrb)
    index = CartesianIndex{2}[]
    value = Float64[]
    prominence = Float64[]
    nhits = Int[]
    lochan = Int[]
    hichan = Int[]
    lorateidx = Int[]
    hirateidx = Int[]
    hitwidth = Int[]
    for (i, fdr) in enumerate(fdrs)
        hits = findhits(fdr, threshold, (m, s); min_prominence, dist)
        joff = (i - 1) * Nrb
        append!(index, hits.index .+ ((i - 1) * offset))
        append!(value, hits.value)
        append!(prominence, hits.prominence)
        append!(nhits, hits.nhits)
        append!(lochan, hits.lochan)
        append!(hichan, hits.hichan)
        append!(lorateidx, hits.lorateidx .+ joff)
        append!(hirateidx, hits.hirateidx .+ joff)
        append!(hitwidth, hits.hitwidth)
    end
    (index = index, value = value, prominence = prominence,
     nhits = nhits, lochan = lochan, hichan = hichan,
     lorateidx = lorateidx, hirateidx = hirateidx, hitwidth = hitwidth)
end

# Validate and unpack the (m, s) normalization pair (any 2-iterable, e.g.
# the NamedTuple returned by `noisestats`): both scalars, or both vectors
# of equal length (per-channel statistics; the length is checked against
# the matrix in `findhits`).
function _findhits_stats(stats)
    m, s = stats
    if m isa AbstractVector || s isa AbstractVector
        (m isa AbstractVector && s isa AbstractVector) || throw(ArgumentError(
            "stats mean and std must both be scalars or both be vectors"))
        length(m) == length(s) || throw(ArgumentError(
            "stats mean and std vectors must have equal lengths " *
            "($(length(m)) vs $(length(s)))"))
        all(>(0), s) || throw(ArgumentError(
            "stats standard deviation values must be positive"))
        return m, s
    end
    s > 0 || throw(ArgumentError("stats standard deviation must be positive (got $s)"))
    return Float64(m), Float64(s)
end

# The empty findhits result (also used by the CUDA method's no-proto-hits
# early return).
_findhits_empty() = (index = CartesianIndex{2}[], value = Float64[],
                     prominence = Float64[], nhits = Int[],
                     lochan = Int[], hichan = Int[],
                     lorateidx = Int[], hirateidx = Int[], hitwidth = Int[])

# Assemble the sigma-domain result from the proto-hits and their (raw)
# values: run the merge-tree sweep and normalize the columns.  `m` and `s`
# are scalars or per-channel vectors; `min_prom_raw` is `nothing`, a scalar
# raw threshold, or a per-channel raw-threshold vector.
function _findhits_result(vals::Vector{Float64}, protohijs::Vector{CartesianIndex{2}},
                          nrows::Int, ncols::Int, m, s,
                          min_prom_raw, dist::Int)
    isempty(protohijs) && return _findhits_empty()
    lijs = [hij[1] + (hij[2] - 1) * nrows for hij in protohijs]
    # Per-proto-hit (raw) min_prominence thresholds when `s` is a vector:
    # each hit is filtered against `min_prominence` of its own channel.
    minprom_ord = min_prom_raw === nothing ? nothing :
                  min_prom_raw isa AbstractVector ?
                  Float64[min_prom_raw[(lij - 1) % nrows + 1] for lij in lijs] :
                  min_prom_raw
    idxs, proms_raw, nhits, lois, hais, lojs, hajs, hws =
        _findhits_sweep(vals, lijs, nrows, ncols, dist, minprom_ord)
    index = protohijs[idxs]
    rows = [(lijs[idx] - 1) % nrows + 1 for idx in idxs]
    value = [(vals[idx] - _statat(m, row)) / _statat(s, row)
             for (idx, row) in zip(idxs, rows)]
    prominence = [prom / _statat(s, row)
                  for (prom, idx, row) in zip(proms_raw, idxs, rows)]
    ord = sortperm(eachindex(idxs), by = k -> (-value[k], lijs[idxs[k]]))
    return (index = index[ord], value = value[ord], prominence = prominence[ord],
            nhits = nhits[ord], lochan = lois[ord], hichan = hais[ord],
            lorateidx = lojs[ord], hirateidx = hajs[ord], hitwidth = hws[ord])
end

# Statistics lookup by channel: vectors index, scalars broadcast.
_statat(x, row) = x isa AbstractVector ? x[row] : x

# Decreasing-value union-find sweep over the proto-hits (a merge-tree
# construction).  `vals`/`lijs` are the (raw) values and linear indices of
# the proto-hits; returns the proto-hit indices of the surviving peaks, their
# raw prominences (`Inf` for component maxima, i.e. roots), and per-hit
# footprint info: the member count and the frequency-row/drift-rate-column
# extrema of the component (at retirement time for retired peaks, final for
# roots), plus the hit's drift-axis run extent (see `_run_extent`).
function _findhits_sweep(vals::Vector{Float64}, lijs::Vector{Int}, nrows::Int,
                         ncols::Int, dist::Int,
                         min_prom::Union{Nothing, Float64, Vector{Float64}})
    n = length(vals)
    order = sortperm(1:n, by = i -> (-vals[i], lijs[i]))
    parent = collect(1:n)
    peakord = collect(1:n)             # proto-hit index of each component's peak
    peakval = copy(vals)               # value of each component's peak
    active = Dict{Int, Int}()          # linear index => proto-hit index
    sizehint!(active, n)
    # Footprint accumulators, indexed by proto-hit ordinal, valid at roots
    size = ones(Int, n)                # member count
    lorow = Vector{Int}(undef, n)      # frequency-row extrema
    harow = Vector{Int}(undef, n)
    locol = Vector{Int}(undef, n)      # drift-rate-column extrema
    hacol = Vector{Int}(undef, n)
    # Member lists as union-find linked lists (`nxt` is 0-terminated;
    # `head`/`tail` are valid at roots).  Splicing keeps a retired
    # component's chain intact as a subchain of the survivor's list, but
    # only final components are walked (see `_run_extent` note below).
    nxt = zeros(Int, n)
    head = Vector{Int}(undef, n)
    tail = Vector{Int}(undef, n)
    retired_site = Int[]
    retired_prom = Float64[]
    retired_nhits = Int[]
    retired_lorow = Int[]
    retired_harow = Int[]
    retired_locol = Int[]
    retired_hacol = Int[]

    for oi in order
        v = vals[oi]
        active[lijs[oi]] = oi
        i = (lijs[oi] - 1) % nrows + 1
        j = (lijs[oi] - 1) ÷ nrows + 1
        lorow[oi] = harow[oi] = i
        locol[oi] = hacol[oi] = j
        head[oi] = tail[oi] = oi
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
            if min_prom !== nothing && persist > 0
                # With per-channel stats, each peak is filtered against the
                # min_prominence of its own channel (see `_findhits_result`)
                promthr = min_prom isa AbstractVector ?
                          min_prom[peakord[lo]] : min_prom
                if persist >= promthr
                    push!(retired_site, peakord[lo])
                    push!(retired_prom, persist)
                    push!(retired_nhits, size[lo])
                    push!(retired_lorow, lorow[lo])
                    push!(retired_harow, harow[lo])
                    push!(retired_locol, locol[lo])
                    push!(retired_hacol, hacol[lo])
                end
            end
            # Merge footprint accumulators and member lists into the survivor
            size[hi] += size[lo]
            lorow[hi] = min(lorow[hi], lorow[lo])
            harow[hi] = max(harow[hi], harow[lo])
            locol[hi] = min(locol[hi], locol[lo])
            hacol[hi] = max(hacol[hi], hacol[lo])
            nxt[tail[hi]] = head[lo]
            tail[hi] = tail[lo]
            parent[lo] = hi
        end
    end

    # Collect the roots' hits (in discovery order) with their final
    # footprints, and record every proto-hit's final root.
    idxs = Int[]
    proms = Float64[]
    nhits = Int[]
    lois = Int[]
    hais = Int[]
    lojs = Int[]
    hajs = Int[]
    seen = Set{Int}()
    rootof = collect(1:n)
    for i in 1:n
        ra = i
        while parent[ra] != ra
            parent[ra] = parent[parent[ra]]
            ra = parent[ra]
        end
        rootof[i] = ra
        (ra in seen) && continue
        push!(seen, ra)
        push!(idxs, peakord[ra])
        push!(proms, Inf)
        push!(nhits, size[ra])
        push!(lois, lorow[ra])
        push!(hais, harow[ra])
        push!(lojs, locol[ra])
        push!(hajs, hacol[ra])
    end
    append!(idxs, retired_site)
    append!(proms, retired_prom)
    append!(nhits, retired_nhits)
    append!(lois, retired_lorow)
    append!(hais, retired_harow)
    append!(lojs, retired_locol)
    append!(hajs, retired_hacol)

    # Run extents (`hitwidth`).  Any two same-column proto-hits within
    # `dist` rows are unioned directly by the sweep, so a chain of
    # consecutive same-column members within `dist` rows never crosses a
    # (former) component boundary: the run is the same in the final
    # component as it was at any retirement, so walking final components
    # suffices for retired peaks too.
    colrows = Dict{Int, Dict{Int, Vector{Int}}}()
    hws = Vector{Int}(undef, length(idxs))
    for (k, p) in enumerate(idxs)
        lij = lijs[p]
        i0 = (lij - 1) % nrows + 1
        j0 = (lij - 1) ÷ nrows + 1
        d = get!(() -> _component_columns(head[rootof[p]], nxt, lijs, nrows),
                 colrows, rootof[p])
        hws[k] = _run_extent(d[j0], i0, dist)
    end
    return idxs, proms, nhits, lois, hais, lojs, hajs, hws
end

# Column => sorted frequency rows map of a component's member list headed
# by `h` (0-terminated via `nxt`).
function _component_columns(h::Int, nxt::Vector{Int}, lijs::Vector{Int},
                            nrows::Int)
    d = Dict{Int, Vector{Int}}()
    while h != 0
        lij = lijs[h]
        push!(get!(() -> Int[], d, (lij - 1) ÷ nrows + 1), (lij - 1) % nrows + 1)
        h = nxt[h]
    end
    foreach(sort!, values(d))
    return d
end

# Extent (in rows) of the maximal run containing `i0` in the sorted rows
# `rows`, where consecutive run members are within `dist` rows.  `i0` must
# be present in `rows`.
function _run_extent(rows::Vector{Int}, i0::Int, dist::Int)
    lo = hi = searchsortedfirst(rows, i0)
    while lo > firstindex(rows) && rows[lo] - rows[lo - 1] <= dist
        lo -= 1
    end
    while hi < lastindex(rows) && rows[hi + 1] - rows[hi] <= dist
        hi += 1
    end
    return rows[hi] - rows[lo] + 1
end
