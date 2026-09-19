# The Taylor tree algorithm computes the sums along all diagonal paths
# through a spectrogram in O(Nf * Nt * log2(Nt)) operations.  A path touches
# one time sample per time step and follows the nearest cell to a straight
# line, so each path approximates a constant frequency drift rate.  This
# implementation is based on the reference CUDA implementation in seticore
# (https://github.com/lacker/seticore/blob/master/taylor.cu), which is itself
# based on Franklin Antonio's CudaTaylor5demo.cu from
# https://github.com/UCBerkeleySETI/dedopplerperf.
#
# The implementation uses only whole-array operations (broadcasts and fill!),
# so it works transparently (though not optimally) on GPU arrays such as
# CuArray; a dedicated CUDA kernel (as in the seticore reference) would be
# considerably faster.

"""
    taylorstep!(target, source, path_length, drift_block[, Nt]) -> target

Run one step of the Taylor tree algorithm, computing the sums of paths of
length `path_length` in `target` from the sums of paths of length
`path_length/2` in `source`.  A *path* is a diagonal path through the data
that touches one time sample per time step and follows the nearest cell to a
straight line.  `source` and `target` must be `AbstractMatrix`s with
frequency along the first dimension and time along the second dimension
(like the `spectrogram` argument of `intfdr!` etc.).  `path_length` must be
a power of 2 (≥ 2) that does not exceed `size(target, 2)`, and `size(target,
2)` must be a multiple of `path_length`.  `Nt` is the number of real
(non-padded) time samples; it defaults to `size(source, 2)`, i.e. that all
time samples of `source` are real.

# Extended help

If `Nt < size(target, 2)` (the *logical*, zero-padded time sample count), the
step behaves as if `source` were zero-padded in time to `size(target, 2)`
samples, without ever storing or computing on the padding:

* Time blocks entirely beyond `Nt` hold all-zero path sums, so they are
  never written (and never read, see below).
* When the second half path of a target entry comes from an all-zero time
  block, adding its (zero) contribution is a no-op, so the first half path
  sum is copied instead of added.  This "boundary" case is the only way an
  all-zero block would be read, so skipping it means the never-written
  (stale) regions of work buffers are never read either.

For this to work, `source` must contain the path sums of all time blocks
containing real data, i.e. the first `cld(Nt, path_length/2) *
(path_length/2)` of its time columns (its own last block may itself be
partially padded).  When `Nt` is an exact multiple of `path_length` (in
particular whenever `Nt == size(target, 2)`), every block with data is a
full add and no padding semantics are in play.

At this stage of the algorithm, column `t` of the buffers holds the path
sums for zero-based time block `div(t-1, path_length)` and path offset
`rem(t-1, path_length)`, where time block `b` holds the sums for paths that
start at time `b*path_length` (zero-based).

`drift_block` shifts the paths by an additional `drift_block` frequency
channels per time step, so a `taylortree!` output built from steps of this
kind holds, for drift block `b`, path sums for the normalized drift rates
`b + p/(Ntp-1)` for `p` in `0:(Ntp-1)`, where `Ntp = size(target, 2)` is the
logical number of time samples of the full tree.  This matches the `intfdr!`
column convention (frequency index = starting frequency at time 1).

Only the entries of `target` for paths that stay within the frequency band
are written; entries for paths that would extend beyond the band retain
their previous contents.  Because paths are monotonic in frequency, such
unwritten entries are never read by subsequent steps of the algorithm (any
path that stays in bounds is composed entirely of sub-paths that stay in
bounds), so uninitialized data never propagates into valid path sums.
[`taylortree!`](@ref) zeroes the invalid entries of its final output.  (For
the full tree, requiring the *total* time sample count to be a power of 2 is
a property of `taylortree!`, where `Ntp = nextpow(2, Nt)` fixes the drift
rate grid.)
"""
function taylorstep!(target::AbstractMatrix, source::AbstractMatrix,
                     path_length::Integer, drift_block::Integer,
                     Nt::Integer = size(source, 2))
    Nf = size(source, 1)
    Ntp = size(target, 2)
    L = path_length
    ispow2(L) && 2 <= L <= Ntp ||
        throw(ArgumentError("path_length ($L) must be a power of 2 (≥ 2) that does not exceed Ntp ($Ntp)"))
    size(target, 1) == Nf ||
        throw(ArgumentError("target must have $Nf rows (got $(size(target, 1)))"))
    Ntp >= Nt ||
        throw(ArgumentError("Nt ($Nt) must not exceed the logical time sample count of target ($Ntp)"))
    rem(Ntp, L) == 0 ||
        throw(ArgumentError("Ntp ($Ntp) must be a multiple of path_length ($L)"))
    half = L ÷ 2
    # Number of source (path_length/2) and target (path_length) time blocks
    # that contain real data; the last of each may be partially padded.
    nd2 = cld(Nt, half)
    ndL = cld(Nt, L)
    # The source must contain the path sums of all real-data source blocks.
    size(source, 2) >= nd2 * half ||
        throw(ArgumentError("source must have at least $(nd2 * half) time columns for Nt ($Nt) and path_length ($L)"))
    for tb in 0:(ndL - 1)
        base = tb * L
        if 2 * tb + 1 <= nd2 - 1
            # Both source time blocks contain real data: full add.
            for path_offset in 0:(L - 1)
                half_offset = path_offset >> 1
                shift = (path_offset + 1) >> 1 + drift_block * half
                drift_channels = path_offset + drift_block * (L - 1)
                tcol = base + path_offset + 1
                scol1 = base + half_offset + 1
                scol2 = scol1 + half
                # Only write entries for paths that stay within the frequency
                # band: the shifted read (start of the second half path) and
                # the path's end frequency (f + drift_channels) must both stay in
                # bounds.
                chan_lo = max(1, 1 - shift, 1 - drift_channels)
                chan_hi = min(Nf, Nf - shift, Nf - drift_channels)
                @inbounds @views target[chan_lo:chan_hi, tcol] .=
                    source[chan_lo:chan_hi, scol1] .+ source[chan_lo+shift:chan_hi+shift, scol2]
            end
        else
            # The second source time block is entirely zero-padded, so the
            # target entries equal the first half path sums.  Copy them,
            # restricted to entries that are valid for both the half path
            # (end frequency f + drift_channels_half) and the full path (end
            # frequency f + drift_channels).
            @assert 2 * tb == nd2 - 1
            for path_offset in 0:(L - 1)
                half_offset = path_offset >> 1
                drift_channels_half = half_offset + drift_block * (half - 1)
                drift_channels = path_offset + drift_block * (L - 1)
                tcol = base + path_offset + 1
                scol = base + half_offset + 1
                chan_lo = max(1, 1 - drift_channels_half, 1 - drift_channels)
                chan_hi = min(Nf, Nf - drift_channels_half, Nf - drift_channels)
                @inbounds @views target[chan_lo:chan_hi, tcol] .= source[chan_lo:chan_hi, scol]
            end
        end
    end
    return target
end

"""
    taylortree!(buffer1, buffer2, spectrogram, drift_block) -> result

Run all rounds of the Taylor tree algorithm on `spectrogram` for the given
`drift_block`, using `buffer1` and `buffer2` as ping-pong work buffers, and
return the buffer containing the results.  `spectrogram` is only read, never
written, and must be distinct from both buffers.  The buffers must be
distinct from each other and both have size
`(size(spectrogram, 1), Ntp)` where `Ntp = nextpow(2, Nt)` and `Nt` is the
number of time samples in `spectrogram` (at least 2).  When CUDA is loaded
and the arrays are `CuArray`s, a CUDA kernel implementation of the tree is
used; it materializes the zero padding described in the extended help
instead of virtualizing it (the results are identical).

# Extended help

If `Nt` is not a power of 2, the algorithm behaves as if `spectrogram` were
zero-padded in time to `Ntp` samples, but without ever storing or computing
on the padding: time blocks entirely beyond `Nt` are never touched, and when
the second half of a path falls entirely within the padding, the first half
path sum is copied instead of added (adding an all-zero block is a no-op).
The cost of the transform is therefore proportional to `Nt` rather than to
`Ntp`, and no padded copy of the input is ever made.

The result is a `Matrix` in which column `p+1` (zero-based path offset `p`)
holds, for each starting frequency, the sum along the path that starts at
time 1 at that frequency and drifts `drift_block*(Ntp-1) + p` frequency
channels over the `Ntp-1` steps between time samples, i.e. the path sum for
normalized drift rate `drift_block + p/(Ntp-1)`.  This matches the `intfdr!`
column convention (frequency index = starting frequency at time 1).  Note
that the path follows the Taylor tree's staircase approximation of a straight
line (as in the reference implementation): the net drift over the full path
is exact, but for `Ntp > 8` individual steps can differ by up to 1 channel
from the per-step rounded straight line used by `intfdr!` (for `Ntp <= 8` the
two coincide exactly).

Path sums that extend beyond the frequency band are set to zero.  Unlike
`intfdr!`, which wraps paths around circularly, the Taylor tree algorithm
does not wrap paths around the edges of the spectrogram.

The Taylor tree algorithm computes all `Ntp` path offsets for a given
`drift_block` in `O(Nf * Nt * log2(Ntp))` operations, i.e. all `Ntp` rates
for the cost of about `log2(Ntp)` rates in the `O(Nf * Nt)` per rate
`intfdr!`.
"""
function taylortree!(buffer1::AbstractMatrix, buffer2::AbstractMatrix,
                     spectrogram::AbstractMatrix{<:Real}, drift_block::Integer)
    Nf, Nt = size(spectrogram)
    Ntp = nextpow(2, Nt)
    size(buffer1) == (Nf, Ntp) ||
        throw(ArgumentError("buffer1 must have size ($Nf, $Ntp) (got $(size(buffer1)))"))
    size(buffer2) == (Nf, Ntp) ||
        throw(ArgumentError("buffer2 must have size ($Nf, $Ntp) (got $(size(buffer2)))"))
    buffer1 === buffer2 &&
        throw(ArgumentError("buffer1 and buffer2 must be distinct"))
    (spectrogram === buffer1 || spectrogram === buffer2) &&
        throw(ArgumentError("spectrogram must be distinct from buffer1 and buffer2"))
    Nt >= 2 || throw(ArgumentError("number of time samples ($Nt) must be at least 2"))
    # The datachan_low among the buffers looks like:
    # spectrogram -> buffer1 -> buffer2 -> buffer1 -> buffer2 -> ...
    # The first step reads `spectrogram` and writes `buffer1`.  After that,
    # the source_buffer/target_buffer aliases simply exchange roles on each
    # pass through the loop: read from source_buffer, write to target_buffer.
    # After the loop, source_buffer holds the final results.
    taylorstep!(buffer1, spectrogram, 2, drift_block, Nt)
    source_buffer, target_buffer = buffer1, buffer2
    path_length = 4
    while path_length <= Ntp
        taylorstep!(target_buffer, source_buffer, path_length, drift_block, Nt)
        source_buffer, target_buffer = target_buffer, source_buffer
        path_length *= 2
    end
    # Entries of the final result for paths that extend beyond the frequency
    # band were never written by taylorstep!, so they may hold arbitrary
    # values.  Zero them so that the result is fully well defined.
    for path_offset in 0:(Ntp - 1)
        drift_channels = (Ntp - 1) * drift_block + path_offset
        chan_lo = max(1, 1 - drift_channels)
        chan_hi = min(Nf, Nf - drift_channels)
        col = path_offset + 1
        source_buffer[1:min(chan_lo - 1, Nf), col] .= zero(eltype(source_buffer))
        source_buffer[max(1, chan_hi + 1):Nf, col] .= zero(eltype(source_buffer))
    end
    return source_buffer
end

"""
    taylortree(spectrogram, drift_block) -> result

Same as the `taylortree!` function, but the two work buffers are allocated
automatically.
"""
function taylortree(spectrogram::AbstractMatrix{<:Real}, drift_block::Integer)
    Nf, Nt = size(spectrogram)
    Ntp = nextpow(2, Nt)
    taylortree!(similar(spectrogram, Nf, Ntp), similar(spectrogram, Nf, Ntp),
                spectrogram, drift_block)
end

"""
    TaylorWorkspace(spectrogram) -> workspace

Create a *workspace* suitable for use with `taylorfdr!`.  The workspace
includes the two work buffers needed for creating a frequency drift rate
matrix for `spectrogram` using `taylorfdr` and `taylorfdr!`.  The buffers are
sized for the zero-padded time sample count `Ntp = nextpow(2, Nt)` (see
`taylortree!`), so a workspace can be reused for any spectrogram with the
same number of rows and up to `Ntp` time samples.  Reusing a workspace avoids
reallocating the work buffers for each transform.
"""
struct TaylorWorkspace
    Nf::Int
    Ntp::Int
    buffer1::AbstractMatrix{<:Real}
    buffer2::AbstractMatrix{<:Real}
end

function TaylorWorkspace(spectrogram::AbstractMatrix{<:Real})
    Nf, Nt = size(spectrogram)
    Ntp = nextpow(2, Nt)
    TaylorWorkspace(Nf, Ntp, similar(spectrogram, Nf, Ntp),
                    similar(spectrogram, Nf, Ntp))
end

function Base.show(io::IO, ws::TaylorWorkspace)
    print(io, typeof(ws))
    print(io, "(Nf=", ws.Nf)
    print(io, ",Ntp=", ws.Ntp)
    print(io, ")")
end

function Base.sizeof(ws::TaylorWorkspace)
    sizeof(ws.buffer1) + sizeof(ws.buffer2)
end

"""
    taylorfdr!(fdr, spectrogram, drift_blocks) -> fdr
    taylorfdr!(fdr, workspace, spectrogram, drift_blocks) -> fdr

Same as the `taylorfdr` function, but store the results in `fdr`, which is
also returned.  The size of `fdr` must be
`(size(spectrogram, 1), length(drift_blocks) * Ntp)`, where
`Ntp = nextpow(2, size(spectrogram, 2))`.  The columns of `fdr` are grouped
by drift block in the order given by `drift_blocks`; the drift rates for the
columns of group `i` are given by the `i`th range returned by
`taylorrates(size(spectrogram, 2), drift_blocks)`.

A `TaylorWorkspace` created for `spectrogram` may be passed as `workspace` to
avoid reallocating the work buffers on each call.
"""
function taylorfdr!(fdr::AbstractMatrix, workspace::TaylorWorkspace,
                    spectrogram::AbstractMatrix{<:Real}, drift_blocks)
    Nf, Nt = size(spectrogram)
    Ntp = nextpow(2, Nt)
    Nr = Ntp * length(drift_blocks)
    size(fdr) == (Nf, Nr) ||
        throw(ArgumentError("fdr must have size ($Nf, $Nr) (got $(size(fdr)))"))
    for (i, b) in enumerate(drift_blocks)
        result = taylortree!(workspace.buffer1, workspace.buffer2, spectrogram, b)
        copyto!(@view(fdr[:, (i-1)*Ntp .+ (1:Ntp)]), result)
    end
    return fdr
end

function taylorfdr!(fdr::AbstractMatrix, workspace::TaylorWorkspace,
                    spectrogram::AbstractMatrix{<:Real}, drift_block::Integer)
    taylorfdr!(fdr, workspace, spectrogram, (drift_block,))
end

function taylorfdr!(fdr::AbstractMatrix, spectrogram::AbstractMatrix{<:Real},
                    drift_blocks)
    taylorfdr!(fdr, TaylorWorkspace(spectrogram), spectrogram, drift_blocks)
end

function taylorfdr!(fdr::AbstractMatrix, spectrogram::AbstractMatrix{<:Real},
                    drift_block::Integer)
    taylorfdr!(fdr, spectrogram, (drift_block,))
end

# The Taylor tree algorithm always reads the raw spectrogram, so unlike
# `zdtfdr!` there is no "spectrogram already loaded" shortcut.
function taylorfdr!(fdr::AbstractMatrix, ::TaylorWorkspace, drift_blocks)
    error("taylorfdr! requires a spectrogram argument")
end

"""
    taylorfdr(spectrogram, drift_blocks) -> fdr

Compute the frequency drift rate matrix for the given `spectrogram` and
Taylor tree drift blocks using the Taylor tree algorithm.  `drift_blocks` may
be a single integer drift block or any collection of integer drift blocks.
The first (fastest changing) dimension of `spectrogram` is frequency and the
second dimension (slowest changing) is time, which must be at least 2.  If
the number of time samples `Nt` is not a power of 2, the data are treated as
if zero-padded in time to `Ntp = nextpow(2, Nt)` without materializing the
padding (see [`taylortree!`](@ref)).  The size of the returned frequency
drift rate matrix will be `(size(spectrogram, 1), length(drift_blocks) * Ntp)`.

Drift block `b` computes path sums for the `Ntp` normalized drift rates
`b + p/(Ntp-1)` for `p` in `0:(Ntp-1)`, i.e. the normalized drift rates from
`b` to `b+1` inclusive.  Use `taylorrates` to get the range(s) of drift
rates corresponding to the columns of the returned matrix.

Note that out-of-band path sums are zeroed (see [`taylortree!`](@ref)),
whereas `intfdr!` wraps paths around the edges of the spectrogram circularly.
"""
function taylorfdr(spectrogram::AbstractMatrix{<:Real}, drift_blocks)
    Nf, Nt = size(spectrogram)
    Ntp = nextpow(2, Nt)
    fdr = similar(spectrogram, Nf, Ntp * length(drift_blocks))
    taylorfdr!(fdr, spectrogram, drift_blocks)
end

function taylorfdr(spectrogram::AbstractMatrix{<:Real}, drift_block::Integer)
    taylorfdr(spectrogram, (drift_block,))
end

"""
    taylorrates(Nt, drift_block) -> range
    taylorrates(Nt, drift_blocks) -> vector of ranges

Return the range of *normalized* drift rates computed by the Taylor tree for
`Nt` time samples and drift block `drift_block`, or a `Vector` of such ranges
(one per entry of `drift_blocks`).  If `Nt` is not a power of 2 it is treated
as zero-padded in time to `Ntp = nextpow(2, Nt)` (see `taylortree!`).  Drift
block `b` covers the normalized drift rates from `b` to `b+1` inclusive in
steps of `1/(Ntp-1)`.  The result(s) match the column ordering of `taylorfdr`
and `taylorfdr!` output and, like the ranges returned by `batchrates`, are
suitable for passing to functions that accept normalized drift rates.
"""
function taylorrates(Nt::Integer, drift_block::Integer)
    Ntp = nextpow(2, Nt)
    Nrm1 = Ntp - 1
    nc1 = drift_block * Nrm1
    range(nc1 // Nrm1, step = 1 // Nrm1, length = Ntp)
end

function taylorrates(Nt::Integer, drift_blocks)
    [taylorrates(Nt, b) for b in drift_blocks]
end
