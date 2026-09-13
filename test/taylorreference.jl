# Independent reference implementation of the Taylor tree for cross-checking
# `taylorfdr`.  This is a deliberate literal transliteration of seticore's
# reference CUDA implementation:
#
#   taylorOneStepOneChannel from
#     https://github.com/lacker/seticore/blob/master/taylor.h
#   basicTaylorTree (the ping-pong driver) from
#     https://github.com/lacker/seticore/blob/master/taylor.cu
#
# which is itself based on Franklin Antonio's CudaTaylor5demo.cu from
# https://github.com/UCBerkeleySETI/dedopplerperf.
#
# The C code stores buffers as row-major [time_block][path_offset][freq]
# arrays, whose linear index ((time_block*path_length + path_offset)*Nf +
# chan) happens to equal the linear index of (chan, time_block*path_length +
# path_offset + 1) in a column-major Julia matrix, so the transliteration can
# use plain (f, t) indexing with t = tb*path_length + path_offset + 1.
#
# Unlike `taylorstep!`, the reference keeps the C code's weaker validity
# check (only the shifted read of the second half path must be in bounds), so
# it writes "doomed" paths (whose end frequency is out of band) and reads
# entries that were never written.  Buffers must therefore be fully defined
# (zero-initialized here), and comparisons against it are only meaningful on
# the *valid* wedge (path end frequency in bounds) -- exactly where
# `taylorfdr` must agree with it bitwise, since both perform identical
# chan_loating point additions in identical order.

function crefstep!(target, source, path_length, drift_block)
    Nf, Nt = size(source)
    ntb = Nt ÷ path_length
    L2 = path_length ÷ 2
    for tb in 0:(ntb - 1), path_offset in 0:(path_length - 1)
        half_offset = div(path_offset, 2)
        chan_shift = div(path_offset + 1, 2) + div(drift_block * path_length, 2)
        tcol = tb * path_length + path_offset + 1
        scol1 = tb * path_length + half_offset + 1
        scol2 = scol1 + L2
        for f in 1:Nf
            f2 = f + chan_shift
            1 <= f2 <= Nf || continue
            target[f, tcol] = source[f, scol1] + source[f2, scol2]
        end
    end
    return target
end

function creftree(spectrogram, drift_block)
    Nf, Nt = size(spectrogram)
    b1 = zeros(eltype(spectrogram), Nf, Nt)
    b2 = zeros(eltype(spectrogram), Nf, Nt)
    source = spectrogram
    target = b1
    path_length = 2
    while path_length <= Nt
        crefstep!(target, source, path_length, drift_block)
        source = target
        target = (target === b1) ? b2 : b1
        path_length *= 2
    end
    return source
end

"""
Check `taylorfdr(spec, drift_block)` against the seticore reference
(`creftree`): bitwise equality on the valid wedge and zeros on the invalid
wedge.  `spec` must have a power-of-2 number of time samples (for
non-power-of-2 counts, pass the zero-padded spectrogram, which `taylorfdr`
must then match exactly -- see the virtual padding tests).
"""
function checktaylor(spec, drift_block)
    Nf, Nt = size(spec)
    mine = taylorfdr(spec, drift_block)
    ref = creftree(spec, drift_block)
    ok = true
    for path_offset in 0:(Nt - 1)
        drift_channels = (Nt - 1) * drift_block + path_offset
        chan_lo = max(1, 1 - drift_channels)
        chan_hi = min(Nf, Nf - drift_channels)
        # Bitwise equality on the valid wedge (same additions, same order)
        ok &= mine[chan_lo:chan_hi, path_offset+1] == ref[chan_lo:chan_hi, path_offset+1]
        # Invalid wedge must be zeroed
        ok &= all(iszero, @view mine[1:min(chan_lo - 1, Nf), path_offset+1])
        ok &= all(iszero, @view mine[max(1, chan_hi + 1):Nf, path_offset+1])
    end
    return ok
end
