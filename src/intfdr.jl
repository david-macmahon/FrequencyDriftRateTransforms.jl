"""
    intshift!(dest, src, rate) -> dest

Circularly shift each column of `src` by an amount proportional to `rate` and
store the results in `dest`.  The first column is un-shifted (i.e. shifted 0).
"""
function intshift!(dest, src, rate)
    size(dest) == size(src) ||
        throw(ArgumentError("dest and src must have the same size (got $(size(dest)) and $(size(src)))"))
    for (i, (cin, cout)) in enumerate(zip(eachcol(src), eachcol(dest)))
        n = round(rate*(i-1))
        circshift!(cout, cin, -n)
    end
    return dest
end

"""
    intshift(src, rate) -> dest

Circularly shift each column of `src` by an amount proportional to `rate` and
return the results in a new Matrix similar to `src`.  The first column is
un-shifted (i.e. shifted 0).
"""
function intshift(src, rate)
    dest = similar(src)
    intshift!(dest, src, rate)
end

"""
    intfdr!(fdr, spectrogram, rates) -> fdr

Same as the `intfdr` function, but store the results in `fdr`, which is also
returned.  The size of `fdr` must be `(size(spectrogram,1), length(rates))`.
"""
function intfdr!(fdr, spectrogram, rates)
    Nf, Nt = size(spectrogram)
    Nr = length(rates)
    size(fdr) == (Nf, Nr) ||
        throw(ArgumentError("fdr must have size ($Nf, $Nr) (got $(size(fdr)))"))
    fill!(fdr, zero(eltype(fdr)))
    # Accumulate the circularly shifted columns of `spectrogram` directly
    # into `fdr` instead of materializing a shifted copy per rate, which
    # keeps the extra memory at O(Nf) instead of O(Nf*Nt*Nr).  Each column
    # is shifted by `round(r*(j-1))` channels and added as two ranged adds.
    for j in axes(spectrogram, 2)
        for (i, r) in enumerate(rates)
            n = mod(round(Int, r*(j-1)), Nf)
            if n == 0
                fdr[:, i] .+= @view spectrogram[:, j]
            else
                fdr[1:Nf-n, i] .+= @view spectrogram[n+1:Nf, j]
                fdr[Nf-n+1:Nf, i] .+= @view spectrogram[1:n, j]
            end
        end
    end
    return fdr
end

"""
     intfdr(spectrogram, rates) -> fdr

Compute the frequency drift rate matrix for the given `spectrogram` and `rates`
values by shifting each frequency spectrum by an integer numbers of frequency
channels.  The first (fastest changing) dimension of `spectrogram` is frequency
and the second dimension (slowest changing) is time.  The size of the returned
frequency drift rate matrix will be `(size(spectrogram,1), length(rates))`.
"""
function intfdr(spectrogram, rates)
    Nf, Nt = size(spectrogram)
    Nr = length(rates)
    fdr = similar(spectrogram, Nf, Nr)
    intfdr!(fdr, spectrogram, rates)
end
