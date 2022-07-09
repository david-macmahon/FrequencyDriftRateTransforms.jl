"""
Circularly shift each column of `din` by an amount proportional to `rate` and
store the results in `dout`.  The first column is un-shifted (i.e. shifted 0).
"""
function intshift!(dout, din, rate)
    @assert size(din) == size(dout)
    for (i, (cin, cout)) in enumerate(zip(eachcol(din), eachcol(dout)))
        n = round(rate*(i-1))
        circshift!(cout, cin, n)
    end
    return dout
end

"""
Circularly shift each column of `din` by an amount proportional to `rate` and
return the results in a new Matrix similar to `din`.  The first column is
un-shifted (i.e. shifted 0).
"""
function intshift(din, rate)
    dout = similar(din)
    intshift!(dout, din, rate)
end

"""
Same as the `intfdr` function, but store the results in `fdr`, which is also
returned.  The size of `fdr` must be `(size(spectrogram,1), length(rates))`.
"""
function intfdr!(fdr, spectrogram, rates)
    Nf, Nt = size(spectrogram)
    Nr = length(rates)
    @assert size(fdr) == (Nf, Nr)
    # Create 3D work array so that each rate will get its own work Matrix.
    # This makes it possible to parallelize the for loop.
    work = similar(spectrogram, Nf, Nt, Nr)
    for (i,r) in enumerate(rates)
        @views sum!(fdr[:,i], intshift!(work[:,:,i], spectrogram, r))
    end
    return fdr
end

"""
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
