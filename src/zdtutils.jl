"""
Calculate the smallest FFT size greater than `Nt + Nr - 1` and having only
factors from the given `factors`.
"""
function calcNl(Nt, Nr, factors::Union{Tuple,AbstractVector}=(2,3,5))
    nextprod(factors, Nt + Nr - 1)
end

"""
Compute the largest `Nr` that fits in the smallest FFT size greater than `Nt +
Nr - 1` and having only factors from the given `factors`.
"""
function growNr(Nt::Integer, Nr::Integer, factors::Union{Tuple,AbstractVector}=(2,3,5))
    return calcNl(Nt, Nr, factors) - Nt + 1
end

"""
    estimate_memory(Nf, Nt, Nr, Ni=1, No=1; factors=(2,3,5))

Estimate the number of bytes required on the GPU for the given paramters:
- `Nf`: The number of frequency channels in the input spectrogram
- `Nt`: The number of time samples in the input spectrogram
- `Nr`: The number of drift rates to search (per batch)
- `Ni`: The number of input buffers (outside ZDTWorkspace)
- `No`: The number of output buffers (outside ZDTWorkspace)
- `factors`: Assume FFT sizes are restricted to products of `factors` (see
  `calcNl()` for more details)
This function may be more accurate for GPU memory estimates than CPU memory
estimates.
"""
function estimate_memory(Nf, Nt, Nr, Ni=1, No=1;
                         factors::Union{Tuple,AbstractVector}=(2,3,5))
    Nl = calcNl(Nt, Nr, factors)

    sizeof(Float32) * (
        Nf * Nt * (Ni+1) +   # Input buffers (`Ni` outside + 1 inside workspace)
        Nf * Nl *  3     +   # ZDT buffers and FFT work area (loosely)
        Nf * Nr *  No        # Output buffers (`No` outside workspace)
    )
end

"""
    driftrates(zdtws::ZDTWorkspace; r0=zdtws.r0)

Return the Range of normalized drift rates that `zdtws` has been configured to
use.  An alternate *normalized* `r0` value may be given.
"""
function driftrates(zdtws::ZDTWorkspace, r0=zdtws.r0)
    range(r0, step=zdtws.δr, length=zdtws.Nr)
end
