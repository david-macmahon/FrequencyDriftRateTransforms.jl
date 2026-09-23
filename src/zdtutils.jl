"""
    calcNl(Nt, Nr, factors=(2,3,5)) -> Nl

Calculate the smallest FFT size greater than `Nt + Nr - 1` and having only
factors from the given `factors`.
"""
function calcNl(Nt, Nr, factors::Union{Tuple,AbstractVector}=(2,3,5))
    nextprod(factors, Nt + Nr - 1)
end

"""
    growNr(Nt, Nr, factors=(2,3,5)) -> newNr

Compute `newNR` that is the largest `Nr` that fits in the smallest FFT size
greater than `Nt + Nr - 1` and having only factors from the given `factors`.
"""
function growNr(Nt::Integer, Nr::Integer, factors::Union{Tuple,AbstractVector}=(2,3,5))
    return calcNl(Nt, Nr, factors) - Nt + 1
end

"""
    estimate_memory(Nf, Nt, Nr, Ni=1, No=1; factors=(2, 3, 5)) -> Int

Estimate the number of bytes required for a ZDT drift rate search with the
given parameters, assuming `Float32` array elements:
- `Nf`: The number of frequency channels in the input spectrogram
- `Nt`: The number of time samples in the input spectrogram
- `Nr`: The number of drift rates to search (per batch)
- `Ni`: The number of spectrogram-sized input buffers outside the workspace
- `No`: The number of FDR-matrix-sized output buffers outside the workspace
- `factors`: Assume FFT sizes are restricted to products of `factors` (see
  [`calcNl`](@ref))

The estimate covers the [`ZDTWorkspace`](@ref) buffers (including FFT work
areas) plus the `Ni` input and `No` output buffers.  It is intended for
planning memory-constrained (e.g. GPU) searches and may be more accurate for
GPU memory than for CPU memory, but it applies wherever the arrays reside.
"""
function estimate_memory(Nf, Nt, Nr, Ni=1, No=1;
                         factors::Union{Tuple,AbstractVector}=(2,3,5))
    Nl = calcNl(Nt, Nr, factors)

    # Ni spectrogram buffers outside workspace,
    # plus 1 buffer in workspace (`F`),
    # plus 2 real FFT work areas
    Ni += 3

    # Nz is number of Nf x Nl complex buffers in the workspace (`Y` and `V`)
    # plus one complex FFT work area
    Nz = 3

    sizeof(Float32) * (
        (Nf * Nt) * Ni +          # Input buffers and real FFT work areas
        (Nf * Nl) * Nz +          # ZDT complex buffers and FFT work area (loosely)
        (Nf * Nr) * (No + 1)      # Output buffers plus half-spectrum buffer (`Ys2`)
    )
end

"""
    driftrates(zdtws::ZDTWorkspace, r0=zdtws.r0) -> Range

Return the Range of normalized drift rates that `zdtws` has been configured to
use.  An alternate *normalized* `r0` value may be given to override
`zdtws.r0`.
"""
function driftrates(zdtws::ZDTWorkspace, r0=zdtws.r0)
    range(r0, step=zdtws.δr, length=zdtws.Nr)
end
