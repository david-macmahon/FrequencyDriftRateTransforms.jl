"""
    batchrates(Nt, δhzps, r1, r2=-r1; Nrb=360) -> normalized_drift_rate_ranges

Return an `Vector` of `Range` instances each of which is a "batch" of equally
spaced *normaized* drift rates (suitable for passing to the `ZDTWorkspace`
constructor).  In total, the `Range` instances span the given *NON-normalized*
drift rates `r1` and `r2` (in `Hz/s`) in steps of `δhzps` (in `Hz/s`) for data
with `Nt` time samples.  Often `δhzps` is calculated as `foff/tsamp/(Nt-1)`,
where `foff` is the channel width in `Hz` and `tsamp` is the  interval between
consecutive time samples in seconds.

Each `Range` in the returned `Vector` has the same (normalized) step size and
length that will be close(ish) to `Nrb`.  The number of drift rates per batch
may be slightly larger than `Nrb`.  To ensure a single batch, pass
`Nrb=typemax(Int)`, but beware of memory requirements.

!!! note
    `δhzps`, `r1` and `r2` must be given in `Hz/s`.
"""
function batchrates(Nt::Integer, δhzps, r1, r2=-r1; Nrb=360)
    if r2 < r1
        r1, r2 = r2, r1
    end

    # Calculate number of channels that span the requested rate range
    # We will do one drift rate for each spanned channel
    nc1 = round(Int, r1 / δhzps)
    nc2 = round(Int, r2 / δhzps)
    Nrspan = abs(nc2 - nc1) + 1
    #@show r1 r2 δhzps nc1 nc2 Nrspan

    # Limit Nrb to one batch of Nrspan
    Nrb = min(Nrb, Nrspan)
    #@info "Nrb brfore growNr: $Nrb"

    # Calculate max Nrb (Nr per batch) given nominal Nrb and Nt
    Nrb = growNr(Nt, Nrb)
    #@info "Nrb after growNr: $Nrb"

    # Calculate next multiple of Nrb that is greater than Nrspan
    Nr = cld(Nrspan, Nrb) * Nrb
    #@info "Nr that is a multiple of Nrspan: $Nr"

    # Calculate number of "bonus" rates
    Nrbonus = Nr - Nrspan
    #@show Nrbonus

    nc1 -= round(Int, sign(δhzps) * Nrbonus/2, RoundFromZero)
    nc2 += round(Int, sign(δhzps) * Nrbonus/2, RoundToZero)
    #@show nc1 nc2 Nt Nr Nrb

    rs = range(nc1//(Nt-1), stop=nc2//(Nt-1), length=Nr)
    #@show rs
    @assert length(rs) % Nrb == 0 "number of rates ($(
                                   length(rs)) is not divisible by Nrb ($Nrb)"

    collect(Iterators.partition(rs, Nrb))
end
