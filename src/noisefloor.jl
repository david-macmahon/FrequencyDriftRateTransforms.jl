using Statistics
using SpecialFunctions: gamma_inc, gamma_inc_inv

# The noise in an integrated power spectrogram (and in the FDR matrices
# produced from one) is well modeled as the sum of two independent Gamma
# distributed polarizations, `Gamma(k, θ1) + Gamma(k, θ2)`, with equal
# per-polarization Gamma shape `k` and possibly different mean powers
# `k * θi`.  The sum of Gammas with different scales is not itself Gamma, but
# its first two moments match an *effective* Gamma with shape
# `k_eff = mean² / std² ∈ [k, 2k]`, and the quantiles of the two-Gamma sum
# deviate from that effective Gamma's quantiles by at most a few percent
# (worst for `k = 1` with imbalanced polarizations).  The estimator below is
# therefore anchored on signal-free quantiles interpreted through the
# effective Gamma (whose shape is iterated to a fixed point), which keeps the
# mean estimate accurate to ~1-2% and the standard deviation to ~5% in the
# worst case, both negligible for "N sigma" thresholding.  When `k` is given,
# it is used to split the estimated moments into per-polarization mean
# powers.

# median / mean ratio of Gamma(k, 1)
_gam_med_mean(k) = gamma_inc_inv(k, 0.5, 0.5) / k

# (median - lower quantile) / standard deviation ratio of Gamma(k, 1)
_gam_med_qlo_sigma(k, qlo) =
    (gamma_inc_inv(k, 0.5, 0.5) - gamma_inc_inv(k, qlo, 1 - qlo)) / sqrt(k)

"""
    noisefloor(data; k=nothing, qlo=0.1, clip=4.0, refine=true) -> NamedTuple

Estimate the noise floor power of `data` (e.g. a power spectrogram or a
Frequency-Drift-Rate (FDR) matrix), which is modeled as the sum of two
independent Gamma distributed polarizations with equal per-polarization
Gamma shape `k` and possibly different mean powers.  Genuinely
single-polarization data is handled naturally as the limit where one
polarization's mean power is zero, for which the estimate is exact and the
split reports `pol1 ≈ mean` and `pol2 ≈ 0`.  The estimate is robust to
signals and RFI, which only add power: it is anchored on the `qlo`
quantile and the median of the data, so the mean stays accurate to within a
few percent even for contamination fractions of order 10% (the plain mean
and standard deviation break down with far less contamination).

The returned `NamedTuple` has fields:

- `mean`: estimated noise floor power (i.e. the mean of the noise).
- `std`: estimated noise standard deviation, `sqrt(k1 * θ1^2 + k2 * θ2^2)`,
  matching the `noisestats` "sigma" semantics for thresholding.
- `shape`: the *effective* Gamma shape `mean² / std²` of the summed
  polarizations used for the quantile conversions.  Under the model this
  lies between `k` (one polarization dominant) and `2k` (balanced sum of
  two polarizations, e.g. Stokes I), since `1/shape = ρ² + (1 - ρ)²` for
  power ratio `ρ`.  This is estimated from the data even when `k` is
  given; a value at or above `2k` indicates data less variable than the
  model allows (e.g. after bandpass flattening).
- `polratio`, `pol1`, `pol2`: estimated per-polarization mean powers and
  their ratio (larger first).  These are only identifiable when `k` is
  given.  Near-balanced polarizations (and small samples) yield `NaN`
  values, indicating that the split is not identified (the effective shape
  estimate reached the `2k` ceiling); `pol1 + pol2 = mean` holds only for
  identified splits, and resolving imbalances requires many samples.  See
  the [noise floor estimation section of the documentation](@ref
  "Noise floor estimation") for the shape convention, split accuracy
  requirements, and a worked example.

Keyword arguments:

- `k`: per-polarization Gamma shape of a single sample, and always the
  value to pass, regardless of how many polarizations are summed into the
  data (the two-polarization sum is part of the model, not the `k` value).
  For filterbank power data this is the number of accumulated FFT frames
  per output sample (`n_accum = abs(foff) * tsamp` in Hz·s), and for an FDR
  matrix produced from `Nt` time samples, `n_accum * Nt`.  If omitted, the
  effective shape is estimated from the data and the per-polarization
  split is not reported.
- `qlo`: lower quantile paired with the median for the spread estimate.
  Lower values (e.g. `0.05`) tolerate more signal/RFI contamination; higher
  values (e.g. `0.2`) are more efficient on clean data; `0.1` is a good
  compromise.  The contamination response of `mean` is largely independent
  of `qlo` since the median anchors it.
- `clip`: clipping threshold, in units of the estimated mean, for the
  optional clipped-mean refinement.
- `refine`: whether to refine the mean estimate with an iterated clipped
  mean (bias-corrected for the Gamma model).  This mainly improves the
  statistical efficiency for small samples; the quantile-anchored estimate
  is already unbiased.

The data is treated as a global ensemble; per-channel (bandpass) estimation
is not performed.  Degenerate data (all zeros, constant, or with a
non-positive median) yields `mean = mean(data)` and `std = Inf` (mirroring
the `noisestats` fallback semantics) with all other fields `nothing`.
"""
function noisefloor(data::AbstractArray{<:Real}; k=nothing, qlo=0.1, clip=4.0,
                    refine=true)
    k !== nothing && k <= 0 &&
        throw(ArgumentError("k must be positive"))
    !(0 < qlo < 0.5) &&
        throw(ArgumentError("qlo must be between 0 and 0.5"))
    qlo_val, q50 = fast_quantile(data, [qlo, 0.5])
    if !(q50 > 0) || !(qlo_val < q50)
        return (mean = Float64(mean(data)), std = Inf, shape = nothing,
                polratio = nothing, pol1 = nothing, pol2 = nothing)
    end

    # Iterate the effective shape and mean to a fixed point: the shape
    # follows from the moment relation `k_eff = mean² / std²` and the mean
    # and standard deviation follow from the quantiles via the effective
    # Gamma conversion factors.
    mom = _noise_moments(qlo_val, q50; qlo)
    mean_est = mom.mean
    std_est = mom.std
    shape = mom.shape

    if refine
        # Iterated clipped mean with the exact Gamma bias correction: for
        # `X ~ Gamma(shape, θ)` with `θ = mean/shape` and clip threshold
        # `s = clip * mean`, the survivor fraction is
        # `P(shape, clip * shape)` and `E[X * 1{X < s}]` is
        # `mean * P(shape + 1, clip * shape)`, so the debiased mean is the
        # empirical survivor mean times `P(shape, c) / P(shape + 1, c)`.
        # The survivor count and sum are fused into one data pass.
        for _ in 1:5
            s = clip * mean_est
            n, total = mapreduce(x -> x < s ? (1, Float64(x)) : (0, 0.0),
                                 (a, b) -> (a[1] + b[1], a[2] + b[2]), data;
                                 init = (0, 0.0))
            n == 0 && break
            mean_new, done = _noise_refine_step(mean_est, total / n, shape, clip)
            mean_est = mean_new
            done && break
        end
    end

    if k === nothing
        polratio = pol1 = pol2 = nothing
    else
        polratio, pol1, pol2 = _noise_pol(mean_est, std_est, k)
    end

    (mean = mean_est, std = std_est, shape, polratio, pol1, pol2)
end

# Moments (mean, standard deviation, effective Gamma shape) of the
# two-Gamma noise model from signal-free quantiles, iterating the shape to a
# fixed point (the quantile-to-moment conversion factors depend on the
# shape).  Pure host math on `(qlo_val, q50)`; shared by `noisefloor` and the
# batched per-band estimation of the CUDA extension.
function _noise_moments(qlo_val, q50; qlo)
    shape = 2.0
    mean_est = q50 / _gam_med_mean(shape)
    std_est = 0.0
    for _ in 1:50
        std_est = (q50 - qlo_val) / _gam_med_qlo_sigma(shape, qlo)
        shape_new = clamp(mean_est^2 / std_est^2, 1e-3, 1e8)
        mean_new = q50 / _gam_med_mean(shape_new)
        done = isapprox(shape_new, shape; rtol = 1e-8)
        shape, mean_est = shape_new, mean_new
        done && break
    end
    std_est = (q50 - qlo_val) / _gam_med_qlo_sigma(shape, qlo)
    (mean = mean_est, std = std_est, shape = shape)
end

# One clipped-mean refinement step: debias the empirical survivor mean for
# the Gamma model and test convergence.  Returns the new mean estimate and
# whether it converged.
function _noise_refine_step(mean_est, empirical, shape, clip)
    p0 = gamma_inc(shape, clip * shape)[1]
    p1 = gamma_inc(shape + 1, clip * shape)[1]
    mean_new = empirical * p0 / p1
    done = isapprox(mean_new, mean_est; rtol = 1e-6)
    return mean_new, done
end

# Per-polarization split from the moments: `θ1 + θ2 = mean/k` and
# `θ1² + θ2² = std²/k`, so `(θ1 - θ2)² = 2 * std²/k - (mean/k)²`.  A
# non-positive value (data less variable than the model allows, e.g.
# near-balanced polarizations with the shape estimate at the `2k` ceiling)
# leaves the split unidentified and is reported as `NaN`.  Pure host math.
function _noise_pol(mean_est, std_est, k)
    sθ = mean_est / k
    d2 = 2 * std_est^2 / k - sθ^2
    if d2 > 0
        d = sqrt(clamp(d2, 0.0, sθ^2))
        pol1 = k * (sθ + d) / 2
        pol2 = k * (sθ - d) / 2
        polratio = pol1 / pol2
        return polratio, pol1, pol2
    end
    return NaN, NaN, NaN
end

# (mean, std) projection of `noisefloor` used by `noisestats(robust = true)`.
function _noisefloor_stats(data; k = nothing, qlo = 0.1, clip = 4.0,
                           refine = true)
    nf = noisefloor(data; k, qlo, clip, refine)
    (mean = nf.mean, std = nf.std)
end
