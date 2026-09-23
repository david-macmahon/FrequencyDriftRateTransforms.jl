# FrequencyDriftRateTransforms.jl

This package is part of a suite of packages that can be used in tandem to search
for, detect, and analyze narrow band signals in frequency-time spectrograms
produced by radio telescopes.  Searching for Doppler drifting narrow band
signals in radio telescope spectrograms is a technique often employed by
scientists engaged in the Search for Extraterrestrial Intelligence (SETI).
One of the first steps in the search process is to transform the frequency-time
matrix (i.e. spectrogram) into a frequency-drift rate matrix ("driftogram"?).
This package provides functions that perform this transform.

## Overview

The basic idea of the Doppler drift search is to sum the power along all
possible diagonal *drift lines* through a frequency-time spectrogram for a given
set of drift rates and then search for ones that contain power above a
specified threshold.  There is one drift line for each starting frequency (or
frequency channel) and drift rate combination.  The set of all starting
frequency and drift rate combinations can be represented as a matrix with one
axis being frequency (or starting channel) and the other axis being drift rate.
The value of each element of the matrix is the total power that was summed up
(integrated) along the drift line corresponding to the element's starting
frequency and drift rate.

This package transforms spectrograms to frequency drift rate (FDR) matrices for
a given set of drift rates.  The brute force approach is not very efficient, but
techniques such as the Taylor Tree algorithm minimize (if not eliminate)
redundant calculations and can be quite efficient.  This package implements the
Taylor Tree algorithm as well as a novel application of the Chirp-Z Transform to
perform the transformation from spectrogram to FDR matrix.  The latter approach
is referred to as the *Chirp-Z De-Doppler Transform*, or just *ZDT* for short.

Computing the FDR matrix with this package is just the first step of a Doppler
drift search.  The next step is finding the points in the FDR matrix that have
values greater a certain threshold.  Usually the threshold is given as a
*signal-to-noise ratio* (SNR), which is essentially a number of standard
deviations above the mean of the FDR values.  The `findprotohits` function can
be used to find such points.  FDR values can be normalized via the
`fdrnormalize!` function which subtracts the mean and divides by the standard
deviation.  This can be useful for plotting so that the displayed values are SNR
values, but for searching it is much more efficient to denormalize the single
SNR threshold via the `fdrdenormalize` function, which multiplies the SNR value
by the standard deviation of the FDR and then adds the mean of the FDR.  The
denormalized SNR value can then be used as the threshold with the non-normalized
FDR values.  This latter approach can also be performed as part of
`findprotohits` by passing `snr=true` as a keyword argument.

Elements of the FDR matrix with a value greater than a specified threshold form
a set of *proto-hits*.  Each proto-hit is not a detection of a unique Doppler
drifting signal because a single signal may be detected at more than one
frequency and drift rate combination.  Additional processing, such as
clustering, is required to determine which proto-hits can be considered to
represent a unique Doppler drifting signal.

## Additional/related packages

* DopplerDriftSearchTools.jl: As the name suggests, this package contains a
  variety of tools that are useful for performing Doppler drift searches:

  - Detection of unique signals, often referred to as *hits*, within the
    proto-hits found in an FDR matrix (or via other means).
  - Visualizing regions of an FDR matrix with proto-hits highlighted
  - Visualizing regions of a spectrogram with overlaid drift lines
  - The ability to read/write hits from/to Apache Arrow files
  - The ability to read turboSETI `.dat` files
  - Find matches and non-matches between sets of hits

* DopplerDriftSearchPipeline.jl: This package combines
  FrequencyDriftRateTransforms.jl and DopplerDriftSearchTools.jl with
  PoolQueues.jl to create a highly parallelized ZDT-based pipeline for
  performing a Doppler drift searches on a collection of input files.

## Core functionality

The current version of FrequencyDriftRateTransforms.jl contains functions for
creating a *frequency drift rate* (FDR) matrix from a spectrogram matrix for a
given set of drift rates.  The FDR matrix may be plotted using, for example, the
`heatmap` function from Plots.jl.  Such a plot is often called a *butterfly
plot* because Doppler drifting narrow band signals are associated with
characteristic structure in the plot resembling a butterfly.

Four different techniques for generating FDR matrices are supported:

- Integer shifting (brute force; `intfdr`)
- Taylor tree (recursive integer shift-and-sum; `taylorfdr`)
- Fourier domain shifting (`fftfdr`)
- Chirp-Z De-Doppler Transform (ZDT; `zdtfdr`)

The integer shifting technique employed by this package uses a brute force
approach.  It is intended to be illustrative rather than practical.

The Taylor tree technique computes the sums along the drift lines for `Ntp`
evenly spaced drift rates spanning a full unit of normalized drift rate (one
"drift block") in `O(Nf * Nt * log2(Ntp))` operations, where `Ntp` is the
number of time samples zero-padded up to the next power of 2.  It is based on
the reference implementation in
[seticore](https://github.com/lacker/seticore).  The CPU implementation is
generic Julia code; when CUDA.jl is loaded, a package extension provides
dedicated GPU kernels for the inner `taylortree!` step, which are considerably
faster than running the generic code on GPU arrays.

The Fourier domain shifting technique employed by this package is also more
illustrative than practical.  It computes each "drift rate spectrum" of the FDR
matrix independently.  Given how fast GPUs can perform many FFTs simultaneously,
this approach may have some practicality, but it is not as efficient as the
Chirp-Z De-Doppler Transform (ZDT).

The Chirp-Z De-Doppler Transform (ZDT) computes the FDR matrix en masse using
an input FFT, a Chirp-Z transform, and an output FFT.  The Chirp-Z transform
itself is implemented with a series of phase factor multiplications and FFTs.
The ZDT algorithm of this package can be used on a CPU or a GPU.  It must be
emphasized that the CPU code and GPU code are the same code; there are not
separate kernels for CPU vs GPU.  The ability to run the same code on CPU or GPU
is one of the many amazing features of Julia and CUDA.jl!  Note that on the
CPU, the FFTW plans used by the ZDT are created single-threaded (because
`FFTW.set_num_threads` sets process-global state, which this package does not
change silently); call `FFTW.set_num_threads(Threads.nthreads())` before
constructing a `ZDTWorkspace` to parallelize the CPU FFTs.

The ZDT can compute an FDR matrix spanning many drift rates in smaller pieces,
which can be very useful when working on a memory constrained device like a
GPU.  The ZDT imposes one constraint: it must be used with evenly spaced drift
rates.  This is rarely a problem in practice.

## Batched drift rate searches

The ZDT computes all `Nr` drift rates of a batch in one pass, so a wide drift
rate search is typically split into batches of manageable size (e.g. to fit in
GPU memory; the `estimate_memory` function estimates the required memory).
The `batchrates` function splits a physical drift rate range (in `Hz/s`) into
batches of evenly spaced normalized drift rates:

```julia
δhzps = foff / tsamp / (Nt - 1)  # drift rate step size in Hz/s
batches = batchrates(Nt, δhzps, rmin, rmax)
```

All batches share the same length and step size, so a single `ZDTWorkspace`
can be reused for every batch, with the `r0` keyword selecting the first rate
of each batch.  Each batch's FDR matrix can then be searched for proto-hits:

```julia
ws = ZDTWorkspace(spectrogram, first(batches))
for rates in batches
    fdr = zdtfdr(ws; r0 = first(rates))
    m, s = fdrstats(fdr)
    hijs = findprotohits(fdr, fdrdenormalize(5.0, m, s))
    # ... process hijs (e.g. cluster them into hits) ...
end
```

Note the use of `fdrdenormalize` to denormalize the SNR threshold (5 sigma
here) rather than normalizing the whole FDR matrix via `fdrnormalize!`; for
searching, denormalizing the single threshold value is more efficient (see
[`fdrdenormalize`](@ref)).

## Noise floor estimation

The mean and standard deviation used for thresholding (see [`fdrstats`](@ref))
are plain statistics, so a single bright signal inflates them and raises every
threshold.  The [`noisefloor`](@ref) function estimates the noise floor power
robustly instead: it models the data as the sum of two independent Gamma
distributed polarizations (the natural distribution of integrated power
samples) and anchors the estimate on signal-free quantiles (the `qlo`
quantile, 10% by default, and the median).  The estimated mean stays within a
few percent of the true noise floor even at contamination fractions of order
15% (where it biases high by ~3%), while the plain mean is already badly
biased at a 1% contamination fraction.  Pass `robust = true` to `fdrstats` to
use it for thresholding:

```julia
m, s = fdrstats(fdr; robust = true, k = k * Nt)
hijs = findprotohits(fdr, fdrdenormalize(5.0, m, s))
```

When `k` is known, `noisefloor` also reports the per-polarization mean powers
(`pol1`, `pol2`, `polratio`); without it, only the effective shape of the
summed polarizations is estimated.  See [`noisefloor`](@ref) for the full
description.

### The `k` convention

`k` is the per-polarization Gamma *shape* of a single data sample, and that
is always the value to pass, regardless of how many polarizations are
summed into the data (the two-polarization sum is part of the model, not
the `k` value).  A spectrometer output sample is the sum of `n_accum`
accumulated FFT frames, and each frame's `real^2 + imag^2` power
contributes 2 degrees of freedom (i.e. Gamma shape 1) per polarization, so
for filterbank power data

```julia
k = n_accum = abs(foff) * tsamp   # foff in Hz, tsamp in s
```

and for a Frequency-Drift-Rate matrix produced from `Nt` path-summed time
samples, `k = n_accum * Nt`.  The number of summed polarizations does *not*
enter `k`; it only shows up in the *reported* effective `shape`, which is
`k` when one polarization dominates and approaches `2k` for a balanced
Stokes I sum.  Genuinely single-polarization data needs no special
treatment either: it is the limit where one polarization's mean power is
zero, for which the estimate is exact and the split reports `pol1 ≈ mean`
and `pol2 ≈ 0`.

As a concrete instance, the Breakthrough Listen Voyager 2020 single coarse
channel file has `foff = 2.794 Hz` and `tsamp = 18.2536 s`, so
`n_accum = 51` and `2e6 * abs(foff[MHz]) * tsamp = 102`; pass `k = 51` for
the spectrogram and `k = 51 * 16 = 816` for its FDR matrices, whether using
single-polarization or Stokes I data.

### Per-polarization split accuracy

The split is computed from the estimated moments via
`(θ1 - θ2)² = 2·std²/k - (mean/k)²`, so its accuracy is limited by the
standard deviation estimate and degrades steeply toward balanced
polarizations (a small difference of large quantities).  The table below
gives the approximate number of samples required for the
larger-polarization fraction to be accurate to within the stated number of
percentage points (RMS; Monte Carlo measured at `k = 51` for pure Gamma
noise):

| split | N @ 2 pp | N @ 5 pp | N @ 10 pp | N @ 20 pp |
|-------|----------|----------|-----------|-----------|
| 100/0 | 2.9e3    | 4.6e2    | 1.2e2     | 2.9e1     |
| 90/10 | 6.6e3    | 9.9e2    | 2.5e2     | 6.1e1     |
| 80/20 | 1.0e4    | 1.4e3    | 3.3e2     | 8.3e1     |
| 70/30 | 1.3e4    | 1.8e3    | 4.5e2     | 1.1e2     |
| 60/40 | 3.9e4    | 6.0e3    | 1.5e3     | 3.7e2     |
| 50/50 | never¹   | 7.0e4    | 1.5e4     | 3.6e3     |

¹ A bias floor of about 2 pp (about 1 pp for imbalanced splits) does not
average down, so near-balanced splits are systematically limited; at exactly
50/50 roughly half of the runs report an unidentified split (`NaN`).  The
values are Monte Carlo estimates, good to ~30%.

### Choosing `qlo`

The `qlo` keyword (lower quantile paired with the median) trades
contamination robustness against clean-data efficiency: lower quantiles are
less affected by signal/RFI contamination (the contaminated samples sit in
the upper tail, and the quantile-value shift under contamination is
smallest where the Gamma density is steepest), while higher quantiles give
a more efficient spread estimate on clean data (the quantile correlation
with the median grows with the quantile index).  Monte Carlo at `k = 51`
gives a clean-data split error of 0.68/0.61/0.37 pp for `qlo` of
0.05/0.1/0.2, versus a split bias of +4.3/+5.3/+6.8 pp at a 10%
contamination fraction; the `mean` estimate is largely `qlo`-independent
since the median anchors it.  The default `qlo = 0.1` is a good compromise;
since RFI occupancy varies strongly between frequency bands, `qlo` can be
tuned per search.

### Worked example: Voyager 2020 single coarse channel

The bandpass of that file is flat (coefficient of variation ~3%) across the
central 80% of its 2^20 channels and rolls off toward the band edges (the
-3 dB point is essentially at the edge), so the central 80% needs no
flattening.  Running `noisefloor` with `k = 51` on the central 80% of the
band (about 13.4 million samples, unflattened) recovers a polarization
split of 64.3%/35.7%, in agreement with the 62%/38% measured directly from
the full-polarization version of the same observation; the residual ~2 pp
is real-data systematics (residual bandpass and RFI) on top of the ~0.6 pp
statistical error at this sample size.
