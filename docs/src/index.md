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
`noisenormalize!` function which subtracts the mean and divides by the standard
deviation.  This can be useful for plotting so that the displayed values are SNR
values, but for searching it is much more efficient to denormalize the single
SNR threshold via the `noisedenormalize` function, which multiplies the SNR value
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

* FastQuantiles.jl and NoiseEstimators.jl: the exact-quantile and
  noise-estimation machinery used by this package (`fast_quantile`,
  `noisefloor`, `noisestats`, and the normalize/denormalize helpers are
  re-exported from them).  See their documentation for the quantile
  selection algorithm and the two-Gamma noise model.
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
of each batch.  Each batch's FDR matrix can then be searched for proto-hits
and clustered into hits:

```julia
ws = ZDTWorkspace(spectrogram, first(batches))
for rates in batches
    fdr = zdtfdr(ws; r0 = first(rates))
    m, s = noisestats(fdr)
    hijs = findprotohits(fdr, noisedenormalize(5.0, m, s))
    # ... process hijs (see "Finding hits" below) ...
end
```

Note the use of `noisedenormalize` to denormalize the SNR threshold (5 sigma
here) rather than normalizing the whole FDR matrix via `noisenormalize!`; for
searching, denormalizing the single threshold value is more efficient (see
`noisedenormalize`).  The [`findhits`](@ref) function (see
[Finding hits](@ref)) clusters the proto-hits into hits directly, taking the
SNR threshold and statistics itself.

## Noise floor estimation

The mean and standard deviation used for thresholding (see `noisestats`)
would be plain statistics, so a single bright signal inflates them and raises
every threshold.  The `noisefloor` function estimates the noise floor
power robustly instead: it models the data as the sum of two independent Gamma
distributed polarizations (the natural distribution of integrated power
samples) and anchors the estimate on signal-free quantiles (the `qlo` and
`qhi` quantiles, 10% and 50% by default, with an optional clipped-mean
refinement of the mean via `clip`).  The estimated mean stays within a
few percent of the true noise floor even at contamination fractions of order
15% (where it biases high by ~3%), while the plain mean is already badly
biased at a 1% contamination fraction.  `noisestats` is robust by default
(pass `robust = false` for the plain statistics):

```julia
m, s = noisestats(fdr; k = k * Nt)
hijs = findprotohits(fdr, noisedenormalize(5.0, m, s))
```

Per-channel noise statistics (e.g. to compensate for bandpass structure or
channel-dependent RFI) are supported via `chans_per_band`, which must evenly
divide the number of channels: the statistics are then estimated per band of
`chans_per_band` frequency channels and returned as vectors of length
`size(fdr, 1)`, which [`findhits`](@ref) consumes as
per-channel thresholds and normalizations:

```julia
m, s = noisestats(fdr; chans_per_band = 32, k = k * Nt)  # vectors
hits = findhits(fdr, 5.0, (m, s))
```

Banding is the robust choice whenever the noise power varies along the
frequency axis — e.g. power-level variations across a coarse channel's
passband that survive the analytic filter-response correction: one estimate
for the whole channel is then biased wherever the power differs, while
per-band estimates track the variation.  Pick the band width small enough
that the variation within a band is negligible, and large enough to pool
many samples (e.g. 16K channels, ≈ 48 kHz for a ~2.9 Hz channelization).
On CUDA, the robust per-band estimates are computed in a fixed handful of
batched device passes, at a cost independent of the number of bands; on the
host, each band is estimated independently.

Band-width-banded statistics aside, [`fdrstats`](@ref) provides the
FDR-specific *plain* statistics: because an FDR's drift-rate columns are
wrap-around sums of the same spectrogram (each column's sum is essentially
the spectrogram's total), any column's mean estimates the noise mean, and
the minimum-σ column is the least RFI-contaminated one, making its σ the
least RFI-skewed plain estimate.  These assumptions hold only for
FDR-like data; generic data should use `noisestats`, whose plain
mode uses ensemble statistics.  The robust mode is recommended in either
case, for which `fdrstats` and `noisestats` are identical.

When `k` is known, `noisefloor` also reports the per-component mean powers
(`pow1`, `pow2`; their ratio is simply `pow1 / pow2`); without it, only the
effective shape of the summed components is estimated.  See
`noisefloor` for the full description.  The exact-quantile and
noise-estimation machinery is provided by
[FastQuantiles.jl](https://david-macmahon.github.io/FastQuantiles.jl) and
[NoiseEstimators.jl](https://david-macmahon.github.io/NoiseEstimators.jl)
and re-exported here; their documentation covers the two-Gamma noise
model, the `k` convention (for filterbank power data
`k = n_accum = abs(foff) * tsamp`, and `k = n_accum * Nt` for an FDR
matrix produced from `Nt` path-summed time samples), per-component split
accuracy, choosing the quantiles, and a worked Breakthrough Listen Voyager 2020
example.

## Finding hits

The [`findhits`](@ref) function turns the proto-hits of an FDR matrix (see
[`findprotohits`](@ref)) into *hits*: the local maxima of the thresholded
region(s), one per unique signal candidate.  Proto-hits within a Chebyshev
distance of 2 indices of each other (bridging single-pixel gaps) belong to
the same region; the regions are found with a decreasing-value union-find
sweep, and each region is reported through its peak:

```julia
hits = findhits(fdr, 5.0)                       # 5-sigma threshold
hits = findhits(fdr, 5.0; min_prominence = 1.5) # + persistence filtering
hits.index        # Vector{CartesianIndex{2}} of hit peaks
hits.value        # peak value of each hit, in sigma units (z-score)
hits.prominence   # prominence of each hit, in sigma units
hits.nhits        # number of proto-hits in each hit's footprint
hits.lochan       # lowest frequency row (channel) of the footprint
hits.hichan       # highest frequency row (channel) of the footprint
hits.lorateidx    # lowest drift-rate column index of the footprint
hits.hirateidx    # highest drift-rate column index of the footprint
hits.hitwidth     # drift-axis run extent of the hit (see below)
```

Each hit also reports *footprint* info: the set of above-threshold
proto-hits its peak dominates (the whole connected region for region maxima;
the portion merged away at the retirement saddle for secondary peaks),
summarized by `nhits` and the frequency-row (`lochan`/`hichan`) and
drift-rate-column (`lorateidx`/`hirateidx`) extrema.  The `hitwidth` field
is the extent (in frequency rows) of the maximal chain of the footprint's
proto-hits in the hit's own drift-rate column, anchored at the hit's row,
where consecutive chain members are within `dist` rows (so the chain never
leaves the footprint; bridged gap rows count toward the extent).  It
measures how localized the hit is along the drift axis at its own frequency
— the narrow "waist" of the butterfly pattern — and an isolated hit has
`hitwidth = 1`.

Everything is expressed in *sigma* units by default: the threshold, the
optional `min_prominence`, and the returned `value`/`prominence` columns.
The normalization pair `(m, s)` is `noisestats(fdr)` by
default (the noise-floor statistics recommended for thresholding; see
[Noise floor estimation](@ref)) and can be passed explicitly as the third
positional argument — e.g. to share one pair across all batches of a
batched search, or `stats = (0, 1)` to work entirely in raw FDR value
units (in which case `threshold` and `min_prominence` must be raw values,
and the returned columns are raw as well).

The `min_prominence` keyword applies a persistence filter to the secondary
peaks of a region: a secondary peak (a local maximum that merges into a
higher peak as the threshold sweeps down) is reported only if it rises at
least `min_prominence` above the saddle at which it merges.  This recovers
distinct signals that are bridged by an above-threshold arm — e.g. the
X-shaped pattern a strong drifting signal leaves in an FDR matrix — and
suppresses the low-contrast wiggles along such arms.  With
`min_prominence = nothing` (the default), only the maximum of each region
is reported.  The maximum of each region is *always* reported and gets
`prominence = Inf` (it never merges into a higher peak), so every
above-threshold region yields exactly one hit at minimum, and
`hits.prominence .>= min_prominence` always reproduces the reported set.

For `CuArray`s, the thresholding runs on the GPU via 32x32 tile maxima (so
only tiles containing proto-hits are transferred) and the clustering runs
on the host.

## Fast quantiles

The noise floor estimate (and any other quantile-based statistic) is only as
fast as its quantiles, and `Statistics.quantile` copies the data
and partially sorts that copy single-threaded — about 95 s for a 4 GiB
array, which dwarfs every other step of a search pipeline.  The
`fast_quantile` function — provided by
[FastQuantiles.jl](https://david-macmahon.github.io/FastQuantiles.jl) and
re-exported here — computes the *same* quantiles bit-for-bit (same rank
arithmetic, interpolation, NaN handling, and result types, which the test
suite verifies against `Statistics.quantile`) without copying or sorting:
the handful of required order statistics are selected by iterative
histogram refinement over order-preserving unsigned integer keys (the
sign-flipped IEEE bit patterns of the values).  Each pass bins the keys
into 2048 bins to narrow the key interval containing each rank, and a bin
covering a single key resolves every rank it holds — three streaming
passes for `Float32` data (about six for `Float64`), with ties resolving
naturally and no sorted copy ever materialized.

```julia
q = fast_quantile(data, [0.1, 0.5])   # exact match to quantile(vec(data), [0.1, 0.5])
m = fast_quantile(data, 0.5)          # scalar probability gives a scalar
q1, q2 = fast_quantile(data, (0.1, 0.9))  # tuple probability gives a tuple
```

On the host, the histogram passes are multithreaded when `Threads.nthreads()`
is greater than one and the data is large (~3 s for 4 GiB with 16 threads,
versus ~95 s for `quantile`).  For `CuArray`s the CUDA extension runs the
same algorithm on the device, so the data never leaves the GPU: ~0.07 s for
4 GiB, and the number of requested quantiles is nearly free since they share
each pass's histograms.  The GPU method needs no scratch space proportional
to the data (unlike a device-side sort, which requires a full-size temporary
buffer).  Inputs with eltypes that have no order-preserving bit pattern fall
back to `Statistics.quantile`.

`noisefloor` uses `fast_quantile` internally for its signal-free quantiles.

