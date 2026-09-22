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
[seticore](https://github.com/lacker/seticore) and, like the ZDT, uses the same
code on CPU and GPU.

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
is one of the many amazing features of Julia and CUDA.jl!

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
