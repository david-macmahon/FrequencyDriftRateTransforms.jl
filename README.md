# DopplerDriftSearch.jl

Searching for Doppler drifting narrow band signals in a frequency-time
spectrogram is a technique often employed by scientists engaged in the Search
for Extraterrestrial Intelligence (SETI).  This package provides functions that
facilitate such a search.

The DopplerDriftSearchPipeline.jl package builds upon the Chirp-Z De-Doppler
Transform (ZDT) code of this package along with the data processing pipeline
enabling [PoolQueues.jl](https://github.com/david-macmahon/PoolQueues.jl)
package to provide a highly parallelized, GPU-enabled Doppler drift search
pipeline based on the ZDT.

## Core functionality

The current version of DopplerDriftSearch.jl contains functions for creating a
*frequency drift rate* (FDR) matrix from a spectrogram matrix for a given set of
drift rates.  The FDR matrix may be plotted using, for example, the `heatmap`
function from Plots.jl.  Such a plot is often called a *butterfly plot* because
Doppler drifting narrow band signals are associated with characteristic
structure in the plot resembling a butterfly. 

Three different techniques for generating FDR matrices are supported:

- Integer shifting (brute force)
- Fourier domain shifting
- Chirp-Z De-Doppler Transform (ZDT)

The integer shifting technique employed by this package uses a brute force
approach.  It is intended to be illustrative rather than practical.  A faster
variation of the integer shifting approach can be realized by using the Taylor
Tree algorithm to minimize redundant calculations, but this package does not
(yet) include such functionality (contributions are always welcome!).

The Fourier domain shifting technique employed by this package is also more
illustrative than practical.  It computes each "drift rate spectrum" of the FDR
matrix independently.  Given how fast GPUs can perform many FFTs simultaneously,
this approach may have some practicality, but it is not as efficient as the
Chirp-Z De-Doppler Transform (ZDT).

The Chirp-Z De-Doppler Transform (ZDT) computes the FDR matrix en masse using
an input FFT, a Chirp-Z transform, and an output FFT.  The Chirp-Z transform
itself is implemented with a series of phase factor multiplications and FFTs.
The ZDT algorithm of this package can be used on a CPU or a GPU.  In fact, the
vast majority of the code is directly usable on either the CPU or GPU.  The only
differences are in the FFT planning stage and in the output FFT stage where
things are done differently to minimize memory usage on the GPU.  It must be
emphsized that these differences are elective for memory footprint
optimizations.  The CPU code and GPU code are essentially the same code; there
are not separate kernels for CPU vs GPU.

The ZDT is also very amenable to computing an FDR matrix spanning many drift
rates in smaller pieces, which can be very useful when working on a memory
contrained device like a GPU.

## Additional functionality

This package also provides functionality to *normalize* an FDR matrix.  This
essentially converts each point in the FDR matrix into a *signal-to-noise* (SNR)
value indicating the ratio of integrated "signal" power to integrated "noise"
power for a given starting frequency and drift rate.
