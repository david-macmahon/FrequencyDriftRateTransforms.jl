# DopplerDriftSearch

Searching for Doppler drifting narrow band signals in a frequency-time
spectrogram is a technique often employed by scientists engaged in the Search
for Extraterrestrial Intelligence (SETI).  This package provides functions that
facilitate such a search.

## Current functionality

The current version of DopplerDriftSearch contains functions for creating a
*frequency drift rate* (FDR) matrix from a spectrogram matrix for a given set of
drift rates.  The FDR matrix may be plotted using, for example, the `heatmap`
function from Plots.jl.  Such a plot is often called a *butterfly plot* because
Doppler drifting narrow band signals are associated with characteristic
structure in the plot resembling a butterfly. 

Two different techniques for generating FDR matrices are supported.  The
`intfdr` and `intfdr!` functions generate an FDR matrix by shifting each
spectrum by an integer number of frequency channels proportional to the drift
rates.  The `fftfdr` and `fftfdr!` functions generate an FDR matrix using
Fourier domain techniques to shift each spectrum by an arbitrary frequency
amount (i.e. by a non-integer number of frequency channels).

## Future functionality

Future functionality will include the ability to search the FDR for local peaks
indicating the best fit starting channel and drift rate of a Doppler drifting
narrow band signal.
