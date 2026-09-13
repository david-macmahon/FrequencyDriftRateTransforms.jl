# FrequencyDriftRateTransforms.jl

[![Test](https://github.com/david-macmahon/FrequencyDriftRateTransforms.jl/actions/workflows/Test.yml/badge.svg)](https://github.com/david-macmahon/FrequencyDriftRateTransforms.jl/actions/workflows/Test.yml)
[![Documentation](https://github.com/david-macmahon/FrequencyDriftRateTransforms.jl/actions/workflows/Documentation.yml/badge.svg)](https://github.com/david-macmahon/FrequencyDriftRateTransforms.jl/actions/workflows/Documentation.yml)
[![docs stable](https://img.shields.io/badge/docs-stable-8CA0B3.svg)](https://david-macmahon.github.io/FrequencyDriftRateTransforms.jl/stable/)

This package is part of a suite of packages that can be used in tandem to search
for, detect, and analyze narrow band signals in frequency-time spectrograms
produced by radio telescopes, a technique often employed by scientists engaged
in the Search for Extraterrestrial Intelligence (SETI).  It provides functions
to transform spectrograms into frequency-drift rate (FDR) matrices for a given
set of drift rates using a variety of techniques:

* Brute force discrete shift-and-sum (`intfdr`)
* Brute force FFT shift-and-sum (`fftfdr`)
* Taylor tree recursive discrete shift-and-sum (`taylorfdr`)
* Chirp-Z De-Doppler Transform, aka ZDT (`zdtfdr`)

See the [documentation](https://david-macmahon.github.io/FrequencyDriftRateTransforms.jl/stable/)
for details.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/david-macmahon/FrequencyDriftRateTransforms.jl")
```
