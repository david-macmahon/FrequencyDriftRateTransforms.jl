# FrequencyDriftRateTransforms.jl

[![Documentation](https://github.com/david-macmahon/FrequencyDriftRateTransforms.jl/actions/workflows/Documentation.yml/badge.svg)](https://github.com/david-macmahon/FrequencyDriftRateTransforms.jl/actions/workflows/Documentation.yml)
[![docs stable](https://img.shields.io/badge/docs-stable-8CA0B3.svg)](https://david-macmahon.github.io/FrequencyDriftRateTransforms.jl/stable/)

This package is part of a suite of packages that can be used in tandem to search
for, detect, and analyze narrow band signals in frequency-time spectrograms
produced by radio telescopes, a technique often employed by scientists engaged
in the Search for Extraterrestrial Intelligence (SETI).  It transforms
spectrograms into frequency-drift rate (FDR) matrices for a given set of drift
rates, e.g. via the *Chirp-Z De-Doppler Transform* (ZDT), on both CPU and GPU.

See the [documentation](https://david-macmahon.github.io/FrequencyDriftRateTransforms.jl/stable/)
for details.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/david-macmahon/FrequencyDriftRateTransforms.jl")
```
