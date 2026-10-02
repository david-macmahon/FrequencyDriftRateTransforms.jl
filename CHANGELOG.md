# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.12.0] - 2026-10-02

### Changed

- `NoiseEstimators` compatibility bumped from 0.3 to 0.4, adopting its
  revised noise-floor API: estimates are anchored on the `qlo` and `qhi`
  quantiles (defaults 0.1 and 0.5, the median; the previous spread
  estimates are reproduced exactly), the clipped-mean refinement of the
  mean is now opt-in via `clip` (default `0`; previously on by default
  via `refine = true`, so default mean estimates shift slightly), and
  the new `qhi` and `clip` keywords pass through `noisestats` and
  `fdrstats` with `robust = true`.

## [v0.11.0] - 2026-10-01

### Added
- `findhits` now accepts a `min_relprominence` keyword, a relative
  companion to `min_prominence`: a secondary peak is reported only when
  its persistence reaches a fraction `min_relprominence` of its own peak
  height above the noise mean (`prominence/value >= min_relprominence`,
  both columns in sigma units of the hit's own channel statistics).
  Unlike the absolute `min_prominence`, this criterion is scale-free
  across brightness levels.  When both keywords are given, a secondary
  peak must satisfy both; region maxima are always reported.
- The ZDT postphase window accepts a `:binom5` preset: the 5-point
  binomial smoothing kernel `[1, 4, 6, 4, 1]/16`.
- Documentation of Gibbs ringing from the ZDT's band-limited
  fractional-drift interpolation and of the apodization remedy, in the
  extended help of `zdtfdr` (along with the window-function interface
  and kernel recipes).
- `fftw_set_num_threads` and `fftw_get_num_threads`: thin exported
  wrappers around FFTW's process-global thread controls.  Call
  `fftw_set_num_threads` before constructing a workspace; plans pick up
  the thread count at planning time.

## [v0.10.0] - 2026-09-29

### Added
- Dependencies on [FastQuantiles.jl](https://github.com/david-macmahon/FastQuantiles.jl)
  and [NoiseEstimators.jl](https://github.com/david-macmahon/NoiseEstimators.jl),
  whose `fast_quantile`, `noisefloor`, `noisestats`, `noisenormalize`,
  `noisenormalize!`, and `noisedenormalize` functions are re-exported here.

### Removed
- The exact-quantile and noise-estimation code that used to live in this
  package (it moved into FastQuantiles.jl v0.2.0 and NoiseEstimators.jl
  v0.3.0, respectively).

### Changed
- `fdrstats` is once again the FDR-specific plain statistics (mean of the
  first non-zero-σ column, σ of the minimum-σ column); generic data should
  use the re-exported `noisestats`.

## [v0.9.0] - 2025-03-26

### Added
- Preliminary window support to smooth the ZDT output (optional `w`
  parameter of `zdtfdr!`).

### Fixed
- Include the scale factor in the CUDA extension's `irfft` results.
- Ensure the destination is returned from `zdtfdr!` methods.

## [v0.8.0] - 2024-08-24

### Changed
- Finish the `brfft`-to-`irfft` conversion in `fftfdr_workspace` (host and
  CUDA paths).

## [v0.7.1] - 2024-06-08

### Changed
- Switch the CUDA extension from backward to inverse FFTs.

## [v0.7.0] - 2024-05-28

### Changed
- Switch from backward FFTs (`brfft`) to inverse FFTs (`irfft`).

## [v0.6.0] - 2024-01-18

### Added
- `findprotohits` function to find above-threshold points of an FDR matrix.

### Removed
- The Plots extension and related plotting code (moved to
  DopplerDriftSearchTools.jl).

## [v0.5.0] - 2024-01-17

### Changed
- Renamed the package from DopplerDriftSearch.jl to
  FrequencyDriftRateTransforms.jl.

## [v0.4.0] - 2024-01-16

### Added
- `fdrstats`, `fdrnormalize`, `fdrnormalize!`, and `fdrdenormalize` for
  thresholding FDR matrices in SNR units.
- `fdrsynchronize(::Type)` for GPU/CPU synchronization.
- `ZDTWorkspace` parameterization by the type of input data.
- Waterfall heatmap plotting from given values and drift-line style
  keyword arguments.

### Changed
- Only share the CUFFT workarea between C2C FFT plans.
- Be stricter about CUDA input/output rank (ndims).
- Allocate `ZDTWorkspace.V` after FFT planning.

## [v0.3.3] - 2024-01-10

### Added
- `batchrates` function for computing drift-rate batches.

### Changed
- Refactor `zdtfdr`/`zdtfdr!` methods to be batch friendly.

## [v0.3.2] - 2024-01-10

### Added
- Documenter.jl documentation (landing page and API reference).
- PlotsDopplerDriftSearchExt extension for waterfall plotting.
- CartesianIndex range related methods.

## [v0.3.1] - 2024-01-09

### Added
- `estimate_memory` function for estimating transform memory usage.
- Utility functions for drift-rate ranges.

## [v0.3.0] - 2024-01-09

### Added
- `sizeof(::ZDTWorkspace)` and a CuArray-aware `output!` method in the
  CUDA extension.

### Changed
- Invert the sign convention of drift rate.
- Change all `ZDop` occurrences to `ZDT`.
- Use non-auto-allocated CUFFT workareas and improve CUDA workarea
  sharing; remove type piracy for `plan_brfft` in the CUDA extension.
- Move the old test suite to `heavytests` and add a new fast `runtests`.

## [v0.2.1] - 2023-12-22

### Fixed
- Initialize the `ZDTWorkspace` `V` field in the constructor.

### Changed
- Update tests to use H5Zbitshuffle and update the CUDA compat entries.

## [v0.1.1] - 2023-12-22

### Added
- Initial release: FFT-based (`fftfdr`, `fftfdr_workspace`) and Chirp-Z
  Transform based (`zdtfdr`, `zdtfdr!` with `ZDTWorkspace` and
  `input!`/`preprocess!`/`convolve!`/`postprocess!`/`output!` stages)
  spectrogram-to-FDR-matrix transforms, `create_fdr` conveniences,
  `fdshift`/`fdshiftsum` (and bang) variants, `intshift`/`intshift!`, and
  CUDA support via a package extension.
