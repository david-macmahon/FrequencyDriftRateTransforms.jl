using FrequencyDriftRateTransforms
using Test

using DataDeps
using HDF5
using H5Zbitshuffle
using FFTW

# The test dataset is downloaded (and its SHA256 checksum verified) on first
# use, then cached in the DataDeps folder (typically ~/.julia/datadeps), so it
# is only ever downloaded once.  The always-accept setting skips DataDeps'
# interactive download prompt (e.g. for CI).
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
voyager_filename = "single_coarse_guppi_59046_80036_DIAG_VOYAGER-1_0011.rawspec.0000.h5"
DataDeps.register(DataDep(
    "voyager-2020-single-coarse-channel",
    "Voyager 2020 single coarse channel HDF5 spectrogram (Breakthrough Listen Public Data Archive)",
    "https://bldata.berkeley.edu/voyager_2020/single_coarse_channel/$(voyager_filename)",
    "a4f9d9da015b4d45054ef91a1d95dc27138e495d887f595970c13828a76c9412",
))

# Read specific range of frequencies (with known drifting signal)
h5 = h5open(joinpath(datadep"voyager-2020-single-coarse-channel", voyager_filename))
freq_range = range(659935, length=150)
spectrogram = h5["data"][freq_range,1,:]
rates = 0:0.25:5

@testset "FrequencyDriftRateTransforms.jl" begin
    fdr = intfdr(spectrogram, rates)
    peak = maximum(fdr)
    peak_idx = findfirst(==(peak), fdr)
    @test peak_idx == CartesianIndex(55, 11)

    fftws = FFTWorkspace(spectrogram)
    ffdr = fftfdr(fftws, rates)
    pkval, pkidx = findmax(ffdr)
    @test pkidx == CartesianIndex(55, 11)

    # De-doppler spectrogram with known drift rate
    dedop = fdshift(fftws, 2.43)
    # Find maximum value for each time sample
    peaks = maximum(dedop, dims=1)
    # All peaks should be in channel 55
    peak_idxs = map(((i,p),)->findfirst(==(p), dedop[:,i]), enumerate(peaks))
    @test all(peak_idxs .∈ Ref(55:56))

    zdtws = ZDTWorkspace(spectrogram, rates)
    zfdr = zdtfdr(zdtws)
    pkval, pkidx = findmax(zfdr)
    @test pkidx == CartesianIndex(55, 11)
end;
