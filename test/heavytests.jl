using DopplerDriftSearch
using Test

using Downloads
using HDF5
using H5Zbitshuffle
using FFTW

# Download test dataset and read specific range of frequencies (with known
# drifting signal)
voyager_url = "http://blpd14.ssl.berkeley.edu/voyager_2020/single_coarse_channel/single_coarse_guppi_59046_80036_DIAG_VOYAGER-1_0011.rawspec.0000.h5"
h5 = h5open(Downloads.download(voyager_url))
freq_range = range(659935, length=150)
spectrogram = h5["data"][freq_range,1,:]
rates = 0:-0.25:-5

@testset "DopplerDriftSearch.jl" begin
    fdr = intfdr(spectrogram, rates)
    peak = maximum(fdr)
    peak_idx = findfirst(==(peak), fdr)
    @test peak_idx == CartesianIndex(55, 11)

    fftws = fftfdr_workspace(spectrogram)
    ffdr = fftfdr(fftws, rates)
    pkval, pkidx = findmax(ffdr)
    @test pkidx == CartesianIndex(55, 11)

    # De-dopper spectrogram with known drift rate
    dedop = fdshift(fftws, -2.43)
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
