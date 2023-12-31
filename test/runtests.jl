using DopplerDriftSearch
using Test

@testset verbose=true "simple tests" begin
    d = zeros(Float32, 3, 2)
    d[2,:] .= 1

    dshift1_expected = zeros(Float32, 3, 2)
    dshift1_expected[1,2] = 1
    dshift1_expected[2,1] = 1

    fdr_expected = Float32[
        0 0 1
        1 2 1
        1 0 0
    ]

    @testset "intfdr" begin
        @test intshift(d, 1) == dshift1_expected 
        @test intfdr(d, -1:1) == fdr_expected
    end

    @testset "fftfdr" begin
        fftws = fftfdr_workspace(d)
        @test fdshift(fftws, 1) ./ 3 ≈ dshift1_expected 
        @test fftfdr(fftws, -1:1) ./ 3 ≈ fdr_expected
    end

    @testset "zdtfdr [FFTW]" begin
        zdtws = ZDTWorkspace(d, -1:1)
        @test zdtfdr(zdtws) ./ 12 ≈ fdr_expected
    end

    if isdefined(Main, :CUDA)
        @testset "zdtfdr [CUDA]" begin
            g = CuArray(d)
            gdtws = ZDTWorkspace(g, -1:1)
            @test Array(zdtfdr(gdtws)) ./ 12 ≈ fdr_expected
        end
    end

end;