using FrequencyDriftRateTransforms
using Test

if dirname(something(Base.current_project(), "")) == @__DIR__
    using CUDA
else
    @info "Skipping CUDA tests: not running in the test environment (use Pkg.test())"
end

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
        @test fdshift(fftws, 1) ≈ dshift1_expected 
        @test fftfdr(fftws, -1:1) ≈ fdr_expected
    end

    @testset "zdtfdr [FFTW]" begin
        zdtws = ZDTWorkspace(d, -1:1)
        @test zdtfdr(zdtws) ≈ fdr_expected
    end

    if isdefined(Main, :CUDA)
        if CUDA.functional()
            @testset "zdtfdr [CUDA]" begin
                g = CuArray(d)
                gdtws = ZDTWorkspace(g, -1:1)
                @test Array(zdtfdr(gdtws)) ≈ fdr_expected
            end
        else
            @info "Skipping CUDA tests: no functional GPU available"
        end
    end

end;

if get(ENV, "FDR_HEAVY_TESTS", "0") == "1"
    @testset "heavy tests" begin
        include("heavytests.jl")
    end
else
    @info "Skipping heavy tests (set FDR_HEAVY_TESTS=1 to enable; downloads data)"
end

# end of runtests.jl