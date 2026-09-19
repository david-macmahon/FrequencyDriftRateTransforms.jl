using FrequencyDriftRateTransforms
using Test
using Statistics
using DataDeps

if dirname(something(Base.current_project(), "")) == @__DIR__
    using CUDA
else
    @info "Skipping CUDA tests: not running in the test environment (use Pkg.test())"
end

include("taylorreference.jl")

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

    # Larger shared test data: Nt=8 (power of 2, staircase matches intfdr's
    # rounding) and Nt=16 (for non-power-of-2/virtual-padding tests)
    Nf, Nt = 64, 8
    d2 = Float32[sin(0.7*i + 0.3*j) * cos(0.1*i*j) for i in 1:Nf, j in 1:Nt]
    d3 = Float32[sin(0.7*i + 0.3*j) * cos(0.1*i*j) for i in 1:Nf, j in 1:16]

    @testset "taylorfdr" begin
        # Nt=2, so drift blocks -1 and 0 cover rates -1:1 (rate 0 twice)
        taylor_expected = Float32[
            0 0 0 1
            1 2 2 1
            1 0 0 0
        ]
        @test taylorfdr(d, -1:0) == taylor_expected
        @test taylorfdr(d, -1) == taylor_expected[:, 1:2]
        @test taylorfdr(d, 0) == taylor_expected[:, 3:4]

        # Requires at least 2 time samples
        @test_throws ArgumentError taylorfdr(zeros(Float32, 8, 1), 0)

        # Compare against intfdr.  Nt=8 is used because the Taylor tree's
        # staircase paths match intfdr's per-step rounded paths exactly for
        # Nt <= 8.  The Taylor tree zeroes paths that leave the frequency
        # band, whereas intfdr wraps them around circularly, so only compare
        # where paths stay in bounds.
        # Non-power-of-2 Nt is handled via virtual zero padding to
        # Ntp = nextpow(2, Nt); results must match actually padding the data
        for Nt2 in (3, 5, 6, 7, 9, 12, 15)
            Ntp = nextpow(2, Nt2)
            spec = d3[:, 1:Nt2]
            padded = hcat(spec, zeros(Float32, Nf, Ntp - Nt2))
            @test taylorfdr(spec, -1:1) == taylorfdr(padded, -1:1)
        end
        @test size(taylorfdr(d3[:, 1:5], 0)) == (Nf, 8)

        for b in -1:1
            tf = taylorfdr(d2, b)
            @test size(tf) == (Nf, Nt)
            ifr = intfdr(d2, taylorrates(Nt, b))
            for path_offset in 0:Nt-1
                drift_channels = (Nt - 1) * b + path_offset
                chan_lo = max(1, 1 - drift_channels)
                chan_hi = min(Nf, Nf - drift_channels)
                @test tf[chan_lo:chan_hi, path_offset+1] ≈ ifr[chan_lo:chan_hi, path_offset+1]
                @test all(iszero, @view tf[1:chan_lo-1, path_offset+1])
                @test all(iszero, @view tf[max(1, chan_hi+1):Nf, path_offset+1])
            end
            fdr2 = create_fdr(d2, Nt)
            taylorfdr!(fdr2, TaylorWorkspace(d2), d2, b)
            @test fdr2 == tf
        end

        # Columns are grouped by drift block in the given order
        tf3 = taylorfdr(d2, -1:1)
        @test size(tf3) == (Nf, 3Nt)
        @test tf3[:, 1:Nt] == taylorfdr(d2, -1)
        @test tf3[:, Nt+1:2Nt] == taylorfdr(d2, 0)
        @test tf3[:, 2Nt+1:3Nt] == taylorfdr(d2, 1)

        @test taylorrates(8, 0) == range(0//7, step=1//7, length=8)
        @test collect(taylorrates(8, -2)) == collect(range(-2, step=1//7, length=8))
    end

    @testset "fdrstats" begin
        # taylorfdr drift blocks with out-of-band drift produce all-zero
        # columns; fdrstats must ignore them (fdrnormalize! used to divide by
        # zero, producing NaNs)
        spec = Float32[mod(1013*i*i + 7*i*j + 61*j*j, 1001) for i in 1:37, j in 1:64]
        fdr = taylorfdr(spec, 0)   # top 27 columns are all zero
        fdr1 = taylorfdr(spec, 1)  # every column is all zero
        m, s = fdrstats(fdr)
        @test 0 < s < Inf
        @test fdrstats([fdr, fdr1]) == (m, s)
        @test all(isfinite, fdrnormalize!(fdr))

        @test fdrstats(fdr1) == (0.0f0, Inf)
        @test fdrnormalize!(fdr1) == zeros(37, 64)
        @test fdrdenormalize(5.0, fdr1) == Inf
        @test isempty(findprotohits(fdr1, 5.0; snr=true))

        # Iterable-of-matrices variants
        @test fdrstats([fdr1]) == (0.0f0, Inf)
        fdrs = [copy(fdr1), copy(fdr)]
        fdrnormalize!(fdrs)
        @test all(isfinite, fdrs[1]) && all(isfinite, fdrs[2])
        @test fdrdenormalize(5.0, [copy(fdr1)]) == Inf
        @test isempty(findprotohits([copy(fdr1)], 5.0; snr=true))

        # The mean comes from the first column with data, even when earlier
        # columns/matrices are all zero
        hand = zeros(Float32, 4, 8)
        hand[:, 3] .= 1.0f0:4.0f0
        m2, s2 = fdrstats(hand)
        @test m2 == 2.5f0
        @test s2 ≈ std(1.0f0:4.0f0)
        @test fdrstats([copy(fdr1), taylorfdr(spec, 0)]) == (m, s)
        @test fdrstats(fill(3.0f0, 4, 8)) == (3.0f0, Inf)
    end

    @testset "taylorstep!" begin
        # One step of path_length 2 against a direct reference, with and
        # without an explicit Nt
        for b in (-1, 0, 1)
            buf = zeros(size(d2))
            taylorstep!(buf, d2, 2, b)
            expected = zeros(size(d2))
            for tb in 0:3, path_offset in 0:1, f in 1:Nf
                shift = (path_offset + 1) >> 1 + b
                drift_channels = path_offset + b
                (1 <= f + shift <= Nf && 1 <= f + drift_channels <= Nf) || continue
                expected[f, 2tb+path_offset+1] = d2[f, 2tb+1] + d2[f+shift, 2tb+2]
            end
            @test buf == expected
            buf2 = zeros(size(d2))
            taylorstep!(buf2, d2, 2, b, Nt)
            @test buf2 == buf
        end

        # Three steps of the tree reproduce intfdr's path sums for the same
        # rates (Nt=8, so the staircase matches intfdr's rounding)
        b1 = zeros(size(d2))
        taylorstep!(b1, d2, 2, 0)
        b2 = zeros(size(d2))
        taylorstep!(b2, b1, 4, 0)
        b3 = zeros(size(d2))
        taylorstep!(b3, b2, 8, 0)
        ifr = intfdr(d2, taylorrates(Nt, 0))
        for path_offset in 0:Nt-1
            @test         b3[1:Nf-path_offset, path_offset+1] ≈ ifr[1:Nf-path_offset, path_offset+1]
        end

        # Virtual padding: stepping on Nt=5 real columns with a logical
        # count of 8 matches stepping on materialized zero padding
        spec5 = d3[:, 1:5]
        v1 = zeros(Float32, Nf, 8)
        taylorstep!(v1, spec5, 2, 0, 5)
        m1 = zeros(Float32, Nf, 8)
        taylorstep!(m1, hcat(spec5, zeros(Float32, Nf, 3)), 2, 0)
        @test v1 == m1
        v2 = zeros(Float32, Nf, 8)
        taylorstep!(v2, v1, 4, 0, 5)
        m2 = zeros(Float32, Nf, 8)
        taylorstep!(m2, m1, 4, 0)
        @test v2 == m2

        # Argument checking
        @test_throws ArgumentError taylorstep!(zeros(8, 8), zeros(8, 8), 3, 0)
        @test_throws ArgumentError taylorstep!(zeros(8, 8), zeros(8, 8), 16, 0)
        @test_throws ArgumentError taylorstep!(zeros(8, 6), zeros(8, 6), 4, 0)
        @test_throws ArgumentError taylorstep!(zeros(8, 8), zeros(8, 8), 2, 0, 16)
        @test_throws ArgumentError taylorstep!(zeros(8, 8), zeros(8, 5), 4, 0, 5)
    end

    @testset "taylorfdr vs seticore reference" begin
        # Cross-check taylorfdr against a literal transliteration of
        # seticore's taylorOneStepOneChannel/basicTaylorTree (see
        # taylorreference.jl).  Nt=16 and up are the only checks that pin the
        # Taylor staircase geometry, which legitimately differs from
        # intfdr's per-step rounded lines for Nt > 8.
        for (Nf, Nt) in ((37, 2), (17, 4), (23, 8), (19, 16), (13, 32), (37, 64))
            spec = Float64[mod(1013*i*i + 7*i*j + 61*j*j, 1001) - 500 for i in 1:Nf, j in 1:Nt]
            @testset "Float64 Nf=$Nf Nt=$Nt" begin
                for b in -2:2
                    @test checktaylor(spec, b)
                end
            end
        end

        # BigInt: exact sums on the valid wedge, zeros on the invalid wedge,
        # and never any reading of unwritten (undefined) entries
        for (Nf, Nt) in ((19, 16), (37, 64))
            spec = [BigInt(mod(1013*i*i + 7*i*j + 61*j*j, 1001) - 500) for i in 1:Nf, j in 1:Nt]
            @testset "BigInt Nf=$Nf Nt=$Nt" begin
                for b in -2:2
                    @test checktaylor(spec, b)
                end
            end
        end

        # Non-power-of-2 Nt (virtual zero padding): the padded result must
        # match the reference computed on the materialized zero-padded
        # spectrogram, and the unpadded spectrogram must match the padded one
        for (Nf, Nt) in ((19, 3), (13, 5), (37, 6), (19, 7), (13, 9), (23, 12), (37, 33))
            Ntp = nextpow(2, Nt)
            spec = Float64[mod(1013*i*i + 7*i*j + 61*j*j, 1001) - 500 for i in 1:Nf, j in 1:Nt]
            padded = hcat(spec, zeros(Nf, Ntp - Nt))
            @testset "virtual-pad Nf=$Nf Nt=$Nt" begin
                for b in -2:2
                    @test checktaylor(padded, b)
                    @test taylorfdr(spec, b) == taylorfdr(padded, b)
                end
            end
        end
    end

    @testset "zdtfdr windows" begin
        zdtws = ZDTWorkspace(d2, -1:0.5:1)
        rect = zdtfdr(zdtws)
        # Rectangular window via explicit symbol and function
        @test zdtfdr(:rect, zdtws) == rect
        @test zdtfdr((n, N) -> 1, zdtws) == rect
        # Hamming window is a circular 3-point convolution with kernel
        # [b/2, a, b/2] along frequency (see postphase extended help)
        a, b = 0.53836, 0.46164
        ham = zdtfdr(:hamming, zdtws)
        @test ham ≈ a .* rect .+ b ./ 2 .*
                    (circshift(rect, (1, 0)) .+ circshift(rect, (-1, 0)))
        # The :hamming symbol matches the equivalent function window
        @test zdtfdr((n, N) -> a + b * cospi(2n/N), zdtws) == ham
        # In-place variant with window
        dest = create_fdr(d2, 5)
        @test zdtfdr!(:hamming, dest, zdtws) === dest
        @test dest ≈ ham
        # Unsupported window type
        @test_throws ErrorException zdtfdr(:nope, zdtws)
    end

    @testset "batchrates" begin
        # Needs at least 2 time samples (like taylorfdr)
        @test_throws ArgumentError batchrates(1, 0.1, 1.0)

        # Unordered endpoints give the same batches
        @test batchrates(64, 1/63, 2.0, -2.0) == batchrates(64, 1/63, -2.0, 2.0)

        # Single batch with bonus rates distributed symmetrically
        brs = batchrates(64, 1/63, 2.0, -2.0)
        @test length(brs) == 1
        @test brs[1] == range(-128//63, 128//63, length=257)

        # Multiple equal-length batches whose concatenation is an
        # ascending range covering the requested span (plus symmetric
        # bonus rates bounded by half a batch length per side)
        for (Nt, δhzps, r1, r2, Nrbkw) in ((64, 1/63, 10.0, -10.0, 360),
                                           (128, 1/127, 0.5, 4.5, 64),
                                           (32, 2.0, -20.0, 20.0, 16))
            brs = batchrates(Nt, δhzps, r1, r2; Nrb=Nrbkw)
            allrs = vcat(collect.(brs)...)
            Nrb = length(first(brs))
            @test length(brs) > 1
            @test all(b -> length(b) == Nrb, brs)
            @test rem(length(allrs), Nrb) == 0
            @test issorted(allrs)
            nc1 = round(Int, min(r1, r2)/δhzps)
            nc2 = round(Int, max(r1, r2)/δhzps)
            @test allrs[1] <= nc1//(Nt-1)
            @test allrs[end] >= nc2//(Nt-1)
            @test nc1 - allrs[1]*(Nt-1) <= cld(Nrb, 2)
            @test allrs[end]*(Nt-1) - nc2 <= cld(Nrb, 2)
        end

        # Nrb=typemax(Int) forces a single batch
        @test length(batchrates(64, 1/63, 10.0, -10.0; Nrb=typemax(Int))) == 1
    end

    @testset "zdtutils" begin
        @test calcNl(64, 21) == 90
        @test calcNl(17, 16) == 32  # Nt + Nr - 1 is exactly 32
        @test calcNl(10, 10, (2,)) == 32
        @test calcNl(64, 21, (2,3,5,7)) == 84
        @test growNr(64, 21) == 27
        @test growNr(16, 16) == 17
        @test growNr(10, 10, (2,)) == 23
        # Growing Nr to the full batch size does not change Nl
        @test calcNl(64, growNr(64, 21)) == calcNl(64, 21)
        @test estimate_memory(100, 50, 40) ==
              4 * (100*50*4 + 100*90*3 + 100*40)
        @test estimate_memory(100, 50, 40, 2, 3) ==
              4 * (100*50*5 + 100*90*3 + 100*40*3)
        @test estimate_memory(8, 8, 8; factors=(2,)) ==
              4 * (8*8*4 + 8*16*3 + 8*8)
        zdtws = ZDTWorkspace(d2, -1:0.5:1)
        @test driftrates(zdtws) == range(-1.0f0, step=0.5f0, length=5)
        @test collect(driftrates(zdtws)) ≈ collect(-1:0.5:1)
        @test collect(driftrates(zdtws, 2.5)) ≈ collect(2.5:0.5:4.5)
    end

    if isdefined(Main, :CUDA)
        if CUDA.functional()
            @testset "zdtfdr [CUDA]" begin
                g = CuArray(d)
                gdtws = ZDTWorkspace(g, -1:1)
                @test Array(zdtfdr(gdtws)) ≈ fdr_expected
            end

            @testset "taylorfdr [CUDA]" begin
                g = CuArray(d2)
                @test Array(taylorfdr(g, 0)) == taylorfdr(d2, 0)
                gout = CuArray{Float32}(undef, Nf, Nt)
                taylorfdr!(gout, TaylorWorkspace(g), g, 0)
                @test Array(gout) == taylorfdr(d2, 0)
                @test Array(taylorfdr(g, -1:1)) == taylorfdr(d2, -1:1)
                # Non-power-of-2 time samples (the GPU materializes the
                # zero padding and the CPU virtualizes it; results match)
                gs = CuArray(d3[:, 1:5])
                @test Array(taylorfdr(gs, 0)) == taylorfdr(d3[:, 1:5], 0)
                # Larger random cases, multiple drift blocks, pow2 and not
                spec = randn(Float32, 128, 64)
                gc = CuArray(spec)
                @test Array(taylorfdr(gc, -1:1)) == taylorfdr(spec, -1:1)
                spec = randn(Float32, 256, 100)
                gc = CuArray(spec)
                @test Array(taylorfdr(gc, -2:2)) == taylorfdr(spec, -2:2)
                # taylortree! returns one of the two workspace buffers
                gws = TaylorWorkspace(gc)
                result = taylortree!(gws.buffer1, gws.buffer2, gc, 0)
                @test result === gws.buffer1 || result === gws.buffer2
                @test Array(result) == taylorfdr(spec, 0)
            end

            @testset "fdrstats [CUDA]" begin
                gz = CuArray(zeros(Float32, 4, 4))
                @test fdrstats(gz) == (0.0f0, Inf)
                m, s = fdrstats(CuArray(d2))
                @test 0 < s < Inf
            end
        else
            @info "Skipping CUDA tests: no functional GPU available"
        end
    end

end;

# Run the heavy tests if their dataset is already available locally (no
# download needed), or if FDR_HEAVY_TESTS=1 is set (downloads on first use).
# Keep the filename below in sync with test/heavytests.jl.
voyager_datadir = DataDeps.try_determine_load_path("voyager-2020-single-coarse-channel", @__DIR__)
voyager_ready = voyager_datadir !== nothing && isfile(joinpath(
    voyager_datadir, "single_coarse_guppi_59046_80036_DIAG_VOYAGER-1_0011.rawspec.0000.h5"))
if voyager_ready || get(ENV, "FDR_HEAVY_TESTS", "0") == "1"
    @testset "heavy tests" begin
        include("heavytests.jl")
    end
else
    @info "Skipping heavy tests (dataset not downloaded; set FDR_HEAVY_TESTS=1 to download and run)"
end

# end of runtests.jl