import RobustAndersonAcceleration as RAA
using Test, JET, LinearAlgebra

@testset "static analysis with JET.jl" begin
    JET.test_package(RAA, target_modules=(RAA,))
end

####
#### subproblem
####

@testset "least squares SVD" begin
    @testset "full rank" begin
        for _ in 1:10000
            m = rand(2:5)
            n = rand(2:5)
            R = randn(m, n)     # rank deficiency in random matrixes should be rare
            f = randn(m)
            (; α, revealed_rank) = RAA.svd_least_squares(R, f)
            @test revealed_rank == min(m, n)
            @test α ≈ R \ (-f)
        end
    end
    @testset "rank deficient" begin
        R = [1 + 4*eps() 1;
             2 + 4*eps() 2]
        f = ones(2)
        (; α, revealed_rank, c_svd) = RAA.svd_least_squares(R, f)
        @test revealed_rank == 1
        @test c_svd == 1
    end
    @testset "inference" begin
        R = randn(3, 3)
        f = randn(3)
        @inferred RAA.svd_least_squares(R, f)
    end
end

"Perturb the vector `a` keeping the its sum invariant."
function perturb(a::AbstractVector{T}, scale = 0.001, ρ = 0.8) where T
    b = copy(a)
    s = zero(T)
    n = length(b)
    for i in firstindex(b):(lastindex(b)-1)
        e = randn() * scale - s * ρ
        b[i] += e
        s += e
    end
    b[end] -= s
    b
end

@testset "optimal coefficients perturbation test" begin
    for _ in 1:100
        F = randn(3, 3)
        γ = RAA.optimal_coefficients(RAA.SVDSolver(), F, 3).α
        @test sum(γ) ≈ 1
        @test norm(F * γ, 2) ≤ norm(F * perturb(γ), 2)
    end
end

@testset "fixed point type stability and sanity checks" begin
    # NOTE: this is not a challenging problem, we just check type stability and printing
    fp = @inferred RAA.fixed_point(x -> 0.3 .* x, ones(3))
    @test repr(fp) isa AbstractString # sanity check printing
    @test fp.x ≈ zeros(3) atol = 1e-8
    @test fp.residual ≈ zeros(3) atol = 1e-8
    @test fp.iterations ≤ 5
end

@testset "QA with Aqua" begin
     import Aqua
     Aqua.test_all(RAA)
end
