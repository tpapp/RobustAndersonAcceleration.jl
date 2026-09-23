"""
Placeholder for a short summary about RobustAndersonAcceleration.
"""
module RobustAndersonAcceleration

using ArgCheck: @argcheck
using DocStringExtensions: SIGNATURES
using LinearAlgebra: norm, svd
using Printf: @printf

public SVDSolver, CheckTermination, FixedPoint, fixed_point

####
#### subproblem solver
####

"""
$(SIGNATURES)

Shorthand for the Euclidean norm.
"""
norm2(x) = norm(x, 2)

const DEFAULT_κ = 8.0

"""
$(SIGNATURES)

Solve ``\\| R α + f \\|`` using SVD. Return `(; α, c_svd, revealed_rank)` where `c_svd`
is the condition number (after truncation).

`κ` is a relative truncation factor.
"""
function svd_least_squares(R::AbstractMatrix{T}, f; κ = DEFAULT_κ) where T
    @argcheck κ > 0
    (; U, S, Vt) = svd(R)        # note specify algorithm
    τ = max(size(R)...) * eps(eltype(R)) * κ * S[1] # cutoff
    r = searchsortedlast(S, τ; rev = true)          # rank
    if isempty(S) || r == 0 || S[1] == 0
        # zero matrix, zero rank, infinite condition number: pick zero as the minimizer
        return (; α = zeros(T, size(R, 2)), revealed_rank = 0, c_svd = oftype(S[1], Inf))
    end
    c_svd = S[1] / S[r]                 # condition number
    α = Vt[1:r, :]' * ((U[:, 1:r]' * -f) ./ S[1:r])
    (; α, revealed_rank = r, c_svd)
end

function subtract_reference!(F::AbstractMatrix{T}, i::Int) where T
    nrow, ncol = size(F)
    @argcheck ncol ≥ 2
    D = similar(F, nrow, ncol - 1)
    c = zero(T)
    f_i = @view F[:, i]
    norm_i = norm2(f_i)
    for j in 1:(ncol-1)
        k = j ≥ i ? j + 1 : j
        D[:, j] .= F[:, k] .- f_i
        c = max(c, norm2(@view D[:, j]) / (norm_i + norm2(@view F[:, k])))
    end
    (; D, c_diff = c)
end

Base.@kwdef struct SVDSolver{K}
    κ::K = DEFAULT_κ
end

"""
$(SIGNATURES)
"""
function optimal_coefficients(::SVDSolver, F::AbstractMatrix, i::Int)
    nrow, ncol = size(F)
    @argcheck ncol ≥ 2
    (; D, c_diff) = subtract_reference!(F, i)
    (; α, revealed_rank, c_svd) = svd_least_squares(D, @view(F[:, i]))
    (α = insert!(α, i, 1 - sum(α)),
     diagnostics = (; revealed_rank, c_diff, c_svd))
end

####
#### circular buffer
####

mutable struct CircularBuffer{M<:AbstractMatrix}
    const x::M
    const fx::M
    i::Int
    rollover::Bool
end

function make_circular_buffer(::Type{T}, len::Int, depth::Int) where T
    CircularBuffer(zeros(T, len, depth), zeros(T, len, depth), 0, false)
end

function get_residuals_values_reference(buffer::CircularBuffer)
    (; x, fx, i, rollover) = buffer
    if rollover
        fx .- x, fx, i
    else
        _fx = @view(fx[:, 1:i])
        _fx .- @view(x[:, 1:i]), _fx, i
    end
end

function add_x_fx(buffer::CircularBuffer, new_x, new_fx)
    (; x, fx, i, rollover) = buffer
    i += 1
    if i > size(x, 2)
        i = 1
        if !rollover
            buffer.rollover = true
        end
    end
    buffer.i = i
    x[:, i] .= new_x
    fx[:, i] .= new_fx
    nothing
end

####
####
####

Base.@kwdef struct CheckTermination
    x_atol = 1e-8
    x_rtol = 1e-8
    residual_atol = 1e-8
    residual_rtol = 1e-8
end

function (ct::CheckTermination)(; previous_x, x, residual)
    x_norm = norm2(x)
    residual_norm = norm2(residual)
    d_norm = previous_x ≡ nothing ? oftype(x_norm, Inf) : norm2(x .- previous_x)
    if residual_norm ≤ ct.residual_atol
        true, :converged_absolute, residual_norm
    else
        relative_residual_norm = residual_norm / x_norm
        if relative_residual_norm ≤ ct.residual_rtol
            true, :converged_relative, relative_residual_norm
        elseif d_norm ≤ ct.x_atol
            true, :notmoving_absolute, d_norm
        else
            relative_d_norm = d_norm / x_norm
            if relative_d_norm ≤ ct.x_rtol
                true, :notmoving_relative, relative_d_norm
            else
                false, :not_converging, oftype(relative_residual_norm, NaN)
            end
        end
    end
end

Base.@kwdef struct FixedPoint{T,V,D}
    iterations::Int
    converged::Bool
    termination::Symbol
    convergence_metric::T
    x::V
    residual::V
    diagnostics::D
end

function Base.show(io::IO, fp::FixedPoint)
    (; iterations, converged, termination, convergence_metric, x, residual, diagnostics) = fp
    # FIXME make this color
    if converged
        print(io, "converged after $(iterations) iterations\n",
              "terminated “:$(termination)” with metric ")
        @printf(io, "%.2e", convergence_metric)
    else
        print(io, "did not converge after $(iterations) iterations\n",
              "terminated “:$(termination)”, diagnostics:\n",
              diagnostics)
    end
end

function fixed_point(f, x0::AbstractVector;
                     solver = SVDSolver(), termination = CheckTermination(),
                     maximum_iterations = 100,
                     depth = 5)
    x = x0
    buffer = make_circular_buffer(eltype(x0), length(x), depth)
    j = 1
    while true
        R, G, i = get_residuals_values_reference(buffer)
        if size(R, 2) ≤ 1
            x′ = f(x)
            add_x_fx(buffer, x, x′)
            x = x′
        else
            (; α, diagnostics) = optimal_coefficients(solver, R, i)
            x′ = G * α
            fx′ = f(x′)
            add_x_fx(buffer, x′, fx′)
            residual = fx′ .- x′
            (converged, termination,
             convergence_metric) = termination(; previous_x = x, x = x′, residual)
            if converged
                return FixedPoint(; iterations = j, converged,
                                  termination, convergence_metric,
                                  x = x′, residual, diagnostics)
            elseif j == maximum_iterations
                return FixedPoint(; iterations = j, converged = false,
                                  termination = :maximum_iterations, convergence_metric,
                                  x = x′, residual, diagnostics)
            end
            x = x′
        end
        j += 1
    end
end

end # module
