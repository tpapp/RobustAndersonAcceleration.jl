"""
A robust Julia implementation of Anderson acceleration.

The subproblem is solved using SVD, which makes sense for numerically challenging
problems where the function evaluation itself is the expensive part.

## Public API


## Public API

No symbols are exported. It is recommended that the user `import`s the package with an
alias, eg

```julia
import RobustAndersonAcceleration as RAA
```

See [`fixed_point`](@ref), [`SVDSolver`](@ref), [`CheckTermination`](@ref), and
[`FixedPointResult`](@ref).
"""
module RobustAndersonAcceleration

using ArgCheck: @argcheck
using DocStringExtensions: SIGNATURES, FIELDS
using LinearAlgebra: norm, svd
using Printf: @sprintf

public SVDSolver, RelAbsDiff, fixed_point, FixedPointResult

####
#### utilities
####

"""
Maintain the invariant that the sum of accumulated values is `scale^2 * scaled_sum`.
`scale` may be negative.
"""
struct AccNorm{T}
    scale::T
    scaled_sum::T
end

__init_norm(::Type{T}) where T = AccNorm(one(T), zero(T))

function __acc_norm(acc_norm, x)
    @argcheck isfinite(x)
    (; scale, scaled_sum) = acc_norm
    scaled_sum += abs2(x / scale)
    if !isfinite(scaled_sum)
        scale *= x
        scaled_sum = acc_norm.scaled_sum / x / x + one(scaled_sum)
    end
    AccNorm(scale, scaled_sum)
end

__done_norm(acc_norm) = sqrt(acc_norm.scaled_sum) * acc_norm.scale

"""
$(SIGNATURES)

Shorthand for the Euclidean norm.
"""
function norm2(x::AbstractVector{T}) where T<:AbstractFloat
    acc = __init_norm(T)
    for x in x
        acc = __acc_norm(acc, x)
    end
    __done_norm(acc)
end

"""
$(SIGNATURES)

Euclidean norm of the difference of two vectors.
"""
function norm2diff(x::AbstractVector{T}, y::AbstractVector{T}) where {T<:AbstractFloat}
    # NOTE failure on mismatching types is deliberate
    acc = __init_norm(T)
    for (x, y) in zip(x, y)
        acc = __acc_norm(acc, x - y)
    end
    __done_norm(acc)
end

####
#### circular buffer
####

"""
Implementation of a circular buffer using matrixes. Internal, not part of the API.
"""
mutable struct CircularBuffer{M<:AbstractMatrix}
    "Inputs"
    const x::M
    "Outputs"
    const fx::M
    "Index of last value in inputs"
    i::Int
    """
    `true` iff `i` has been reset to `1` at least once. When `false`, only columns
    `1:i` in `x` and `fx` are valid.
    """
    rollover::Bool
end

"""
$(SIGNATURES)

Construct a circular buffer of the given type `T`, for vectors of length `len`, with
space for `depth` vectors.
"""
function make_circular_buffer(::Type{T}, len::Int, depth::Int) where T
    CircularBuffer(zeros(T, len, depth), zeros(T, len, depth), 0, false)
end

"""
$(SIGNATURES) → count

Number of vectors in the buffer.
"""
function get_count(buffer::CircularBuffer)
    (; x, i, rollover) = buffer
    rollover ? size(x, 2) : i
end

"""
$(SIGNATURES) → x, fx, i

Return the input matrix, the output matrix, and the current index of the buffer. When
the buffer has not rolled over, matrices are resized as needed.
"""
function get_inputs_outputs_index(buffer::CircularBuffer)
    (; x, fx, i, rollover) = buffer
    k = rollover ? lastindex(x, 2) : i
    @view(x[:, 1:k]), @view(fx[:, 1:k]), i
end

"""
$(SIGNATURES) → fx

Get the output matrix.
"""
function get_outputs(buffer::CircularBuffer)
    (; x, fx, i, rollover) = buffer
    k = rollover ? lastindex(x, 2) : i
    @view(fx[:, 1:k])
end

"""
$(SIGNATURES) → (; D, r, residuals_diagnostics)

Let `R = outputs .- input` (the residuals), then

- `r` is the reference column of `R` (the last evaluation),
- `D` is `R` with `r` subtracted, omitting the reference column,
- `residuals_diagnostics = (; c_diff)` is the largest norm difference of the columns of
  `R` vs the reference column `r`; small numbers are a cause for concern.
"""
function get_differential_residuals(buffer::CircularBuffer)
    _x, _fx, i = get_inputs_outputs_index(buffer)
    nrow, ncol = size(_fx)
    T = eltype(_fx)
    @argcheck ncol ≥ 2 "Not enough vectors in circular buffer."
    D = similar(_fx, nrow, ncol - 1)
    c = zero(T)
    r_i = @view(_fx[:, i]) .- @view(_x[:, i])
    norm_i = norm2(r_i)
    for j in 1:(ncol-1)
        k = j ≥ i ? j + 1 : j
        acc_d = __init_norm(T)
        acc_r = __init_norm(T)
        for l in 1:nrow
            r = _fx[l, k] - _x[l, k] # residuals
            acc_r = __acc_norm(acc_r, r)
            d = r - r_i[l]      # residuals minus reference
            acc_d = __acc_norm(acc_d, d)
            D[l, j] = d
        end
        c = max(c, __done_norm(acc_d) / (norm_i + __done_norm(acc_r)))
    end
    (; D, r = r_i, residuals_diagnostics = (; c_diff = c), i)
end

"""
$(SIGNATURES)

Add an input-output pair to the circular buffer.
"""
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
#### subproblem solver
####

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

"""
`SVDSolver(; kwargs...)`

Solve the subproblem using singular value decomposition.

Works well with ill-conditioned problems.

# Fields (also keyword arguments to the constructor)

$(FIELDS)
"""
Base.@kwdef struct SVDSolver{K}
    "Determines the relative cutoff for singular values."
    κ::K = DEFAULT_κ
end

"""
$(SIGNATURES) → (; α, diagnostics)

Calculate the optimal coefficients `α` for the subproblem, ie minimizing the Euclidean
norm of ``residuals ⋅ α``.

Also return the `diagnostics` for the `solver`.
"""
function optimal_coefficients(solver::SVDSolver, buffer)
    (; D, r, i, residuals_diagnostics) = get_differential_residuals(buffer)
    @argcheck !isempty(D)
    (; α, revealed_rank, c_svd) = svd_least_squares(D, r)
    (α = insert!(α, i, 1 - sum(α)),
     diagnostics = (; revealed_rank, c_svd, residuals_diagnostics...))
end

####
#### termination
####

struct RelAbsDiff{T}
    atol::T
    rtol::T
    """
    $(SIGNATURES)

    A callable with vector arguments `(a, b)` that evaluates to a boolean, true iff
    ``\\| a - b \\|_2 ≤ \\max(1, \\|a\\|_2, \\|b\\|_2)
    """
    function RelAbsDiff(atol::T = 1e-8, rtol::T = atol) where T
        @argcheck atol ≥ 0
        @argcheck rtol ≥ 0
        new{T}(atol, rtol)
    end
end

function (rac::RelAbsDiff{T})(a::AbstractVector, b::AbstractVector) where T
    (; atol, rtol) = rac
    norm2diff(a, b) ≤ atol + rtol * max(one(T), norm2(a), norm2(b))
end

Base.@kwdef struct FixedPointResult{V,D}
    iterations::Int
    converged::Bool
    termination::Symbol
    x::V
    residual::V
    diagnostics::D
end

function Base.show(io::IO, fp::FixedPointResult)
    (; iterations, converged, termination, x, residual, diagnostics) = fp
    r_norm = @sprintf("%.2e", norm2(residual))
    if converged
        printstyled(io,
                    "converged after $(iterations) iterations with residual norm $(r_norm)";
                    color = :green)
    else
        printstyled(io, "did not converge after $(iterations) iterations\n",
                    "terminated “:$(termination)”, residual norm $(r_norm), diagnostics:\n",
                    diagnostics; color = :red)
    end
end

"""
$(SIGNATURES)

FIXME document
"""
function fixed_point(f, x0::AbstractVector;
                     solver = SVDSolver(),
                     check_solution = RelAbsDiff(),
                     check_stagnation = RelAbsDiff(),
                     stagnation_threshold::Int = 8,
                     maximum_iterations::Int = 100,
                     depth::Int = 5)
    x = x0
    @argcheck all(isfinite, x)
    buffer = make_circular_buffer(eltype(x0), length(x), depth)
    j = 1
    stagnation_counter = 0
    while true
        if get_count(buffer) ≤ 1
            x′ = f(x)
            add_x_fx(buffer, x, x′)
            x = x′
        else
            (; α, diagnostics) = optimal_coefficients(solver, buffer)
            x′ = get_outputs(buffer) * α
            fx′ = f(x′)
            add_x_fx(buffer, x′, fx′)
            converged = check_solution(x′, fx′)
            stagnation = check_stagnation(x, x′)
            if stagnation
                stagnation_counter += 1
            else
                stagnation_counter = 0
            end
            stagnating = stagnation_counter ≥ stagnation_threshold
            maxiter = j == maximum_iterations
            if converged || stagnating || maxiter
                termination = converged ? :convergence :
                    (stagnating ? :stagnation : :maximum_iterations)
                return FixedPointResult(; iterations = j, converged,
                                        termination, x = x′, residual = fx′ .- x′,
                                        diagnostics)
            end
            x = x′
        end
        j += 1
    end
end

end # module
