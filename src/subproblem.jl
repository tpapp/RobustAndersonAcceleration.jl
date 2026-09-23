norm2(x) = norm(x, 2)

"""
$(SIGNATURES)

Solve ``\\| R α + f \\|`` using SVD. Return `(; α, c_svd, r)` where `r` is the revealed rank
and `c_R` is the condition number (after truncation, see below).

`κ` is a relative truncation factor.
"""
function svd_least_squares(R::AbstractMatrix{T}, f; κ = 8.0) where T
    @argcheck κ > 0
    (; U, S, Vt) = svd(R)        # note specify algorithm
    τ = max(size(R)...) * eps(eltype(R)) * κ * S[1] # cutoff
    r = searchsortedlast(S, τ; rev = true)          # rank
    if isempty(S) || r == 0 || S[1] == 0
        # zero matrix, zero rank, infinite condition number: pick zero as the minimizer
        return (; α = zeros(T, size(R, 2)), r = 0, c_svd = oftype(S[1], Inf))
    end
    c_svd = S[1] / S[r]                 # condition number
    α = Vt[1:r, :]' * ((U[:, 1:r]' * -f) ./ S[1:r])
    (; α, r, c_svd)
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

struct SVDSolver end

function optimal_coefficients(::SVDSolver, F::AbstractMatrix, i::Int)
    nrow, ncol = size(F)
    (; D, c_diff) = subtract_reference!(F, i)
    (; α, r, c_svd) = svd_least_squares(D, @view(F[:, i]))
    insert!(α, i, 1 - sum(α))
end
