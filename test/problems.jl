#####
##### test suite
#####

"""
Fixed-point test problems for Anderson acceleration.

Every problem has the form

    g(x::Vector) -> Vector

and is represented by

    (
        name = "...",
        g = ...,
        x0 = ...,
        xstar = ...,
    )

where xstar satisfies g(xstar) ≈ xstar.

NOTE: The test suite was compiled with with ChatGPT.
"""
const FP_PROBLEMS = [

    # ------------------------------------------------------------------
    # 1. 1D vector version of x = cos(x)
    #
    # Classic fixed-point problem.
    # ------------------------------------------------------------------
    (
        name = "cosine_1d",
        g = x -> [cos(x[1])],
        x0 = [1.0],
        xstar = [0.7390851332151607],
    ),

    # ------------------------------------------------------------------
    # 2. 1D vector version of x = exp(-x)
    # ------------------------------------------------------------------
    (
        name = "exponential_1d",
        g = x -> [exp(-x[1])],
        x0 = [1.0],
        xstar = [0.5671432904097838],
    ),

    # ------------------------------------------------------------------
    # 3. Deliberately slow linear problem:
    #
    #     x = 0.99*x + 1
    #
    # Fixed point = 100.
    # ------------------------------------------------------------------
    (
        name = "slow_linear_1d",
        g = x -> [0.99*x[1] + 1.0],
        x0 = [0.0],
        xstar = [100.0],
    ),

    # ------------------------------------------------------------------
    # 4. Linear problem with a negative eigenvalue:
    #
    #     x = -0.9*x + 1
    #
    # Picard iteration oscillates strongly.
    # ------------------------------------------------------------------
    (
        name = "oscillatory_linear_1d",
        g = x -> [-0.9*x[1] + 1.0],
        x0 = [0.0],
        xstar = [1.0 / 1.9],
    ),

    # ------------------------------------------------------------------
    # 5. Nonlinear scalar-equivalent problem:
    #
    #     x = x - 0.1*(x^2 - 2)
    #
    # Fixed point = sqrt(2).
    # ------------------------------------------------------------------
    (
        name = "sqrt2_1d",
        g = x -> [x[1] - 0.1*(x[1]^2 - 2.0)],
        x0 = [1.0],
        xstar = [sqrt(2.0)],
    ),

    # ------------------------------------------------------------------
    # 6. 2D linear problem.
    #
    #     x = A*x + b
    #
    # A = [0.7 0.2
    #      0.1 0.7]
    #
    # The fixed point is [50/27, -50/27].
    # ------------------------------------------------------------------
    (
        name = "linear_2d",
        g = x -> [
            0.7*x[1] + 0.2*x[2] + 1.0,
            0.1*x[1] + 0.7*x[2] - 1.0,
        ],
        x0 = zeros(2),
        xstar = [
            10.0 / 7.0,
            -20.0 / 7.0,
        ],
    ),

    # ------------------------------------------------------------------
    # 7. Diagonal linear problem with widely separated convergence
    # rates.
    #
    # Eigenvalues = 0.99, 0.5, 0.1.
    # ------------------------------------------------------------------
    (
        name = "linear_spectral_spread",
        g = x -> [
            0.99*x[1] + 1.0,
            0.50*x[2] + 1.0,
            0.10*x[3] + 1.0,
        ],
        x0 = zeros(3),
        xstar = [
            100.0,
            2.0,
            1.0 / 0.9,
        ],
    ),

    # ------------------------------------------------------------------
    # 8. Dense 5D linear problem.
    #
    # Useful for testing vector handling and the least-squares problem.
    # ------------------------------------------------------------------
    (
        name = "linear_5d",
        g = let
            A = [
                0.60  0.10  0.00  0.00  0.05
                0.05  0.55  0.10  0.00  0.00
                0.00  0.05  0.60  0.10  0.00
                0.00  0.00  0.05  0.55  0.10
                0.05  0.00  0.00  0.05  0.60
            ]
            b = [1.0, -1.0, 0.5, 2.0, -0.5]

            x -> A*x + b
        end,

        x0 = zeros(5),

        xstar = let
            A = [
                0.60  0.10  0.00  0.00  0.05
                0.05  0.55  0.10  0.00  0.00
                0.00  0.05  0.60  0.10  0.00
                0.00  0.00  0.05  0.55  0.10
                0.05  0.00  0.00  0.05  0.60
            ]
            b = [1.0, -1.0, 0.5, 2.0, -0.5]

            (I - A) \ b
        end,
    ),

    # ------------------------------------------------------------------
    # 9. Nonlinear 2D problem.
    #
    # The reference solution is obtained independently by many Picard
    # iterations.
    # ------------------------------------------------------------------
    (
        name = "nonlinear_2d",
        g = x -> [
            0.5*x[1] + 0.2*sin(x[2]) + 0.5,
            0.3*x[2] + 0.2*cos(x[1]) + 0.7,
        ],
        x0 = [0.0, 0.0],

        xstar = let
            x = [1.0, 1.0]

            for _ in 1:1000
                xnew = [
                    0.5*x[1] + 0.2*sin(x[2]) + 0.5,
                    0.3*x[2] + 0.2*cos(x[1]) + 0.7,
                ]

                if norm(xnew - x) < 1e-15
                    x = xnew
                    break
                end

                x = xnew
            end

            x
        end,
    ),

    # ------------------------------------------------------------------
    # 10. Nonlinear coupled 2D problem.
    #
    #     g(x) = [
    #         0.7*x1 + 0.1*x2^2 + 0.2,
    #         0.6*x2 + 0.1*x1^2 + 0.3
    #     ]
    #
    # (1, 1) is an exact fixed point.
    # ------------------------------------------------------------------
    (
        name = "nonlinear_quadratic_2d",
        g = x -> [
            0.7*x[1] + 0.1*x[2]^2 + 0.2,
            0.6*x[2] + 0.1*x[1]^2 + 0.3,
        ],
        x0 = [0.5, 0.5],
        xstar = [1.0, 1.0],
    ),

    # ------------------------------------------------------------------
    # 11. Very slowly converging linear 2D problem.
    #
    # Eigenvalues = 0.995 and 0.990.
    #
    # This is particularly useful for testing whether Anderson actually
    # accelerates a problem for which Picard iteration is very slow.
    # ------------------------------------------------------------------
    (
        name = "very_slow_linear_2d",
        g = x -> [
            0.995*x[1] + 0.5,
            0.990*x[2] - 1.0,
        ],
        x0 = zeros(2),
        xstar = [100.0, -100.0],
    ),

    # ------------------------------------------------------------------
    # 12. Mixed positive/negative spectrum.
    #
    # Eigenvalues = 0.9, -0.8, 0.4.
    # ------------------------------------------------------------------
    (
        name = "mixed_spectrum",
        g = x -> [
            0.90*x[1] + 0.20,
            -0.80*x[2] + 0.30,
            0.40*x[3] - 0.50,
        ],
        x0 = zeros(3),
        xstar = [
            2.0,
            0.30 / 1.80,
            -0.50 / 0.60,
        ],
    ),
]

@testset "problems" begin
    for (; name, g, x0, xstar) in FP_PROBLEMS
        # (; name, g, x0, xstar) = FP_PROBLEMS[6]
        @info "solving" name
        sol = RAA.fixed_point(g, x0)
        @test sol.converged
        @info sol
        @test sol.x ≈ xstar atol = 1e-4
    end
end
