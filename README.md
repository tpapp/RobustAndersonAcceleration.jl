# RobustAndersonAcceleration.jl

![lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)
[![build](https://github.com/tpapp/RobustAndersonAcceleration.jl/workflows/CI/badge.svg)](https://github.com/tpapp/RobustAndersonAcceleration.jl/actions?query=workflow%3ACI)
<!-- Documentation -- uncomment or delete as needed -->
<!--
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://tpapp.github.io/RobustAndersonAcceleration.jl/stable)
[![Documentation](https://img.shields.io/badge/docs-master-blue.svg)](https://tpapp.github.io/RobustAndersonAcceleration.jl/dev)
-->[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)

A robust Julia implementation of [Anderson acceleration](https://en.wikipedia.org/wiki/Anderson_acceleration).

At this point this is an experimental package that I am using in my own research, to solve economic models.

It is *robust* in two ways:

1. The subproblem algorithm uses truncated [SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition) for numerical stability. This is important for ill-conditioned problems.

2. The implementation is type stable and is verified with the amazing [JET.jl](https://github.com/aviatesk/JET.jl).

It has the following *drawbacks*, which may or may not be relevant for you:

1. It is written with the assumption that the most expensive part of your problem is the actual fixed point calculation. In this context the SVD is worth it.

2. At the moment the package does not go out of its way to minimize allocations, which are, in comparison, a trivial cost for real-world problems but obfuscate the code. This may change.

3. It is not as heavily tested as more mature alternatives, which include [SpeedMapping.jl](https://github.com/NicolasL-S/SpeedMapping.jl) and [FixedPointAcceleration.jl](https://github.com/s-baumann/FixedPointAcceleration.jl).

See the package docstring for documentation.
