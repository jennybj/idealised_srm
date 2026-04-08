import Pkg

Pkg.add([
    # geoengi_v1.jl
    "SlurmClusterManager",
    "Optim",
    "Roots",
    "Interpolations",
    "Plots",
    "CSV",
    "DataFrames",
    "Formatting",
    "BenchmarkTools",
    "FastGaussQuadrature",
    "ProgressMeter",
    # plot_helper.jl / make_figures.jl
    "GMT",
    "StatsBase",
])
