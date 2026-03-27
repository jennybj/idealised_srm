include("io3.jl")
include("creategrid1.jl")

using DelimitedFiles
using GMT
using StatsBase, Plots, Statistics

ENV["PATH"] *= ":/opt/homebrew/bin"


# Set input path where lat-lon files are stored
lat_lon_path = ""  

dm = readdlm(lat_lon_path, skipstart = 0)
cols = [2,3]
dm1 = dm[:,cols ]
latitude = dm1[:,1] .+ 0.5
longitude = dm1[:,2] .+ 0.5

function make_grid(data)
    df = [longitude latitude data]
    grd = xyz2grd(df, region=(-180, 180, -60, 75), registration=:p, spacing=(1.0, 1.0))
    return grd
end

# make_diverging_linear_map_symmetric creates a diverging map centered at 0 with a symmetric linear scale. The range extends from -range_max to +range_max (default -5 to 5).


function make_diverging_linear_map_symmetric(data, heading::String, filename; n_bins=6, range_max=5)
    grid = make_grid(data)

    # Create colormap with fixed symmetric range centered at 0
    # bg/fg set colors for values below/above range to match endpoints (not black/white)
    C1 = makecpt(cmap=:vik, range=(-range_max, range_max), continuous=true,
                 bg=:match, fg=:match)

    # Calculate tick interval (aim for ~10 ticks)
    tick_interval = 1

    grdimage(grid,
             proj=:Miller,
             cmap=C1,
             xaxis=(annot=60, ticks=60),
             yaxis=(annot=20, ticks=20),
             title=heading,
             region=(-180, 180, -60, 75))

    coast!(water="lightblue",
           region=(-180, 180, -60, 75),
           proj=:Miller)

    colorbar!(
        par=(FORMAT_FLOAT_MAP="%.1f",),
        pos=(anchor=:paper, justify=:CT, size=(7,0.2), offset=(0.5,0)),
        nolines=true,
        cmap=C1,
        frame=(annot=1,),
        savefig=output_path * filename
    )
end


# make_symmetric_linear_map creates a symmetric linear map with range from -range_max to +range_max.

function make_symmetric_linear_map(data, range_max::Real, filename; heading="")
    grid = make_grid(data)

    # Create colormap with fixed symmetric range centered at 0 (same as tempdiff_2100_linear)
    C1 = makecpt(cmap=:vik, range=(-range_max, range_max), continuous=true,
                 bg=:match, fg=:match)

    # Calculate tick interval for colorbar (aim for ~10-15 ticks)
    tick_interval = max(1, round(Int, 2 * range_max / 14))

    grdimage(grid,
             proj=:Miller,
             cmap=C1,
             xaxis=(annot=60, ticks=60),
             yaxis=(annot=20, ticks=20),
             title=heading,
             region=(-180, 180, -60, 75))

    coast!(water="lightblue",
           region=(-180, 180, -60, 75),
           proj=:Miller)

    colorbar!(
        par=(FORMAT_FLOAT_MAP="%.1f",),
        pos=(anchor=:paper, justify=:CT, size=(7,0.2), offset=(0.5,0)),
        nolines=true,
        cmap=C1,
        frame=(annot=tick_interval,),
        savefig=output_path * filename
    )
end
