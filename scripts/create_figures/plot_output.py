# --------------------------------------------------------------------------------------

# import sys as sys
import glob as glob
import os as os
import sys as sys
from collections import defaultdict

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn.apionly as sns
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import (
    BoundaryNorm,
    LinearSegmentedColormap,
    ListedColormap,
    TwoSlopeNorm,
)

sys.path.insert(
    0, "/home/jennybj/Documents/idealised_srm/scripts/"
)  # CHANGE path to location on module
from module_coupling import *

# --------------------------------------------------------------------------------------

alpha = 0.36  # capital’s share of income (capital share + labor share = 1)
delta = 0.06  # The (annual) rate of depreciation of the capital stock
price = get_price()
ga, beta, delta, alpha, energyshare, rss, theta, b = get_constants()

srm_start_year = 2030

pi_temperature = get_pi_temperature()
gamma1, gamma2, rhos = get_coefficients()
a_coeff, b_coeff, const = get_srm_coefficients()

file_path = "/home/jennybj/uio/home/coupling/"  # CHANGE

# --------------------------------------------------------------------------------------

global_pi_temperature = 14.460473280816053

pi_temperatures = get_pi_temperature()
population = get_population()
chi = get_chit()

diam_latitudes, diam_longitudes = get_coordinate_data()

ncells = diam_latitudes.shape[0]

global_population = np.sum(population, axis=1)

# --------------------------------------------------------------------------------------

# READ IN DATA

years, srm_fp_sum_actual_emissions = np.loadtxt(
    file_path + "emissions.txt", unpack=True
)
base_fp_sum_actual_emissions = np.loadtxt(
    file_path + "full_couple_population_e2/emissions.txt", usecols=1
)

simulations = [
    "srm_fp",
    "base_fp",
    "srm_e1",
    "srm_e2",
    "srm_e3",
    "base_e1",
    "base_e2",
    "base_e3",
]
names = [
    "full_couple_population/fixed_point",
    "full_couple_SRM_ens/full_couple_SRM_2028",
    "full_couple_SRM_ens/full_couple_SRM_2029",
    "full_couple_SRM_ens/full_couple_SRM_2030",
    "full_couple_population/full_couple_population",
    "full_couple_population_cont/full_couple_population_cont",
    "full_couple_population_e3/full_couple_population_e3",
]
variables = [
    "wealth",
    "regtemp",
    "capital",
    "actual_emissions",
]

wealth = np.loadtxt(
    file_path + "full_couple_SRM_ens/full_couple_SRM_2028_wealth.txt", skiprows=2
)
nyears = wealth.shape[0]

for isim in range(len(names)):
    for ivar in variables:
        var = np.loadtxt(file_path + names[isim] + "_" + ivar + ".txt", skiprows=2)[
            :nyears, :
        ]
        print(simulations[isim + 1] + "_" + ivar + "= var")
        exec(simulations[isim + 1] + "_" + ivar + "= var")

    if isim != 0:
        var = np.loadtxt(file_path + names[isim] + "_global_temp.txt", usecols=1)[
            :nyears
        ]
        print(simulations[isim + 1] + "_global_temp= var")
        exec(simulations[isim + 1] + "_global_temp= var")


# Read in Henri's data:
srm_ai = np.loadtxt("../../standalone_output/ai.txt")[:, 1 : nyears + 1].T
srm_fp_gdp = np.loadtxt("../../standalone_output/reg_gdp.txt")[:, 1 : nyears + 1].T
srm_fp_capital = np.loadtxt("../../standalone_output/capital.txt")[:, 1 : nyears + 1].T
# srm_fp_emissions = np.loadtxt("standalone_output/regional_emissions.txt")[:, 1 : nyears + 1].T
srm_fp_regtemp = np.loadtxt("../../standalone_output/regional_temperature.txt")[
    :, 1 : nyears + 1
].T


# --------------------------------------------------------------------------------------

# FUNCTIONS


def descale(in_variable, in_ai):
    nyears = in_variable.shape[0]
    out_variable = np.zeros((nyears, ncells))

    for iyear in range(nyears):
        out_variable[iyear, :] = in_variable[iyear, :] * (
            population[iyear, :] * in_ai[iyear, :]
        )

    return out_variable


def damages(regtemp, tstar=12.609, scale1=0.00327721, scale2=0.00362887):
    """The regional damage function. Already raised to the power of 1/(1 - alpha)"""

    # Define constants:
    pbound = 0.02
    toler = 1.0e-4

    diff = regtemp - tstar

    if regtemp.ndim == 1:
        myears = 1
        mcells = regtemp.shape[0]
        diff.reshape((1, mcells))
    elif regtemp.ndim == 2:
        myears = regtemp.shape[0]
        mcells = regtemp.shape[1]
    else:
        exit("Too many dimensions in temperature given to damage function.")

    if mcells != ncells:
        print("Number of cells is ", mcells, " not ", ncells)

    fval = np.zeros((myears, mcells))

    # ((1 - d) * exp(-κ_minus * (t - T) ^ 2) + d) ^ (1 / (1 - α))

    for iyear in range(myears):
        for icell in range(mcells):
            if diff[iyear, icell] < 0:
                fval[iyear, icell] = (
                    np.exp(-scale1 * diff[iyear, icell] * diff[iyear, icell])
                    * (1 - pbound)
                    + pbound
                ) ** (1 / (1 - alpha))
            else:
                fval[iyear, icell] = (
                    np.exp(-scale2 * diff[iyear, icell] * diff[iyear, icell])
                    * (1 - pbound)
                    + pbound
                ) ** (1 / (1 - alpha))

            if fval[iyear, icell] < toler:
                fval[iyear, icell] = toler

    if myears == 1:
        fval.reshape((mcells))

    return fval


def get_ai(expected_temperature):
    nyears = expected_temperature.shape[0]
    ncells = expected_temperature.shape[1]
    ai = np.zeros((nyears, ncells))
    expected_damages = damages(expected_temperature)

    ai[0, :] = get_initial_ai()

    for iyear in range(1, nyears):
        ai[iyear, :] = (
            (1 + ga)
            * ai[iyear - 1, :]
            * (expected_damages[iyear, :] / expected_damages[iyear - 1, :])
        )

    return ai


def get_temp_without_srm(iyear, emissions):
    temperature = (
        pi_temperature
        + gamma1 * emissions[iyear - 1990]
        + gamma2 * emissions[iyear - 1990] ** 2
    )

    return temperature


def get_srm_offset(iyear):
    if iyear <= 2130:
        offset = a_coeff * (1 - np.exp(b_coeff * (iyear - srm_start_year)))
    else:
        offset = const

    return offset


def get_expected_temperature(iyear, emissions):
    """Calculate the expected temperature using the expected cumulative emissions and
    regression coefficients"""

    # Check that we use correct year:
    if not iyear - 1989 == years[iyear - 1990]:
        (
            print(
                "Using wrong year of expected cumulative emissions. Should have ",
                iyear,
                ", using ",
                years[iyear - 1990],
            )
            + 1989
        )
        exit()

    else:
        expected_temperature = get_temp_without_srm(iyear, emissions)

        if iyear >= srm_start_year:
            expected_temperature += get_srm_offset(iyear)

    return expected_temperature


def add_bubble_label(fig, position, labels, label_values, title):
    # Generate legend to indicate GDP size:
    ax = fig.add_axes(position, frameon=False)
    ax.set_yticks([]), ax.set_xticks([])
    for i, value in enumerate(label_values):
        ax.scatter(
            [],
            [],
            c="None",
            edgecolor="black",
            linewidths=0.7,
            s=value,
            label=labels[i],
        )
    legend = ax.legend(
        scatterpoints=1,
        frameon=False,
        labelspacing=0.7,
        title=title,
        loc=2,
        fontsize=10,
    )
    legend.get_title().set_fontsize("12")


def add_country_names(
    ax,
    x,
    y,
    tx,
    ty,
    country_name,
    cmap,
    color,
    size,
    horizontalalignment,
    vmin=None,
    vmax=None,
    norm=None,
    text=True,
):

    ax.scatter(
        x,
        y,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        edgecolors="k",
        linewidth=0.2,
        alpha=0.8,
        label=None,
        c=color,
        s=size,
    )

    if text == True:
        ax.text(
            tx,
            ty,
            country_name,
            fontsize=10,
            zorder=3,
            horizontalalignment=horizontalalignment,
        )


def add_global_value(
    ax, x, y, color, size, cmap, vmin=None, vmax=None, norm=None, text=True
):
    ax.scatter(
        x,
        y,
        cmap=gdp_cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        c=color,
        linewidth=1,
        edgecolor="black",
        s=size,
        alpha=0.8,
    )

    if text == True:
        ax.text(
            x,
            y,
            "GLOBAL",
            fontsize=12,
        )


# --------------------------------------------------------------------------------------

# COMPARE FIXED POINTS

srm_fp_wealth = srm_fp_gdp + (1 - delta) * descale(srm_fp_capital, srm_ai)
srm_fp_capital = descale(srm_fp_capital, srm_ai)

# --------------------------------------------------------------------------------------


# CALCULATIONS

base_ai = get_ai(base_fp_regtemp)

cumulative_emissions_1990 = np.array([216.865])
srm_fp_cumulative_emissions = np.cumsum(
    np.concatenate((cumulative_emissions_1990, srm_fp_sum_actual_emissions))
)
base_fp_cumulative_emissions = np.cumsum(
    np.concatenate((cumulative_emissions_1990, base_fp_sum_actual_emissions))
)

for isim, sim in enumerate(simulations[2:]):
    if "base" in sim:
        fp_cumulative_emissions = base_fp_cumulative_emissions
        fp_sum_actual_emissions = base_fp_sum_actual_emissions
    else:
        fp_cumulative_emissions = srm_fp_cumulative_emissions
        fp_sum_actual_emissions = srm_fp_sum_actual_emissions

    exec("sim_years = " + sim + "_wealth.shape[0]")
    exec("actual_emissions = " + sim + "_actual_emissions")

    # Calculate emissions:
    sum_actual_emissions = np.sum(actual_emissions, axis=1)
    actual_cumulative_emissions = np.cumsum(
        np.concatenate((cumulative_emissions_1990, sum_actual_emissions / 1e3))
    )
    diff_cumulative_emissions = (
        actual_cumulative_emissions[:sim_years] - fp_cumulative_emissions[:sim_years]
    )
    diff_emissions = (
        sum_actual_emissions[:sim_years] / 1e3 - fp_sum_actual_emissions[:sim_years]
    )

    exec(sim + "_sum_actual_emissions = sum_actual_emissions")
    exec(sim + "_diff_cumulative_emissions = diff_cumulative_emissions")
    exec(sim + "_diff_emissions = diff_emissions")

for isim, sim in enumerate(simulations[1:]):
    exec("sim_years = " + sim + "_wealth.shape[0]")

    if "base" in sim:
        ai = base_ai
        fp_regtemp = base_fp_regtemp
    else:
        ai = srm_ai
        fp_regtemp = srm_fp_regtemp

    # Calculate wealth and GDP:
    exec(
        sim + "_wealth = " + sim + "_wealth / damages(fp_regtemp[:sim_years])"
    )  # Make up for bug in coupling script
    exec(sim + "_capital = descale(" + sim + "_capital[:sim_years], ai[:sim_years])")

    exec(sim + "_gdp = " + sim + "_wealth - (1 - delta) * " + sim + "_capital")


print(
    "Total diff",
    np.sum(srm_e1_sum_actual_emissions / 1e3 - srm_fp_sum_actual_emissions[:nyears]),
)


for isim, sim in enumerate(simulations):
    exec("sim_years = " + sim + "_wealth.shape[0]")
    exec("gdp = " + sim + "_gdp")
    exec("regtemp = " + sim + "_regtemp")

    sum_gdp = np.sum(gdp, axis=1)

    # Create empty arrays:
    gdp_detrended = np.zeros((sim_years, ncells))
    sum_gdpper_detrended = np.zeros(sim_years)
    pop_temp = np.zeros((sim_years))

    for iyear in range(sim_years):
        # Detrend the GDP:
        gdp_detrended[iyear, :] = gdp[iyear, :] / (1 + ga) ** iyear
        sum_gdpper_detrended[iyear] = (
            np.sum(gdp[iyear, :])
            * 1e9
            / ((1 + ga) ** iyear * np.sum(population[iyear, :] * 1e3))
        )

        # Calculate population weighted temperature:
        pop_temp[iyear] = np.average(
            regtemp[iyear, :] - pi_temperatures, weights=population[iyear, :]
        )

    sum_gdp_detrended = np.sum(gdp_detrended, axis=1)

    exec(sim + "_sum_gdp_detrended = sum_gdp_detrended")
    exec(sim + "_sum_gdpper_detrended = sum_gdpper_detrended")

    # Calculate area weighted temperature:
    area_temp = np.average(
        regtemp - pi_temperatures, axis=1, weights=np.cos(np.deg2rad(diam_latitudes))
    )

    exec(sim + "_pop_temp = pop_temp")
    exec(sim + "_area_temp = area_temp")

# --------------------------------------------------------------------------------------

# CALCULATE ENSEMBLE MEANS

avg_srm_pop_temp = np.average(
    np.stack((srm_e1_pop_temp, srm_e2_pop_temp, srm_e3_pop_temp)), axis=0
)
avg_srm_area_temp = np.average(
    np.stack((srm_e1_area_temp, srm_e2_area_temp, srm_e3_area_temp)), axis=0
)
avg_srm_global_temp = (
    np.average(
        np.stack((srm_e1_global_temp, srm_e2_global_temp, srm_e3_global_temp)), axis=0
    )
    - global_pi_temperature
)
avg_srm_sum_actual_emissions = np.average(
    np.stack(
        (
            srm_e1_sum_actual_emissions,
            srm_e2_sum_actual_emissions,
            srm_e3_sum_actual_emissions,
        )
    ),
    axis=0,
)

avg_base_sum_actual_emissions = np.average(
    np.stack(
        (
            base_e1_sum_actual_emissions,
            base_e2_sum_actual_emissions,
            base_e3_sum_actual_emissions,
        )
    ),
    axis=0,
)

avg_base_pop_temp = np.average(
    np.stack((base_e1_pop_temp, base_e2_pop_temp, base_e3_pop_temp)), axis=0
)
avg_base_area_temp = np.average(
    np.stack((base_e1_area_temp, base_e2_area_temp, base_e3_area_temp)), axis=0
)
avg_base_global_temp = (
    np.average(
        np.stack((base_e1_global_temp, base_e2_global_temp, base_e3_global_temp)),
        axis=0,
    )
    - global_pi_temperature
)

base_global_gdpper = np.average(
    np.stack(
        (
            base_e1_sum_gdpper_detrended,
            base_e2_sum_gdpper_detrended,
            base_e3_sum_gdpper_detrended,
        ),
        axis=0,
    ),
    axis=0,
)
srm_global_gdpper = np.average(
    np.stack(
        (
            srm_e1_sum_gdpper_detrended,
            srm_e2_sum_gdpper_detrended,
            srm_e3_sum_gdpper_detrended,
        ),
        axis=0,
    ),
    axis=0,
)


# --------------------------------------------------------------------------------------

# DEFINE AND EXTRACT COUNTRIES/REGIONS:

country_names = get_country_names()
all_indices = []


# Make empty dictionary with empty lists:
country_indices = defaultdict(list)
country_latitudes = defaultdict(list)
country_pops = defaultdict(list)
country_pop = defaultdict(float)

# Sort all indices for each country into the dictionary:
for index, country in enumerate(country_names):
    country_indices[country].append(index)
    country_latitudes[country].append(diam_latitudes[index])
    country_pop[country] = country_pop[country] + population[:, index]
    country_pops[country].append(population[:, index])

for isim, sim in enumerate(simulations):
    exec(sim + "_country_gdp = defaultdict(float)")

    for index, country in enumerate(country_names):
        exec(
            sim
            + "_country_gdp[country] = "
            + sim
            + "_country_gdp[country] + "
            + sim
            + "_gdp[:, index]"
        )


# Make list of all countries without duplicates:
all_countries = list(country_indices.keys())
n_countries = np.arange(len(all_countries))

# Remove some regions:
for c, country in enumerate(country_indices.keys()):
    # Remove if pop under 250k and GDP under 2:
    if country_pop[country][0] < 250 and srm_fp_country_gdp[0, 0, country] < 2:
        all_countries.remove(country)

# Make list of chosen countries:
chosen_countries = all_countries
text_countires = [
    "Norway",
    "United States",
    "Russia",
    "United Kingdom",
    "China",
    "Somalia",
    "Germany",
    "Sudan",
    "Canada",
    "New Zealand",
    "Spain",
    "Somalia",
    "Brazil",
    "India",
    "Saudi Arabia",
    "Iraq",
    "Niger",
    "Mali",
    "Namibia",
]  #'Algeria', 'Indonesia'

population_countries = np.zeros((nyears, len(chosen_countries)))
print(nyears)


for isim, sim in enumerate(simulations):
    exec("sim_years = " + sim + "_wealth.shape[0]")

    # Make arrays with the GDP, damages, and PI temperature of the chosen countries:
    exec(sim + "_gdp_country = np.zeros((sim_years, len(chosen_countries)))")
    exec(sim + "_gdpper_country = np.zeros((sim_years, len(chosen_countries)))")
    exec(sim + "_dtemp_country = np.zeros((sim_years, len(chosen_countries)))")
    exec(sim + "_temp_country = np.zeros((sim_years, len(chosen_countries)))")


for c, country in enumerate(chosen_countries):
    indices = country_indices[country]
    pops = np.asarray(country_pops[country]).T
    lat_weight = np.cos(np.deg2rad(country_latitudes[country]))
    population_countries[:, c] = np.asarray(country_pop[country][:nyears])

    for isim, sim in enumerate(simulations):
        exec("sim_years = " + sim + "_wealth.shape[0]")

        exec(sim + "_gdp_country[:, c] = " + sim + "_country_gdp[country] * 1e9")

        for iyear in range(sim_years):
            exec(
                sim
                + "_gdpper_country[iyear, c] = "
                + sim
                + "_gdp_country[iyear, c] / ((1 + ga) ** iyear * population_countries[iyear, c] * 1e3)"
            )
            exec(
                sim
                + "_dtemp_country[iyear, c] = calculate_regional_mean("
                + sim
                + "_regtemp[iyear, :] - pi_temperatures, indices, weights=pops[iyear, :])"
            )
            exec(
                sim
                + "_temp_country[iyear, c] = calculate_regional_mean("
                + sim
                + "_regtemp[iyear, :], indices, weights=pops[iyear, :])"
            )

base_dtemp_country = np.average(
    np.stack(
        (base_e1_dtemp_country, base_e2_dtemp_country, base_e3_dtemp_country), axis=0
    ),
    axis=0,
)
srm_dtemp_country = np.average(
    np.stack(
        (srm_e1_dtemp_country, srm_e2_dtemp_country, srm_e3_dtemp_country), axis=0
    ),
    axis=0,
)
base_gdp_country = np.average(
    np.stack((base_e1_gdp_country, base_e2_gdp_country, base_e3_gdp_country), axis=0),
    axis=0,
)
srm_gdp_country = np.average(
    np.stack((srm_e1_gdp_country, srm_e2_gdp_country, srm_e3_gdp_country), axis=0),
    axis=0,
)
base_gdpper_country = np.average(
    np.stack(
        (base_e1_gdpper_country, base_e2_gdpper_country, base_e3_gdpper_country), axis=0
    ),
    axis=0,
)
srm_gdpper_country = np.average(
    np.stack(
        (srm_e1_gdpper_country, srm_e2_gdpper_country, srm_e3_gdpper_country), axis=0
    ),
    axis=0,
)

start_temp_country = np.average(srm_fp_temp_country[:10, :], axis=0)


# --------------------------------------------------------------------------------------

# PLOT EMISSIONS AND AVERAGE TEMPERATURE AGAINST TIME

fig3, ax3 = plt.subplots(nrows=3, ncols=1, figsize=(10, 16))

years = np.arange(1990, 2101)

for isim, sim in enumerate(simulations[2:]):
    exec("sim_years = " + sim + "_wealth.shape[0]")
    if "base" in sim:
        color = "blue"
        start_year = 0
    else:
        color = "red"
        start_year = 39

    exec("sum_actual_emissions = " + sim + "_sum_actual_emissions")
    exec("pop_temp = " + sim + "_pop_temp")
    exec("area_temp = " + sim + "_area_temp")
    exec("global_temp = " + sim + "_global_temp - global_pi_temperature")
    exec("global_gdpper = " + sim + "_sum_gdpper_detrended")
    global_gdpper_change = (
        100 * (global_gdpper - base_global_gdpper[0]) / base_global_gdpper[0]
    )

    ax3[0].plot(
        years[start_year:sim_years],
        sum_actual_emissions[start_year:] / 1e3,
        linewidth=1,
        alpha=0.5,
        color=color,
    )

    ax3[1].plot(
        years[start_year:sim_years],
        pop_temp[start_year:],
        linewidth=1,
        alpha=0.4,
        color=color,
    )

    ax3[1].plot(
        years[start_year:sim_years],
        area_temp[start_year:],
        linewidth=1,
        alpha=0.4,
        color=color,
        linestyle=":",
    )
    #    ax3[1].plot(
    #        years[start_year:sim_years],
    #        global_temp[start_year:],
    #        linewidth=1,
    #        alpha=0.4,
    #        color=color,
    #        linestyle="--",
    #    )

    ax3[2].plot(
        years[start_year:sim_years],
        global_gdpper_change[start_year:],
        linewidth=1,
        alpha=0.4,
        color=color,
    )


ax3[0].plot(
    years[:sim_years],
    avg_base_sum_actual_emissions / 1e3,
    linewidth=3,
    color="blue",
    label="Without SRM",
)
ax3[1].plot(
    years[:sim_years],
    avg_base_pop_temp,
    linewidth=3,
    color="blue",
)
ax3[1].plot(
    years[:sim_years],
    avg_base_area_temp,
    linewidth=3,
    color="blue",
    linestyle=":",
)
# ax3[1].plot(
#    years[:sim_years],
#    avg_base_global_temp,
#    linewidth=3,
#    color="blue",
#    linestyle="--",
# )
ax3[2].plot(
    years[:sim_years],
    100 * (base_global_gdpper - base_global_gdpper[0]) / base_global_gdpper[0],
    linewidth=3,
    color="blue",
)

ax3[0].plot(
    years[39:sim_years],
    avg_srm_sum_actual_emissions[39:] / 1e3,
    linewidth=3,
    color="red",
    label="With SRM",
)
ax3[1].plot(
    years[39:sim_years],
    avg_srm_pop_temp[39:],
    linewidth=3,
    color="red",
)
ax3[1].plot(
    years[39:sim_years],
    avg_srm_area_temp[39:],
    linewidth=3,
    color="red",
    linestyle=":",
)
# ax3[1].plot(
#    years[39:sim_years],
#    avg_srm_global_temp[39:],
#    linewidth=3,
#    color="red",
#    linestyle="--",
# )
ax3[2].plot(
    years[39:sim_years],
    100 * (srm_global_gdpper[39:] - base_global_gdpper[0]) / base_global_gdpper[0],
    linewidth=3,
    color="red",
)

ax3[1].plot(0, 0, color="k", label="Area-weighted", linewidth=3, linestyle=":")
ax3[1].plot(0, 0, color="k", label="Population-weighted", linewidth=3)
# ax3[1].plot(0, 0, color="k", label="Global", linewidth=3, linestyle="--")


ax3[2].set_xlabel("Year", fontsize=20)
ax3[0].set_ylabel("Emissions (GtC)", fontsize=20)
ax3[1].set_ylabel("Temperature change (\N{DEGREE SIGN}C)", fontsize=20)
ax3[2].set_ylabel("GDP per capita change (%)", fontsize=20)

for ax in ax3:
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    ax.legend(fontsize=20)
    ax.set_xlim(1985, 2105)

fig3.text(0.01, 0.98, "(a)", fontsize=16, wrap=True)
fig3.text(0.01, 0.65, "(b)", fontsize=16, wrap=True)
fig3.text(0.01, 0.34, "(c)", fontsize=16, wrap=True)

fig3.subplots_adjust(left=0.1, right=0.98, top=0.96, bottom=0.05, hspace=0.1)

fig3.savefig("../../figures/emissions_temperature_gdpper_compare.pdf")

# --------------------------------------------------------------------------------------

print(
    "SRM last 20 years: ",
    np.mean(avg_srm_area_temp[-20:]),
    np.mean(avg_srm_pop_temp[-20:]),
)
print(
    "SRM last 10 years: ",
    np.mean(avg_srm_area_temp[-10:]),
    np.mean(avg_srm_pop_temp[-10:]),
    np.mean((srm_global_gdpper[-10:] - base_global_gdpper[0]) / base_global_gdpper[0]),
)
print(
    "Baseline last 10 years: ",
    np.mean(avg_base_area_temp[-10:]),
    np.mean(avg_base_pop_temp[-10:]),
    np.mean((base_global_gdpper[-10:] - base_global_gdpper[0]) / base_global_gdpper[0]),
)
print(
    "2020s:",
    np.mean(
        np.mean(
            (srm_global_gdpper[30:40] - base_global_gdpper[0]) / base_global_gdpper[0]
        ),
    ),
)

for iyear in range(110):
    print(iyear + 1990, avg_base_area_temp[iyear], avg_base_pop_temp[iyear])

print(
    "Difference in emissions: ",
    avg_srm_sum_actual_emissions - avg_base_sum_actual_emissions,
)
print(
    "Difference in emissions (%): ",
    100
    * (avg_srm_sum_actual_emissions - avg_base_sum_actual_emissions)
    / avg_base_sum_actual_emissions,
)
print(
    "Total difference: ",
    np.sum(avg_srm_sum_actual_emissions[40:] - avg_base_sum_actual_emissions[40:]),
    100
    * np.sum(avg_srm_sum_actual_emissions[40:] - avg_base_sum_actual_emissions[40:])
    / np.sum(avg_base_sum_actual_emissions),
)

# --------------------------------------------------------------------------------------

# SPECIFY COLOUR MAPS

ncolors = 11
colors = sns.color_palette("YlOrRd", ncolors).as_hex()
vmin = -3
vmax = 30
color_bins = np.linspace(vmin, vmax, ncolors + 1)


gdp_cmap = ListedColormap(colors)

# https://matplotlib.org/stable/gallery/color/custom_cmap.html
cdict3 = {
    "red": (
        (0.0, 0.0, 0.0),
        (0.25, 0.0, 0.0),
        (0.5, 0.8, 1.0),
        (0.75, 1.0, 1.0),
        (1.0, 0.4, 1.0),
    ),
    "green": (
        (0.0, 0.0, 0.0),
        (0.25, 0.0, 0.0),
        (0.5, 0.9, 0.9),
        (0.75, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    ),
    "blue": (
        (0.0, 0.0, 0.4),
        (0.25, 1.0, 1.0),
        (0.5, 1.0, 0.8),
        (0.75, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    ),
}


cmap = LinearSegmentedColormap("BlueRed3", cdict3)
cmap_red = cmr.get_sub_cmap(cmap, 0.5, 1)
cmap_blue = cmr.get_sub_cmap(cmap, 0.33, 0.5)

colors1 = cmap_blue(np.linspace(0.0, 1, 128))
colors2 = cmap_red(np.linspace(0, 1, 128))

# combine them and build a new colormap
colors = np.vstack((colors1, colors2))
population_cmap = LinearSegmentedColormap.from_list("my_colormap", colors)

vmin2 = -100
vmax2 = 300
divnorm = TwoSlopeNorm(vmin=vmin2, vcenter=0, vmax=vmax2)

# --------------------------------------------------------------------------------------

# PLOT CHANGE OF LAST DECADE

ndecades = nyears // 10

print("Decade:", 1990 + (ndecades - 1) * 10, "-", 2000 + (ndecades - 1) * 10)
print(ndecades, 10 * (ndecades - 1), 10 * ndecades)


expected_gdpper_start = np.average(srm_fp_gdpper_country[:10, :], axis=0)
expected_gdpper_2020s = np.average(srm_fp_gdpper_country[30:40, :], axis=0)

expected_srm_gdpper_country_decade = (
    100
    * (
        np.average(
            srm_fp_gdpper_country[10 * (ndecades - 1) : 10 * ndecades, :], axis=0
        )
        - expected_gdpper_2020s
    )
    / expected_gdpper_2020s
)

srm_dgdpper_country_decade = (
    100
    * (
        np.average(srm_gdpper_country[10 * (ndecades - 1) : 10 * ndecades, :], axis=0)
        - expected_gdpper_2020s
    )
    / expected_gdpper_2020s
)
base_dgdpper_country_decade = (
    100
    * (
        np.average(base_gdpper_country[10 * (ndecades - 1) : 10 * ndecades, :], axis=0)
        - expected_gdpper_2020s
    )
    / expected_gdpper_2020s
)


fp_dtemp_country_decade = np.average(
    srm_fp_dtemp_country[10 * (ndecades - 1) : 10 * ndecades, :], axis=0
)
srm_dtemp_country_decade = np.average(
    srm_dtemp_country[10 * (ndecades - 1) : 10 * ndecades, :], axis=0
) - np.average(srm_dtemp_country[30:40, :], axis=0)
base_dtemp_country_decade = np.average(
    base_dtemp_country[10 * (ndecades - 1) : 10 * ndecades, :], axis=0
) - np.average(base_dtemp_country[30:40, :], axis=0)

dpopulation_country_decade = (
    100
    * (
        np.average(population_countries[10 * (ndecades - 1) : 10 * ndecades, :], axis=0)
        - np.average(population_countries[30:40, :], axis=0)
    )
    / np.average(population_countries[30:40, :], axis=0)
)
print(np.max(dpopulation_country_decade), np.min(dpopulation_country_decade))

# --------------------------------------------------------------------------------------

fig5, ax5 = plt.subplots(nrows=3, ncols=1, figsize=(7.5, 15))

pscat1 = ax5[0].scatter(
    base_dtemp_country_decade,
    base_dgdpper_country_decade,
    cmap=population_cmap,
    norm=divnorm,
    linewidth=0.2,
    alpha=0.8,
    label=None,
    c=dpopulation_country_decade,
    s=np.sqrt(population_countries[30, :] * 1e3 / 1e3),
)

pscat1 = ax5[1].scatter(
    srm_dtemp_country_decade,
    srm_dgdpper_country_decade,
    cmap=population_cmap,
    norm=divnorm,
    linewidth=0.2,
    alpha=0.8,
    label=None,
    c=dpopulation_country_decade,
    s=np.sqrt(population_countries[30, :] * 1e3 / 1e3),
)

pscat1 = ax5[2].scatter(
    srm_dtemp_country_decade - base_dtemp_country_decade,
    srm_dgdpper_country_decade - base_dgdpper_country_decade,
    cmap=population_cmap,
    norm=divnorm,
    linewidth=0.2,
    alpha=0.8,
    label=None,
    c=dpopulation_country_decade,
    s=np.sqrt(population_countries[30, :] * 1e3 / 1e3),
)

# Add country names:
for c, country in enumerate(all_countries):
    if country in text_countires:
        x = base_dtemp_country_decade[c]
        y = base_dgdpper_country_decade[c]
        tx = x
        ty = y
        alignment = "left"

        if (
            country == "United States"
            or country == "Somalia"
            or country == "Mali"
            or country == "Spain"
        ):
            alignment = "right"
        elif country == "China" or country == "India":
            alignment = "center"

        print(country, alignment)

        add_country_names(
            ax=ax5[0],
            x=x,
            y=y,
            tx=tx,
            ty=ty,
            country_name=country,
            cmap=population_cmap,
            color=dpopulation_country_decade[c],
            size=np.sqrt(population_countries[30, c] * 1e3 / 1e3),
            horizontalalignment=alignment,
            norm=divnorm,
        )

        x = srm_dtemp_country_decade[c]
        y = srm_dgdpper_country_decade[c]
        tx = x
        ty = y
        alignment = "left"

        if country == "Canada":
            alignment = "right"
        elif country == "United Kingdom":
            alignment = "center"
            ty = y - 2.5
            tx = x - 0.1
        elif country == "United States":
            ty = y + 2
        elif country == "Somalia":
            ty = y - 3
        elif country == "Russia":
            ty = y - 1
            alignment = "center"
        elif country == "Saudi Arabia":
            ty = y - 1.5
        elif country == "China":
            alignment = "center"
        elif country == "India" or country == "Niger":
            ty = y + 1

        add_country_names(
            ax=ax5[1],
            x=x,
            y=y,
            tx=tx,
            ty=ty,
            country_name=country,
            cmap=population_cmap,
            color=dpopulation_country_decade[c],
            size=np.sqrt(population_countries[30, c] * 1e3 / 1e3),
            horizontalalignment=alignment,
            norm=divnorm,
        )

        x = srm_dtemp_country_decade[c] - base_dtemp_country_decade[c]
        y = srm_dgdpper_country_decade[c] - base_dgdpper_country_decade[c]
        tx = x
        ty = y
        alignment = "left"

        if country == "United States" or country == "Russia" or country == "Namibia":
            alignment = "right"
        elif country == "China" or country == "India":
            alignment = "center"

        add_country_names(
            ax=ax5[2],
            x=x,
            y=y,
            tx=tx,
            ty=ty,
            country_name=country,
            cmap=population_cmap,
            color=dpopulation_country_decade[c],
            size=np.sqrt(population_countries[30, c] * 1e3 / 1e3),
            horizontalalignment=alignment,
            norm=divnorm,
        )


# Calculate global values:
base_dtemp_global_decade = np.average(
    avg_base_pop_temp[10 * (ndecades - 1) : 10 * ndecades]
) - np.average(avg_base_pop_temp[30:40])
base_dgdpper_global_decade = (
    100
    * (
        np.average(base_global_gdpper[10 * (ndecades - 1) : 10 * ndecades])
        - np.average(srm_fp_sum_gdpper_detrended[30:40])
    )
    / np.average(srm_fp_sum_gdpper_detrended[30:40])
)
srm_dtemp_global_decade = np.average(
    avg_srm_pop_temp[10 * (ndecades - 1) : 10 * ndecades]
) - np.average(avg_base_pop_temp[30:40])
srm_dgdpper_global_decade = (
    100
    * (
        np.average(srm_global_gdpper[10 * (ndecades - 1) : 10 * ndecades])
        - np.average(srm_fp_sum_gdpper_detrended[30:40])
    )
    / np.average(srm_fp_sum_gdpper_detrended[30:40])
)
dpopulation_global_decade = (
    100
    * (
        np.average(global_population[10 * (ndecades - 1) : 10 * ndecades])
        - np.average(global_population[30:40])
    )
    / np.average(global_population[30:40])
)

# Add global value:
add_global_value(
    ax=ax5[0],
    x=base_dtemp_global_decade,
    y=base_dgdpper_global_decade,
    color=dpopulation_global_decade,
    size=50,
    cmap=population_cmap,
    norm=divnorm,
    text=True,
)
add_global_value(
    ax=ax5[1],
    x=srm_dtemp_global_decade,
    y=srm_dgdpper_global_decade,
    color=dpopulation_global_decade,
    size=50,
    cmap=population_cmap,
    norm=divnorm,
    text=True,
)

add_global_value(
    ax=ax5[2],
    x=srm_dtemp_global_decade - base_dtemp_global_decade,
    y=srm_dgdpper_global_decade - base_dgdpper_global_decade,
    color=dpopulation_global_decade,
    size=50,
    cmap=population_cmap,
    norm=divnorm,
    text=True,
)


# Generate legend to indicate population size:
add_bubble_label(
    fig=fig5,
    position=[0.83, 0.577, 0.02, 0.4],
    labels=["10$^5$", "10$^{6}$", "10$^{7}$", "10$^{8}$", "10$^{9}$"],
    label_values=[
        np.sqrt(1e5 / 1e3),
        np.sqrt(1e6 / 1e3),
        np.sqrt(1e7 / 1e3),
        np.sqrt(1e8 / 1e3),
        np.sqrt(1e9 / 1e3),
    ],
    title="Year 2020\npopulation",
)

"""
# Generate legend to indicate GDP size:
add_bubble_label(
    fig=fig5,
    position=[0.83, 0.243, 0.02, 0.4],
    labels=["100$", "1000$", "10 000$", "100 000$"],
    label_values=[np.sqrt(1e2), np.sqrt(1e3), np.sqrt(1e4), np.sqrt(1e5)],
    title="Initial\nGDP/capita",
)
"""

# Generate color bar to indicate population change:
cbar_ax = fig5.add_axes([0.86, 0.699, 0.02, 0.14])
cbar = fig5.colorbar(pscat1, cax=cbar_ax)
cbar.set_label(
    "Population change (%)",
    fontsize=12,
    rotation=270,
    labelpad=18,
)

cbar.set_ticks([-100, 0, 100, 200, 300])
cbar.ax.tick_params(labelsize=10)


"""
# Generate color bar to indicate 2000 temperature:
cbar_ax = fig5.add_axes([0.86, 0.369, 0.02, 0.14])
cbar = fig5.colorbar(pscat2, ticks=color_bins[1:-1], cax=cbar_ax)
cbar.set_label(
    "Initial temperature (\N{DEGREE SIGN}C)",
    fontsize=12,
    rotation=270,
    labelpad=18,
)
cbar.ax.tick_params(labelsize=10)
"""
for ax in ax5:
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)

    # Add the 0-line:
    ax.axhline(0, color="grey", alpha=0.6, linestyle="--", linewidth=1)


ax5[0].set_xlim(-2.5, 3.3)
ax5[1].set_xlim(-2.5, 3.3)
ax5[2].set_xlim(-4.2, 1)
ax5[0].set_ylim(-43, 40)
ax5[1].set_ylim(-43, 40)
ax5[2].set_ylim(-42, 52)

ax5[0].set_xlabel(r"$\Delta$temperature " + "(\N{DEGREE SIGN}C)", fontsize=14)
ax5[1].set_xlabel(r"$\Delta$temperature " + "(\N{DEGREE SIGN}C)", fontsize=14)
ax5[2].set_xlabel(
    r"Difference in $\Delta$temperature " + "(\N{DEGREE SIGN}C)", fontsize=14
)
ax5[0].set_ylabel(r"$\Delta$GDP/capita (%)", fontsize=14)
ax5[1].set_ylabel(r"$\Delta$GDP/capita (%)", fontsize=14)
ax5[2].set_ylabel(r"Difference in $\Delta$GDP/capita (%)", fontsize=14)

ax5[0].set_title("Without SRM", fontsize=14)
ax5[1].set_title("With SRM", fontsize=14)
ax5[2].set_title("With SRM - without SRM", fontsize=14)

fig5.text(0.01, 0.98, "(a)", fontsize=12, wrap=True)
fig5.text(0.01, 0.65, "(b)", fontsize=12, wrap=True)
fig5.text(0.01, 0.33, "(c)", fontsize=12, wrap=True)

fig5.subplots_adjust(left=0.11, right=0.83, top=0.97, bottom=0.04, hspace=0.22)

fig5.savefig("../../figures/country_difference_gdpper_percent_SRM_2090s-2020s.pdf")
fig5.savefig("../../figures/country_difference_gdpper_percent_SRM_2090s-2020s.png")

plt.close()

# --------------------------------------------------------------------------------------

# PLOT HISTOGRAM

colors = sns.color_palette("RdYlBu", 5).as_hex()
colors[2] = "#A9A9A9"

total_gdp = np.sum(srm_gdp_country, axis=1)
initial_gdp_share = 100 * srm_gdp_country[0, :] / total_gdp[0]

df = pd.DataFrame(initial_gdp_share, columns=["initial_gdp_share"], index=all_countries)

df["gdp_per_capita_1990"] = srm_gdpper_country[0, :]
df["gdpper_diff"] = srm_dgdpper_country_decade - base_dgdpper_country_decade
print(df)
df["initial_population"] = population_countries[0, :]
print(population_countries[0, :])


# gdp_bin = [3750, 7500, 15000, 30000, 200000]
gdp_bin = [3500, 7000, 14000, 28000, 200000]
gdp_bin = [3000, 6000, 12000, 24000, 200000]
for ig, gdpcap in enumerate(gdp_bin[::-1]):
    df.loc[df["gdp_per_capita_1990"] < gdpcap, "color"] = colors[ig]


df.sort_values(["gdpper_diff"], inplace=True)

df["cum_gdp_share"] = df["initial_gdp_share"].copy()

for i in np.arange(1, len(df)):
    df["cum_gdp_share"].iloc[i] = (
        df["cum_gdp_share"].iloc[i] + df["cum_gdp_share"].iloc[i - 1]
    )

bins = np.array([0] + df["cum_gdp_share"].values.tolist())
widths = bins[1:] - bins[:-1]
heights = df["gdpper_diff"]

print(df)
print(df["gdp_per_capita_1990"]["United States"])
print(df["cum_gdp_share"]["United States"])

# Write data to file:
df.to_csv("../../data/output_figures/geoengineering_histogram_data.csv")

fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
p_bar = plt.bar(
    bins[:-1], heights, width=widths, color=df["color"], align="edge", alpha=0.8
)

# Add country names:
for c, country in enumerate(df.index):
    if country in text_countires:
        x = bins[c + 1]
        y = df["gdpper_diff"][country] + 0.7
        color = df["color"][c]
        alignment = "right"

        if country in ["Norway", "Russia", "Canada", "Germany"]:
            y -= 2.5
            x = bins[c]
            alignment = "left"
            if country == "Norway":
                y -= 2
                # Add line:
                ax1.vlines(
                    bins[c] + 0.5 * (bins[c + 1] - bins[c]),
                    y + 2,
                    df["gdpper_diff"][country] - 0.7,
                    linewidth=0.5,
                    color=color,
                )
        elif country == "United States":
            x = bins[c] + 0.5 * (bins[c + 1] - bins[c])
            alignment = "center"
        elif country == "Namibia":
            y += 1
        elif country == "Iraq":
            y += 2
            # ax1.vlines(bins[c] + 0.5*(bins[c+1] - bins[c]), df["gdpper_diff"][country]+0.7,y-2, linewidth=0.5, color=color)

        ax1.text(x, y, country, horizontalalignment=alignment, fontsize=10, color=color)


plt.xlim([-0.1, 100.1])
plt.ylim([-35, 55])

plt.xticks(size=16)
plt.yticks(size=16)
plt.ylabel("GDP per capita difference due to solar reduction (%)", size=16)
plt.xlabel("Cumulative share of global GDP (%)", size=16)

# plot colorbar
ax_cb = fig1.add_axes([0.14, 0.55, 0.03, 0.28], frame_on=False)

colors_4cb = ListedColormap(colors[::-1])
norm = BoundaryNorm([0] + gdp_bin, 1 + colors_4cb.N)
cb = ColorbarBase(ax_cb, cmap=colors_4cb, norm=norm, ticks=gdp_bin[:-1], alpha=0.8)
cb.set_label("GDP per capita in 1990\n(1990 US$)", rotation=270, size=14, labelpad=35)
cb.ax.tick_params(labelsize=14)

plt.savefig(
    "../../figures/histogram_GDPper_difference_GDP_share.pdf",
    dpi=300,
    bbox_inches="tight",
)

# --------------------------------------------------------------------------------------
