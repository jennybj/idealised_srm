import sys as sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, "../")
from module_coupling import (
    get_coefficients,
    get_country_names,
    get_initial_gdpnet,
    get_initial_population,
    get_pi_temperature,
)

# ---------------------------------------------------------------------------

# READ IN DATA

onlyCO2 = np.loadtxt("temperatures_onlyCO2_start_1990.txt")
temperature_geoe_370 = np.loadtxt("temperatures_1percRed_start_2030.txt")[:, 2:]
temperature_base_585 = np.loadtxt("temperatures_ssp585_onlyCO2_start_2015.txt")[:, 2:]
temperature_geoe_585 = np.loadtxt("temperatures_ssp585_1percRed_start_2030.txt")[:, 2:]

list_latitudes = onlyCO2[:, 0]
list_longitudes = onlyCO2[:, 1]
temperature_base_370 = onlyCO2[:, 2:]

population = get_initial_population()
country_names = get_country_names()
gdp = get_initial_gdpnet()
ncells = population.shape[0]
nyears_base_370 = temperature_base_370.shape[1]
nyears_geoe_370 = temperature_geoe_370.shape[1]
nyears_base_585 = temperature_base_585.shape[1]
nyears_geoe_585 = temperature_geoe_585.shape[1]

pi_temperature = get_pi_temperature()

gamma1, gamma2, dummy = get_coefficients()

years, cumulative_emissions_ssp370, cumulative_emissions_ssp585 = np.loadtxt(
    "SSP_cumulative_emissions.txt",
    usecols=(0, 3, 4),
    skiprows=6,
    unpack=True,
)  # from couple_noresm2_diam

gdp_skorea = 0
gdp_nkorea = 0
gdp_russia = 0
gdp_china = 0
pop_skorea = 0
pop_nkorea = 0
pop_russia = 0
pop_china = 0

for icell in range(ncells):
    if country_names[icell] == "South Korea":
        print(gdp[icell])
        gdp_skorea += gdp[icell]
        pop_skorea += gdp[icell]

    if country_names[icell] == "North Korea":
        gdp_nkorea += gdp[icell]
        pop_nkorea += gdp[icell]

    if country_names[icell] == "Russia":
        gdp_russia += gdp[icell]
        pop_russia += gdp[icell]

    if country_names[icell] == "China":
        gdp_china += gdp[icell]
        pop_china += gdp[icell]


# ---------------------------------------------------------------------------

# CALCULATE

coordinates = list(zip(list_latitudes, list_longitudes))
weights = np.zeros(ncells)

# Create latitude weights that only count each grid cell once:
for icell in range(ncells):
    if coordinates[icell] not in coordinates[:icell]:
        weights[icell] = np.cos(np.deg2rad(list_latitudes[icell]))

"""
# Calculate temperature difference:
global_temp_base_370 = np.average(
    temperature_base_370 - np.tile(pi_temperature, (nyears_base_370, 1)).T,
    weights=weights,
    axis=0,
)
global_temp_geoe_370 = np.average(
    temperature_geoe_370 - np.tile(pi_temperature, (nyears_geoe_370, 1)).T,
    weights=weights,
    axis=0,
)

global_temp_base_585 = np.average(
    temperature_base_585 - np.tile(pi_temperature, (nyears_base_585, 1)).T,
    weights=weights,
    axis=0,
)
global_temp_geoe_585 = np.average(
    temperature_geoe_585 - np.tile(pi_temperature, (nyears_geoe_585, 1)).T,
    weights=weights,
    axis=0,
)
"""
expected_temp_base_370 = np.zeros((ncells, nyears_base_370))
expected_temp_geoe_370 = np.zeros((ncells, nyears_geoe_370))
expected_temp_base_585 = np.zeros((ncells, nyears_base_585))
expected_temp_geoe_585 = np.zeros((ncells, nyears_geoe_585))
expected_370 = np.zeros((ncells, nyears_geoe_370))
expected_585 = np.zeros((ncells, nyears_geoe_585))

for iyear in range(nyears_base_370):
    expected_temp_base_370[:, iyear] = (
        gamma1 * cumulative_emissions_ssp370[iyear]
        + gamma2 * cumulative_emissions_ssp370[iyear] ** 2
    )

for iyear in range(nyears_base_585):
    expected_temp_base_585[:, iyear] = (
        gamma1 * cumulative_emissions_ssp585[25 + iyear]
        + gamma2 * cumulative_emissions_ssp585[25 + iyear] ** 2
    )

"""
global_expected_temp_base_370 = np.average(
    expected_temp_base_370, weights=weights, axis=0
)
global_expected_temp_base_585 = np.average(
    expected_temp_base_585, weights=weights, axis=0
)

# Calculate the expected average difference with SRM (from 10 years):
for icell in range(ncells):
    expected_temp_geoe_370[icell, :] = expected_temp_base_370[icell, 40:] - np.average(
        expected_temp_base_370[icell, 50:]
        - (temperature_geoe_370[icell, 10:] - pi_temperature[icell])
    )
    expected_370[icell, :] = np.average(expected_temp_geoe_370[icell, 10:])

for icell in range(ncells):
    expected_temp_geoe_585[icell, :] = expected_temp_base_585[icell, 40:] - np.average(
        expected_temp_base_585[icell, 50:]
        - (temperature_geoe_585[icell, 10:] - pi_temperature[icell])
    )
    expected_585[icell, :] = np.average(expected_temp_geoe_585[icell, 10:])
"""
global_expected_temp_geoe_370 = np.average(
    expected_temp_geoe_370, weights=weights, axis=0
)
global_expected_370 = np.average(expected_370, weights=weights, axis=0)

global_expected_temp_geoe_585 = np.average(
    expected_temp_geoe_585, weights=weights, axis=0
)
global_expected_585 = np.average(expected_585, weights=weights, axis=0)

diff_370 = (
    temperature_geoe_370
    - np.tile(pi_temperature, (nyears_geoe_370, 1)).T
    - expected_temp_base_370[:, 40:]
)
global_diff_370 = np.average(diff_370, axis=0, weights=weights)

diff_585 = (
    temperature_geoe_585
    - np.tile(pi_temperature, (nyears_geoe_585, 1)).T
    - expected_temp_base_585[:, 15:]
)
global_diff_585 = np.average(diff_585, axis=0, weights=weights)


# ---------------------------------------------------------------------------

xdata = np.arange(70)
xdata2 = np.concatenate((xdata, xdata))
diff_both = np.concatenate((diff_370, diff_585), axis=1)

nyears = 110
years = np.arange(nyears)


def f(x, a, b):
    return -a * (1 - np.exp(-b * x))


fit = np.zeros((3, ncells, nyears))
a_coeffs = np.zeros((3, ncells))
b_coeffs = np.zeros((3, ncells))
fails = []

fails3 = 0

for icell in range(ncells):
    try:
        popt, covt = curve_fit(f, xdata, diff_370[icell], bounds=(0, np.inf))
        a_coeffs[0, icell] = popt[0]
        b_coeffs[0, icell] = popt[1]

        if a_coeffs[0, icell] < 0:
            print(icell, a_coeffs[0, icell])

    except RuntimeError:
        print("Error: curve_fit for SSP370 failed in grid cell ", icell)

    fit[0, icell, :] = f(years, popt[0], popt[1])

    try:
        popt, covt = curve_fit(f, xdata, diff_585[icell], bounds=(0, np.inf))
        a_coeffs[1, icell] = popt[0]
        b_coeffs[1, icell] = popt[1]

        if a_coeffs[1, icell] < 0:
            print(icell, a_coeffs[1, icell])

    except RuntimeError:
        print("Error: curve_fit for SSP585 failed in grid cell ", icell)

    fit[1, icell, :] = f(years, popt[0], popt[1])

    try:
        popt, covt = curve_fit(f, xdata2, diff_both[icell], bounds=(0, np.inf))
        a_coeffs[2, icell] = popt[0]
        b_coeffs[2, icell] = popt[1]

        if a_coeffs[2, icell] < 0:
            print(icell, a_coeffs[1, icell])

    except RuntimeError:
        print("Error: curve_fit for SSP370+SSP585 failed in grid cell ", icell)
        fails.append(icell)

    fit[2, icell, :] = f(years, a_coeffs[2, icell], b_coeffs[2, icell])

    # print(icell, popt)

mean_a_370 = np.average(a_coeffs[0, :], weights=weights)
mean_b_370 = np.average(b_coeffs[0, :], weights=weights)
mean_a_585 = np.average(a_coeffs[1, :], weights=weights)
mean_b_585 = np.average(b_coeffs[1, :], weights=weights)
mean_a_all = np.average(a_coeffs[2, :], weights=weights)
mean_b_all = np.average(b_coeffs[2, :], weights=weights)


print("Number of fails: ", len(fails))
print(
    np.max(a_coeffs[2, :]),
    np.min(a_coeffs[2, :]),
    np.max(b_coeffs[2, :]),
    np.min(b_coeffs[2, :]),
)
print(mean_a_370, mean_b_370, mean_a_585, mean_b_585, mean_a_all, mean_b_all)

for icell in fails:
    print(icell)
    end_temp = np.average(
        np.concatenate((diff_370[icell, -10:], diff_585[icell, -10:]))
    )
    a_coeffs[2, icell] = -end_temp
    b_coeffs[2, icell] = mean_b_all

    fit[2, icell, :] = f(years, a_coeffs[2, icell], b_coeffs[2, icell])

global_fit = np.average(fit, axis=1, weights=weights)

# ---------------------------------------------------------------------------

# SET FINAL OFFSET
# Should be same as a_coeff, unless temperature change is far from this at the end.

final_offset = a_coeffs[2, :]
count = 0

for icell in range(ncells):
    if np.abs(fit[2, icell, 100]) > 1.1 * np.abs(a_coeffs[2, icell]) or np.abs(
        fit[2, icell, 100]
    ) < 0.9 * np.abs(a_coeffs[2, icell]):
        print(
            icell,
            np.abs(fit[2, icell, -1]),
            np.abs(a_coeffs[2, icell]),
        )
        final_offset[icell] = -fit[2, icell, 100]
        count += 1

    fit[2, icell, 100:] = -final_offset[icell]


print(count)

# ---------------------------------------------------------------------------

# WRITE COEFFICIENTS TO FILE

file1 = open("investigating_SRM_coefficients.txt", "w")
file2 = open("SRM_coefficients.txt", "w")

file1.write(
    "# Temperature expectation to add to the expected temperature after SRM is added.\n"
)
file1.write(
    "# a*(1 - exp(b*t)) where t is the number of years since SRM was initiated.\n"
)
file1.write("# Column 1: Latitudes\n")
file1.write("# Column 2: Longitudes\n")
file1.write("# Column 3: Offset coefficient (a) SSP3-7.0\n")
file1.write("# Column 4: Exponent coefficent (b) SSP3-7.0\n")
file1.write("# Column 5: Offset coefficient (a) SSP5-8.5\n")
file1.write("# Column 6: Exponent coefficent (b) SSP5-8.5\n")
file1.write("# Column 7: Offset coefficient (a) both\n")
file1.write("# Column 8: Exponent coefficent (b) both\n")
file1.write("# Column 9: Country name\n")

file2.write(
    "# Temperature expectation to add to the expected temperature after SRM is added.\n"
)
file2.write(
    "# a*(1 - exp(b*t)) where t is the number of years since SRM was initiated.\n"
)
file2.write("# Column 1: Latitudes\n")
file2.write("# Column 2: Longitudes\n")
file2.write("# Column 3: Offset coefficient (a)\n")
file2.write("# Column 4: Exponent coefficent (b)\n")
file2.write("# Column 5: Constant offset after 100 years\n")
file2.write("# Column 6: Country name\n")


for c, country in enumerate(country_names):
    file1.writelines(["%16.1f" % list_latitudes[c]])
    file1.writelines(["%16.1f" % list_longitudes[c]])
    file1.writelines(["%16.7f" % -a_coeffs[0, c]])
    file1.writelines(["%16.7f" % -b_coeffs[0, c]])
    file1.writelines(["%16.7f" % -a_coeffs[1, c]])
    file1.writelines(["%16.7f" % -b_coeffs[1, c]])
    file1.writelines(["%16.7f" % -a_coeffs[2, c]])
    file1.writelines(["%16.7f" % -b_coeffs[2, c]])
    file1.writelines("    ")
    file1.writelines(country.replace(" ", "_"))
    file1.write("\n")

    file2.writelines(["%16.1f" % list_latitudes[c]])
    file2.writelines(["%16.1f" % list_longitudes[c]])
    file2.writelines(["%16.7f" % -a_coeffs[2, c]])
    file2.writelines(["%16.7f" % -b_coeffs[2, c]])
    file2.writelines(["%16.7f" % -final_offset[c]])
    file2.writelines("    ")
    file2.writelines(country.replace(" ", "_"))
    if c in fails:
        file2.writelines("*")
    file2.write("\n")

file1.close()
file2.close()

# ---------------------------------------------------------------------------


# PLOT


# Plot global average:

years = np.arange(1990, 1990 + nyears_base_370)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))

ax.plot(years[40:], global_diff_370, label="temp diff ssp370")
ax.plot(years[40:], global_fit[0, :70], label="fit ssp370")
ax.plot(years[40:], global_diff_585, label="temp diff ssp585")
ax.plot(years[40:], global_fit[1, :70], label="fit ssp585")

ax.set_ylabel("Temperature difference")
ax.set_xlabel("Year")
ax.legend()

fig.savefig("global_temperature_difference_fit.pdf")

years = np.arange(1990, 1990 + 40 + nyears)

# Plot some regions:

cells = [0, 182, 1565, 2146, 2331, 1987, 16071] + fails

for icell in cells:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

    ax.plot(years[40 : 40 + nyears_geoe_370], diff_370[icell, :], label="temp diff 370")
    ax.plot(years[40:], fit[0, icell, :], label="fit 370")
    ax.plot(years[40 : 40 + nyears_geoe_370], diff_585[icell, :], label="temp diff 585")
    ax.plot(years[40:], fit[1, icell, :], label="fit 585")
    ax.plot(years[40:], fit[2, icell, :], label="fit both")

    ax.set_ylabel("Temperature difference")
    ax.set_xlabel("Year")
    ax.legend()

    fig.savefig(
        "temperature_difference_region_"
        + str(list_latitudes[icell])
        + "_"
        + str(list_longitudes[icell])
        + "_"
        + country_names[icell]
        + ".pdf"
    )
    plt.close()


# ---------------------------------------------------------------------------
