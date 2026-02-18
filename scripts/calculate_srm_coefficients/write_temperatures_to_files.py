import sys as sys

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

sys.path.insert(0, "../")
from module_coupling import (
    calculate_annual_mean,
    get_coordinate_data,
    regrid_from_noresm_to_diam,
    sort_in_diam_order,
)

# ---------------------------------------------------------------------------

# READ IN DATA

path = "../../data/input_to_regression/"

file_base_370 = path + "onlyCO2.nc"
file_geoe_370 = path + "reduced_solar_const_1percent.nc"
file_base_585 = path + "ssp585_onlyCO2.nc"
file_geoe_585 = path + "ssp585_reduced_solar_const_1percent.nc"

ncfile = Dataset(file_base_370)
temp_base_370 = ncfile.variables["TREFHT"][140 * 12 :, :, :] - 273.15
lats = ncfile.variables["lat"][:]
ncfile.close()

ncfile = Dataset(file_geoe_370)
temp_geoe_370 = ncfile.variables["TREFHT"][:] - 273.15
ncfile.close()

ncfile = Dataset(file_base_585)
temp_base_585 = ncfile.variables["TREFHT"][:] - 273.15
ncfile.close()

ncfile = Dataset(file_geoe_585)
temp_geoe_585 = ncfile.variables["TREFHT"][:] - 273.15
ncfile.close()


list_latitudes, list_longitudes = get_coordinate_data()
ncells = list_latitudes.shape[0]

# ---------------------------------------------------------------------------

# CONVERT TO DIAM GRID

temp_base_370 = calculate_annual_mean(temp_base_370)
temp_geoe_370 = calculate_annual_mean(temp_geoe_370)
temp_base_585 = calculate_annual_mean(temp_base_585)
temp_geoe_585 = calculate_annual_mean(temp_geoe_585)

global_temp_base_370 = np.average(
    np.average(temp_base_370, axis=2), axis=1, weights=np.cos(np.deg2rad(lats))
)
global_temp_geoe_370 = np.average(
    np.average(temp_geoe_370, axis=2), axis=1, weights=np.cos(np.deg2rad(lats))
)
global_temp_base_585 = np.average(
    np.average(temp_base_585, axis=2), axis=1, weights=np.cos(np.deg2rad(lats))
)
global_temp_geoe_585 = np.average(
    np.average(temp_geoe_585, axis=2), axis=1, weights=np.cos(np.deg2rad(lats))
)

print(global_temp_geoe_585)

nyears_base_370 = temp_base_370.shape[0]
nyears_geoe_370 = temp_geoe_370.shape[0]
nyears_base_585 = temp_base_585.shape[0]
nyears_geoe_585 = temp_geoe_585.shape[0]
print(nyears_base_370, nyears_geoe_370, nyears_base_585, nyears_geoe_585)

temperature_base_370 = np.zeros((nyears_base_370, ncells))
temperature_geoe_370 = np.zeros((nyears_geoe_370, ncells))
temperature_base_585 = np.zeros((nyears_base_585, ncells))
temperature_geoe_585 = np.zeros((nyears_geoe_585, ncells))

for iyear in range(nyears_base_370):
    temp_dgrid = regrid_from_noresm_to_diam(temp_base_370[iyear, :, :])
    temperature_base_370[iyear, :] = sort_in_diam_order(temp_dgrid)

    if iyear < nyears_geoe_370:
        temp_dgrid = regrid_from_noresm_to_diam(temp_geoe_370[iyear, :, :])
        temperature_geoe_370[iyear, :] = sort_in_diam_order(temp_dgrid)

    if iyear < nyears_base_585:
        temp_dgrid = regrid_from_noresm_to_diam(temp_base_585[iyear, :, :])
        temperature_base_585[iyear, :] = sort_in_diam_order(temp_dgrid)

    if iyear < nyears_geoe_585:
        temp_dgrid = regrid_from_noresm_to_diam(temp_geoe_585[iyear, :, :])
        temperature_geoe_585[iyear, :] = sort_in_diam_order(temp_dgrid)


print(np.max(temperature_base_370), np.min(temperature_base_370))
print(np.max(temperature_geoe_370), np.min(temperature_geoe_370))
print(np.max(temperature_base_585), np.min(temperature_base_585))
print(np.max(temperature_geoe_585), np.min(temperature_geoe_585))

print(
    temperature_base_370.shape,
    temperature_geoe_370.shape,
    temperature_base_585.shape,
    temperature_geoe_585.shape,
)


# ---------------------------------------------------------------------------

# PLOT GLOBAL VALUES

years = np.arange(1990, 1990 + nyears_base_370)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))

ax.plot(years, global_temp_base_370, label="ssp370")
ax.plot(years[40:], global_temp_geoe_370, label="ssp370 geo")
ax.plot(years[25 : 25 + nyears_base_585], global_temp_base_585, label="ssp585")
ax.plot(years[40 : 40 + nyears_geoe_585], global_temp_geoe_585, label="ssp585 geo")

ax.set_ylabel("Temperature")
ax.set_xlabel("Year")
ax.legend()

fig.savefig("global_temperature.pdf")

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))

ax.plot(years[40:], global_temp_geoe_370 - global_temp_base_370[40:], label="ssp370")
ax.plot(
    years[40 : 40 + nyears_geoe_585],
    global_temp_geoe_585 - global_temp_base_585[15:],
    label="ssp585",
)

ax.set_ylabel("Temperature")
ax.set_xlabel("Year")
ax.legend()

fig.savefig("global_temperature_difference.pdf")

# ---------------------------------------------------------------------------

# WRITE TO FILES

file1 = open("temperatures_onlyCO2_start_1990.txt", "w")
file2 = open("temperatures_1percRed_start_2030.txt", "w")
file3 = open("temperatures_ssp585_onlyCO2_start_2015.txt", "w")
file4 = open("temperatures_ssp585_1percRed_start_2030.txt", "w")


for icell in range(ncells):
    file1.writelines(["%16.1f" % list_latitudes[icell]])
    file1.writelines(["%16.1f" % list_longitudes[icell]])
    file1.writelines(["%16.7f" % item for item in temperature_base_370[:, icell]])
    file1.write("\n")

    file2.writelines(["%16.1f" % list_latitudes[icell]])
    file2.writelines(["%16.1f" % list_longitudes[icell]])
    file2.writelines(["%16.7f" % item for item in temperature_geoe_370[:, icell]])
    file2.write("\n")

    file3.writelines(["%16.1f" % list_latitudes[icell]])
    file3.writelines(["%16.1f" % list_longitudes[icell]])
    file3.writelines(["%16.7f" % item for item in temperature_base_585[:, icell]])
    file3.write("\n")

    file4.writelines(["%16.1f" % list_latitudes[icell]])
    file4.writelines(["%16.1f" % list_longitudes[icell]])
    file4.writelines(["%16.7f" % item for item in temperature_geoe_585[:, icell]])
    file4.write("\n")

file1.close()
file2.close()
file3.close()
file4.close()
