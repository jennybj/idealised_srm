# -------------------------------------------------------------------------------------

# Modules needed:
import os as os
import shutil

from netCDF4 import Dataset

# -------------------------------------------------------------------------------------

# SPECIFY FILES AND CONSTANTS

# File names:
original_file = "../../data/input_standard/SolarForcingCMIP6piControl_c160921.nc"
outfile = "../../data/input_noresm2diam/SolarForcing_1percent_reduced.nc"

# -------------------------------------------------------------------------

# MAKE OUTPUT FILE

# Delete if file exists:
if os.path.exists(outfile):
    os.remove(outfile)

shutil.copyfile(original_file, outfile)

# Create new files open for writing:
file = Dataset(outfile, "r+")

tsi = file["tsi"][:]
file["tsi"][:] = 0.99 * tsi

ssi = file["ssi"][:]
file["ssi"][:] = 0.99 * ssi

file.close()

# -------------------------------------------------------------------------------------
