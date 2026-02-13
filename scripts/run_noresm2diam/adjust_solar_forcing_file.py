#-------------------------------------------------------------------------------------

# Modules needed:
from netCDF4 import Dataset
import numpy as np
import os as os
import shutil

#-------------------------------------------------------------------------------------

# SPECIFY FILES AND CONSTANTS

# File names:
original_file = 'SolarForcingCMIP6piControl_c160921.nc'
outfile = 'SolarForcing_1percent_reduced.nc'

#-------------------------------------------------------------------------

# MAKE OUTPUT FILE

# Delete if file exists:
if os.path.exists(outfile):
    os.remove(outfile)

shutil.copyfile(original_file, outfile)

# Create new files open for writing:
file = Dataset(outfile, 'r+')

tsi = file['tsi'][:]
file['tsi'][:] = 0.99 * tsi

ssi = file['ssi'][:]
file['ssi'][:] = 0.99 * ssi

file.close()

#-------------------------------------------------------------------------------------
