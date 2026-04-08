---
Contributors:
  - Jenny Bjordal
  - Henri Cornec
  - Evelien van Dijk
  - Anthony A. Smith, Jr.
  - Trude Storelvmo
---
<!--
[![DOI](https://zenodo.org/badge/948414289.svg)](https://doi.org/10.5281/zenodo.17176879)
-->

# README and Guidance

## Overview

An idealised Solar Radiation Management (SRM) experiment using the coupled climate-economy model NorESM2-DIAM (Bjordal el al., 2026).

This repository contains the experimental setup, modifications, and analysis code for replicating the experiments in the paper "Regional Economic Impacts and Emission Responses under Solar Radiation Modification" (in prep). The coupled NorESM2-DIAM model and implementation can be found in the [NorESM2-DIAM GitHub](https://github.com/jennybj/coupling_noresm2_diam), except the Earth system component, NorESM2, which can be found in the [NorESM GitHub](https://github.com/NorESMhub/NorESM). The current repo focuses on:

- Modifications made to the original model for the experiments in the paper
- Code for setting up and running the experiment
- Data analysis and visualization: including scripts to analyze and plot the experimental results

Refer to the NorESM2-DIAM repository for the model's usage instructions. This repository assumes you have already set up the base model and provides additional tools to run and evaluate the experiments specific to the paper.

## Data Availability and Provenance Statements

<!--
### Statement about Rights
- [x] I certify that the author(s) of the manuscript have legitimate access to and permission to use the data used in this manuscript.
- [x] I certify that the author(s) of the manuscript have documented permission to redistribute/publish the data contained within this replication package. Appropriate permission are documented in the [LICENSE.txt](LICENSE.txt) file.


### License for Data

The data are licensed under a Creative Commons/CC-BY-NC license. See LICENSE.txt for details.
-->
### Summary of Availability

- [x] All data **are** publicly available.
- [ ] Some data **cannot be made** publicly available.
- [ ] **No data can be made** publicly available.


## Dataset list

| Data  | File name| Path | Location | Notes                                                                 |
|-----------------------------------------------------------------|-------------------------------|---------------------------------------|--------------------------|------------------------------------------------|
| Cumulative CO2 emissions for SSP126, SSP245, SSP370, and SSP585 | `SSP_cumulative_emissions.txt`| `data/input_coefficients/` | Current repo | Created from script in [NorESM2-DIAM GitHub](https://github.com/jennybj/coupling_noresm2_diam)  | |
| Historical + SSP370 with only CO2 emissions | `onlyCO2.nc`   | `data/input_coefficients/` | [NorESM2-DIAM GitHub](https://github.com/jennybj/coupling_noresm2_diam) | Full output from NIRD RDA (Bjordal, 2025c)|
| SSP370 with SRM from 2030 and only CO2 emissions | `reduced_solar_const_1percent.nc`   | `data/input_coefficients/` | Current repo | Full output from NIRD RDA (Bjordal, 2025d) |
| SSP585 with only CO2 emissions | `ssp585_onlyCO2.nc`   | `data/input_coefficients/` | Current repo | Full output from NIRD RDA (Bjordal, 2025e) |
| SSP585 with SRM from 2030 and only CO2 emissions | `ssp585_reduced_solar_const_1percent.nc`   | `data/input_coefficients/` | Current repo | Full output from NIRD RDA (Bjordal, 2025f) |
| Solar forcing standard input file | `SolarForcingCMIP6piControl_c160921.nc` | `data/input_standard/` | Current repo | From the standard NorESM2 input data |
| Solar forcing reduced 1% input file | `SolarForcing_1percent_reduced.nc` | `data/input_noresm2diam/` | Current repo | To be used from 2030 in the SRM experiments |


## Computational requirements


### Software Requirements

<!--
The replication package contains two programs to install the necessary dependencies for Julia and Python. R scripts are installed within the scripts.

**Julia 1.10.4**

Run `setup/install_packages.jl` to install all necessary Julia packages.
Note that the GenericMappingTools (GMT) require a separate installation via homebrew.

Simply run: 
```bash
julia install_packages.jl
```
**Python 3.7.3**

To run the Python scripts in `scripts/create_figures/`, `scripts/create_input_files/`, and `scripts/calculate_coefficients/` use the setup from `setup/environment.yml`.
The easiest way is to create a new conda environment from the `environment.yml` file.
This can be done in the terminal as follows:

```bash
conda env create -f environment.yml
conda activate base_env
```

The first command need only be run once, while the second activates the conda environment `base_env` (as specified by the file) and must be activated before running the scripts.

**Running NorESM2-DIAM on HPC system**

The scripts in `scripts/running_noresm2diam/` are used to run the coupled model and must be done on a HPC system. To set up and run the mode, follow the instructions from the [NorESM2-DIAM GitHub](https://github.com/jennybj/coupling_noresm2_diam) and replace the corresponding files with the ones in the current repository.


<!--

### Controlled Randomness

- [x] Random seed is set at line 192 of program 'scripts/standalone\_noresm2diam.jl'.
- [x]  No Pseudo random generator is used elsewhere in the analysis described here.

### Memory, Runtime, Storage Requirements

Portions of this code were last run on a 6-core Apple M2-Pro laptop with MacOS version 15.5 with 50GB of free space.

Portions of the code were last run on a 3-node cluster (1x cascadelake, 2x icelake) with a SLURM cluster manager.

NorESM2 (including the coupling scripts) was run on an Atos BullSequana XH2000, using 10 CPU nodes (each with 128 cores and 256 GiB of memory). The machine, named Betzy, is provided by Sigma2 AS, and more details can be found [here](https://documentation.sigma2.no/hpc_machines/betzy.html). With this setup, the coupled model takes approximately one hour per year of simulation.

The rest of the python scripts (not for coupling) can be run on any laptop. Each script can be run in less than 5 minutes, and in total they produce output requiring storage of approximately 90 MB.
-->

## Description of code

### General

- The program `scripts/module_coupling.py` reads in various data and performs calculations used by the various Python scripts in `scripts/create_figures/` and `scripts/calculate_coefficients` as well as by the coupling script `scripts/run_noresm2diam/couple_with_decision_rules.py`.

### Generate input files

- The program `scripts/create_input_files/adjust_solar_forcing_file.py` takes the original NorESM2 solar forcing file and reduces the solar forcing by 1%. This file must be used from 2030 in the SRM experiments, and is specified in the `user_nl_cam` file of the NorESM2 case.
- The program `scripts/calculate_coefficients/write_temperatures_to_files.py` reads in the NetCDF files with monthly temperature data from NorESM2 in `data/input_coefficients/`, calculates the annual means, regrid to the 1x1° grid of DIAM and writes them to .txt files for easier use.
- The program `scripts/calculate_coefficients/calculate_coefficients.py` reads in the temperatures in the .txt files created above, use them to calculate the coefficients of the SRM temperature offset, and write them to the file `SRM_coefficients.txt`.

### Running Standalone DIAM
- The program `scripts/geoengi_v1.jl` will calculate decision rules used in the coupled run as well as generate the output files for a so-called fixed-point run where all shocks ( z_{it} ) are set to 0. (Corresponding to `scripts/decrule_calc.jl` in [NorESM2-DIAM GitHub](https://github.com/jennybj/coupling_noresm2_diam).)

### Running NorESM2-DIAM

- The program `scripts/run_noresm2diam/set_up_noresm_srm_case.py` creates a new NorESM2 case (our simulation) starting in 2030 with SRM to be used in the coupling. For the baseline simulation without SRM, see the [NorESM2-DIAM GitHub](https://github.com/jennybj/coupling_noresm2_diam).
- The program `scripts/run_noresm2diam/couple_iterations.sh` loads modules and activates the correct conda environment before initializing the coupling script `scripts/run_noresm2diam/couple_with_decision_rules.py`.
- The program `scripts/run_noresm2diam/couple_with_decision_rules.py` couples the two models. It reads in the last year temperature data from NorESM2, uses the decision rules as calculated by `scripts/decrule_calc.jl`, calculates the emissions of next year, and writes the emissions to an input file for NorESM2 to read.
- The program `scripts/run_noresm2diam/calculate_fp_srm.py` is not needed for the coupling. It simply calculates the same data as the DIAM standalone (the fixed point) and writes this output to files of the same format as the coupling script, to make future calculations/comparisons easier. This one is for SRM, for the baseline simulation, see the [NorESM2-DIAM GitHub](https://github.com/jennybj/coupling_noresm2_diam).

### Calculations and figures
- The program `scripts/create_figures/plot_output.py` reads in the data produced by the coupled model, performs calculations—at grid cell, country, and global level—and produces figures.

- The program `scripts/create_figures/make_figures.jl`reads in the data produced by the coupled model, performs calculations—at grid cell, country, and global level—and produces figures.
- 
### License for Code

The code is licensed under a MIT license. See [LICENSE](LICENSE) for details.

## Instructions to Replicators

To set up, run the DIAM standalone model, and finally run NorESM2-DIAM, see the instructions in [NorESM2-DIAM GitHub](https://github.com/jennybj/coupling_noresm2_diam) and replace the scripts described above.

### Calculations and figures

- Run the programs in `scripts/create_figures/` to create the figures in the paper. 

```bash
python plot_output.py
julia make_figures.jl
```

## Model output

We performed two experiments---baseline and SRM implementation---with three ensemble members each. All the output from the coupled model is available from the [Norwegian Research Infrastructure Services (NIRD) Research Data Archive (RDA)](https://data.archive.sigma2.no/):

| Simulation        | Location          | Reference      | Note                      | 
|-------------------|-------------------|----------------|----------------------------------|
| Baseline ensemble member 1 | NIRD RDA | Bjordal, J. (2025a) and Bjordal, J. (2025b) | This is the same simulation as the one presented in Bjordal et al., 2026. NorESM2 standard output and the specific coupled output are separated. |
| Baseline ensemble member 2 | NIRD RDA | Bjordal, J. (2026a) | |
| Baseline ensemble member 3 | NIRD RDA | Bjordal, J. (2026b) | |
| SRM ensemble member 1 | NIRD RDA | Bjordal, J. (2026c) | |
| SRM ensemble member 2 | NIRD RDA | Bjordal, J. (2026d) | |
| SRM ensemble member 3 | NIRD RDA | Bjordal, J. (2026e) | |



## List of figures

The provided code reproduces:

| Figure    | Program                  | Line Numbers | Output file                      | 
|-------------------|--------------------------|-------------|----------------------------------|
| Fig. 1 | `scripts/create_figures/plot_output.py`| 691-845 | `figures/emissions_temperature_gdpper_compare.pdf` | 
| Fig. 2 |`scripts/create_figures/make_figures.jl`|195-217| `figures/no_srm_tempdiff_linear.pdf`, `figures/srm_tempdiff_linear.pdf`, `figures/tempdiff_2100_linear.pdf` | 
| Fig. 3 |`scripts/create_figures/make_figures.jl`|352-375|`figures/nosrm_gdp_pct_change_sym70.pdf`, `figures/srm_gdp_pct_change_sym70.pdf`, `figures/gdp_pct_change_diff_sym70.pdf`|
| Fig. 4 |`scripts/create_figures/plot_output.py`  | 1014-1309 |`figures/country_difference_gdpper_percent_SRM_2090s-2020s.pdf`| 
| Fig. 5 | `scripts/create_figures/plot_output.py`| 1314-1416 | `figures/histogram_GDPper_difference_GDP_share.pdf` | 
| Fig. 6 |`scripts/create_figures/make_figures.jl`|218-350|`figures/pop_90_10_ratio_srm_nosrm.png`|



## References

Bjordal, J., Smith Jr., A. A., Cornec, H., and Storelvmo, T.: ***NorESM2–DIAM: a coupled model for investigating global and regional climate-economy interactions***, Geosci. Model Dev., 19, 1337–1365, [DOI](https://doi.org/10.5194/gmd-19-1337-2026), 2026

Bjordal, J. (2025a). ***NorESM2-DIAM prototype simulation, coupled output*** [Data set]. NIRD RDA. [DOI](https://doi.org/10.11582/2025.90v981qk)

Bjordal, J. (2025b). ***NorESM2-DIAM prototype simulation, NorESM2 standard output*** [Data set]. NIRD RDA. [DOI](https://doi.org/10.11582/2025.31ney5y8)

Bjordal, J. (2025c). ***NorESM2-LME Historical and SSP3-7.0 with only CO2 emissions*** [Data set]. NIRD RDA. [DOI](https://doi.org/10.11582/2025.tdi6hhfl)

Bjordal, J. (2025d). ***NorESM2-LME SSP3-7.0 from 2030 with only CO2 emissions and solar forcing 1% reduced*** [Data set]. NIRD RDA. [DOI](https://doi.org/10.11582/2025.jtui66g0)

Bjordal, J. (2025e). ***NorESM2-LME SSP5-8.5 with only CO2 emissions*** [Data set]. NIRD RDA. [DOI](https://doi.org/10.11582/2025.4uncny33)

Bjordal, J. (2025f). ***NorESM2-LME SSP5-8.5 from 2030 with only CO2 emissions and solar forcing 1% reduced*** [Data set]. NIRD RDA. [DOI](https://doi.org/10.11582/2025.p4mv0r06)

Bjordal, J. (2026a). ***NorESM2-DIAM baseline simulation - ensemble member 2*** [Data set]. NIRD RDA. [DOI](https://doi.org/10.11582/2026.8p3mr4b3)

Bjordal, J. (2026b). ***NorESM2-DIAM baseline simulation - ensemble member 3*** [Data set]. NIRD RDA. [DOI](https://doi.org/10.11582/2026.kien66g0)

Bjordal, J. (2026c). ***NorESM2-DIAM solar forcing 1% reduced simulation - ensemble member 1*** [Data set]. NIRD RDA. [DOI](https://doi.org/10.11582/2026.ngzr041o)

Bjordal, J. (2026d). ***NorESM2-DIAM solar forcing 1% reduced simulation - ensemble member 2*** [Data set]. NIRD RDA. [DOI](https://doi.org/10.11582/2026.5louigpd)

Bjordal, J. (2026e). ***NorESM2-DIAM solar forcing 1% reduced simulation - ensemble member 3*** [Data set]. NIRD RDA. [DOI](https://doi.org/10.11582/2026.q2umwad7)

---

## Acknowledgements

The structure of this README is adapted from the README template by Villhuber, Koren, Llull, Connolly and Morrow. Available [by clicking here](https://github.com/social-science-data-editors/template_README/blob/release-candidate/templates/README.md).
