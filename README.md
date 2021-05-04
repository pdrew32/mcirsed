# Monte Carlo InfraRed Spectral Energy Distribution (MCIRSED)

A Python tool for fitting the 8--1000 micron dust emission of galaxies published in Drew et al. 2021 in preparation.

## Required Packages:
As of the time of writing, the main Python package we use for MCMC fitting, pymc3, will not work with Python 3.8 without a hack to the code. We recommend you create a new conda environment with Python 3.7, following the instructions [here](https://github.com/pdrew32/mcirsed/blob/master/install-help.md).

- Python 3.7
- latest anaconda stable build
- pymc3
- libpython
- corner
- m2w64-toolchain (if using Windows)

## Getting Started:
For an example of how to use the code, run example_data.py to generate data for two galaxies. Next run example_fit_mcirsed.py to see an example. There are instructions in the comments of this script that will guide you through using the script. We recommend you save a copy of example_fit_mcirsed.py before editing the inputs. 
