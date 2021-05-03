# Monte Carlo InfraRed Spectral Energy Distribution (MCIRSED)

A Python tool for fitting the 8--1000 micron dust emission of galaxies published in Drew et al. 2021 in preparation.

## Required Packages:

As of the time of writing, the main Python package we use for MCMC fitting, pymc3, will not work with Python 3.8 without an annoying hack. We recommend you create a new conda environment with Python 3.7, following the instructions [here](https://github.com/pdrew32/mcirsed/edit/master/install-help.md).

For an example of how to use the code, run example_data.py to generate data for two galaxies and run the script example_fit_mcirsed.py.
