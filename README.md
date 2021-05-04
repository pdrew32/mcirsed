# Monte Carlo InfraRed Spectral Energy Distribution (MCIRSED)

A Python tool for fitting the 8--1000 micron dust emission of galaxies published in Drew et al. 2021 in preparation.

## Required Packages:
As of the time of writing, the main Python package we use for MCMC fitting, pymc3, will not work with Python 3.8 without a hack to the code. Hopefully this will change in the future. We recommend you follow the instructions [here](https://github.com/pdrew32/mcirsed/blob/master/install-help.md) to create a new conda environment with Python 3.7.

- Python 3.7
- latest anaconda stable build
- pymc3
- libpython
- corner
- m2w64-toolchain (if using Windows)

## Getting Started:
For an example of how to use the code, run example_data.py to generate data for two galaxies. Next run example_fit_mcirsed.py. There are instructions in the comments of this script that will guide you through working with. We recommend you save a copy of example_fit_mcirsed.py before editing the inputs. 

## Inputs:
Required inputs to the code are wavelengths of observations in microns, flux densities and uncertainties in mJy, and a redshift.

Parameters that may be free or held fixed: 
- Alpha (power law slope)
- Beta (dust emissivity)
- Lambda_0 (wavelength where dust opacity = 1. Referred to in the code as w0)

## Outputs:
The code will output a pandas dataframe save as a .pkl file containing, in this order:

- redshift
- value alpha was fixed to or None if free parameter
- value beta was fixed to or None if free parameter
- value lambda_0 (W0) was fixed to or None if free parameter
- how many tuning steps the sampling was run with
- number of MC samples
- whether CMB was corrected for
- norm1 parameter for each MC sample
- dust temperature for each MC sample
- alpha for each MC sample
- beta for each MC sample
- lambda_0 (w0) for each MC sample
- log LIR for each MC sample
- peak wavelength for each MC sample
- median norm1 across all samples
- median tdust
- median alpha
- median beta
- median lambda_0 (w0)
- median log LIR
- median peak wavelength
- 16th percentile for norm1 samples
- 16th tdust
- 16th alpha
- 16th beta
- 16th lambda_0 (w0)
- 16th log LIR
- 16th peak wavelength
- 84th percentile for norm1 samples
- 84th tdust
- 84th alpha
- 84th beta
- 84th lambda_0 (w0)
- 84th log LIR
- 84th peak wavelength
- wavelengths of data used for each galaxy in microns
- fluxes in mJy used for each galaxy in microns
- errors in mJy used for each galaxy in microns
