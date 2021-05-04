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

- ```z```: redshift
- ```fixAlphaValue```: value alpha was fixed to or None if free parameter
- ```fixBetaValue```: value beta was fixed to or None if free parameter
- ```fixW0Value```: value lambda_0 (W0) was fixed to or None if free parameter
- ```tune```: how many tuning steps the sampling was run with
- ```MCSamples```: number of MC samples
- ```CMBCorrection```: whether CMB was corrected for
- ```trace_Norm1```: norm1 parameter for each MC sample
- ```trace_Tdust```: dust temperature for each MC sample
- ```trace_alpha```: alpha for each MC sample
- ```trace_beta```: beta for each MC sample
- ```trace_w0```: lambda_0 (w0) for each MC sample
- ```trace_LIR```: log LIR for each MC sample
- ```trace_lPeak```: peak wavelength for each MC sample
- ```median_Norm1```: median norm1 across all samples
- ```median_Tdust```: median tdust
- ```median_alpha```: median alpha
- ```median_beta```: median beta
- ```median_w0```: median lambda_0 (w0)
- ```median_LIR```: median log LIR
- ```median_lPeak```: median peak wavelength
- ```Norm1_16th```: 16th percentile for norm1 samples
- ```Tdust_16th```: 16th tdust
- ```alpha_16th```: 16th alpha
- ```beta_16th```: 16th beta
- ```w0_16th```: 16th lambda_0 (w0)
- ```LIR_16th```: 16th log LIR
- ```lPeak_16th```: 16th peak wavelength
- ```Norm1_84th```: 84th percentile for norm1 samples
- ```Tdust_84th```: 84th tdust
- ```alpha_84th```: 84th alpha
- ```beta_84th```: 84th beta
- ```w0_84th```: 84th lambda_0 (w0)
- ```LIR_84th```: 84th log LIR
- ```lPeak_84th```: 84th peak wavelength
- ```dataWave```: wavelengths of data used for each galaxy in microns
- ```dataFlux```: fluxes in mJy used for each galaxy in microns
- ```dataErr```: errors in mJy used for each galaxy in microns
