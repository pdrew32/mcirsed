# Monte Carlo InfraRed Spectral Energy Distribution (MCIRSED)

A Python tool for fitting the 8--1000 micron dust emission of galaxies published in Drew et al. 2021 in preparation.

## Required Packages:

As of the time of writing, the main Python package we use for MCMC fitting, pymc3, will not work with Python 3.8 without an annoying hack. We recommend you create a new conda environment with Python 3.7, following the instructions [here](https://github.com/pdrew32/mcirsed/edit/master/install-help.md).

For an example of how to use the code, run example_data.py to generate data for two galaxies and run the script example_fit_mcirsed.py.

Parameters:  
-----------  
dataWave : array of floats or Quantities  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;array of observed wavelengths, assumed to be in microns if not an astropy unit quantity.  

dataFlux : array of floats or Quantities  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;array of observed flux densities, assumed to be in mJy if not an astropy unit quantity.  

errFlux : array of floats or Quantities  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;array of flux density errors, assumed to be in mJy if not an astropy unit quantity.  

redshift : float  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;redshift of the galaxy.  

fixAlpha : float or None  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;float of value to fix Alpha to or None if you want alpha to vary.  

fixBeta : float or None  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;float of value to fix Beta to or None if you want beta to vary.  

fixW0 : float, Quantity, or None  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;float or quantity of value to fix the wavelength where the opacity is equal to 1 or None if you want w0 to vary. Assumed to be in microns if not an astropy unit quantity.  

Returns:  
--------  
trace : pymc3 trace object  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pymc3 trace with all of the samples  
