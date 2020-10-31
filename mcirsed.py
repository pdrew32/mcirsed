import platform
import numpy as np
import pymc3 as pm
import anahelper as ah
import theano as T
import theano.tensor as tt
from astropy.cosmology import Planck15 as cosmo
from astropy import units as u
from matplotlib import pyplot as plt


"""
Main functions for the Monte Carlo InfraRed Spectral Energy Distribution fitter (MCIRSED)
published in Drew et al. 2021 in preparation.
"""


def BB(nbb, Tdust, beta, w0, restWave):
    """Modified blackbody function

    Math takes the form:
    10**nbb * (1.0-np.exp(-(w0/restWave)**beta)) * restWave**(-3.0) / (np.exp(ah.h.hck/restWave/Tdust)-1.0)
    """
    a = tt.pow(10, nbb)
    b = tt.sub(1.0, tt.exp(-tt.pow(tt.true_div(w0, restWave), beta)))
    c = tt.pow(restWave, -3.0)
    d = tt.sub(tt.exp(tt.true_div(tt.true_div(ah.h.hck, restWave), Tdust)), 1.0)
    return tt.true_div(tt.mul(tt.mul(a, b), c), d)


def powerLaw(npl, restWave, alpha):
    """Equation of the power law portion of SED"""
    return tt.mul(npl, tt.pow(restWave, alpha))


def derivativeLogBB(Tdust, beta, w0):
    """Solves for the (approx) derivatives of the BB function."""
    extra_fine_rest_wave = np.logspace(np.log10(20), np.log10(200), 1000)
    log_bb = np.log10(BB(10.0, Tdust, beta, w0, extra_fine_rest_wave))
    delta_y = log_bb[1:] - log_bb[:-1]
    delta_x = np.log10(extra_fine_rest_wave[1:]) - np.log10(extra_fine_rest_wave[:-1])
    return delta_y / delta_x


def eqWave(alpha, Tdust, beta, w0):
    """Compute the wavelength where the derivative of the log of BB equals the slope of the power law"""
    der_bb_reverse = derivativeLogBB(Tdust, beta, w0)[::-1]
    # only search 20um to 200um because the eqWave is definitely between there
    extra_fine_rest_wave = T.shared(np.logspace(np.log10(20), np.log10(200), 1000)[::-1])
    return extra_fine_rest_wave[tt.extra_ops.searchsorted(der_bb_reverse, alpha)]


def SnuNoBump(norm1, Tdust, alpha, beta, w0, restWave):
    """Combined MBB and Power Law functional form to fit with MCMC
    For speed of computation, uses a sigmoid function with a sharp cutoff
    times BB and 1-sigmoid times pl to achieve the piecewise function.
    """
    eq_w = eqWave(alpha, Tdust, beta, w0)
    bb = BB(norm1, Tdust, beta, w0, restWave)
    n = BB(norm1, Tdust, beta, w0, eq_w) * eq_w**-alpha
    pl = powerLaw(n, restWave, alpha)
    sig = tt.nnet.sigmoid(200*(restWave-eq_w))
    return (1-sig) * pl + sig * bb


def IRLum(norm1, Tdust, alpha, beta, w0, z, fourPiLumDistSquared):
    """Calculate LIR"""
    return tt.log10(tt.sum(SnuNoBump(norm1, Tdust, alpha, beta, w0, ah.h.xWa)) * ah.h.deltaHz/(1+z) * fourPiLumDistSquared)


def lambdaPeak(norm1, Tdust, alpha, beta, w0):
    """Calculate Peak Wavelength"""
    x = tt.cast(ah.h.xWa,'float64')
    return x[tt.argmax(SnuNoBump(norm1, Tdust, alpha, beta, w0, x))]


def Tredshift0(redshift, beta, Tdust):
    """equation for calculating the dust temperature if the galaxy were at z=0"""
    power = 4+beta
    return (Tdust**power - cosmo.Tcmb0.value**power * ((1+redshift)**power - 1)) ** (1/power)


def mcirsed(dataWave, dataFlux, errFlux, redshift, fixAlpha=None, fixBeta=None, fixW0=None, CMBCorrection=False, MCSamples=5000, tune=2000, discardTunedSamples=True, loNorm1=1, upNorm1=5e10, upTdust=150., flat_alpha_prior=True):
    """Function to fit an infrared (8-1000um) spectral energy distribution to a galaxy's infrared data points
    
    Parameters:
    -----------
    dataWave : array of floats or Quantities
        array of observed wavelengths, assumed to be in microns if not an astropy unit quantity.
    dataFlux : array of floats or Quantities
        array of observed flux densities, assumed to be in mJy if not an astropy unit quantity.
    errFlux : array of floats or Quantities
        array of flux density errors, assumed to be in mJy if not an astropy unit quantity.
    redshift : float
        redshift of the galaxy.
    fixAlpha : float or None
        float of value to fix Alpha to or None if you want alpha to vary.
    fixBeta : float or None
        float of value to fix Beta to or None if you want beta to vary.
    fixW0 : float, Quantity, or None
        float or quantity of value to fix the wavelength where the opacity is equal to 1 or None if you want w0 to vary. Assumed to be in microns if not an astropy unit quantity.

    Returns:
    --------
    trace : pymc3 trace object
        pymc3 trace with all of the samples
    """
    # do some unit handling
    if hasattr(dataWave,'unit'):
        dataWave = dataWave.to('micron').value
    else: dataWave = (dataWave*u.micron).value
    if hasattr(dataFlux,'unit'):
        dataFlux = dataFlux.to('mJy').value
    else: dataFlux = (dataFlux*u.mJy).value
    if hasattr(errFlux,'unit'):
        errFlux = errFlux.to('mJy').value
    else: errFlux = (errFlux*u.mJy).value
    if fixW0!=None:
        if hasattr(fixW0,'unit'):
            fixW0 = fixW0.to('micron').value
        else: fixW0 = (fixW0*u.micron).value    

    # require the number of samples to be an integer
    MCSamples = int(MCSamples)

    # format alpha, beta, and wavelength opacity is unity values if necessary
    print('\n*****************************************************')
    if fixBeta is None: 
        print('Running with Beta as a free parameter')
    else:
        beta = float(fixBeta)
        print('Fixing Beta to '+str(fixBeta))
    if fixAlpha is None: 
        print('Running with Alpha as a free parameter')
    else:
        alpha = float(fixAlpha)
        print('Fixing Alpha to '+str(fixAlpha))
    if fixW0 is None: 
        print('Running with wavelength where opacity = 1 as a free parameter')
    else:
        fixW0 = float(fixW0)
        print('Fixing rest wavelength where opacity = 1 to '+str(fixW0)+' microns')
    print('*****************************************************\n')
    
    # a few definitions:
    restWave = dataWave / (1+redshift)
    fourPiLumDistSquared = (4 * np.pi * cosmo.luminosity_distance(redshift)**2.).value * ah.h.conversionFactor
    
    mod = pm.Model()
    with mod:
        # Priors for unknown model parameters
        norm1 = pm.Bound(pm.Flat, lower=loNorm1, upper=upNorm1)('norm1') # normalization of SED
        Tdust = pm.Bound(pm.Flat, lower=cosmo.Tcmb(redshift).value, upper=upTdust)('Tdust') # dust temperature of the system representing the emission from the greybody bounded between the CMB temperature at the galaxy's redshift and 250 K.
        if fixBeta is None:
            fixBeta = pm.Bound(pm.Flat, lower=0.5, upper=5.0)('Beta') # emissivity of the greybody
        if fixAlpha is None:
            if flat_alpha_prior is True:
                fixAlpha  = pm.Bound(pm.Flat, lower=0.0, upper=6.)('alpha') # the slope of the powerlaw component
            if flat_alpha_prior is False:
                fixAlpha = pm.Normal('alpha', mu=2.3, sigma=0.5)
        if fixW0 is None:
            fixW0 = pm.Bound(pm.Flat, lower=5., upper=2000.)('w0') # rest-wave bounds in microns

        # Expected value of outcome.
        mu = SnuNoBump(norm1, Tdust, fixAlpha, fixBeta, fixW0, restWave)
        
        if CMBCorrection is True:
            Tdust0 = pm.Deterministic('Tdust0', Tredshift0(redshift, fixBeta, Tdust))
            LIR = pm.Deterministic('LIR', IRLum(norm1, Tdust0, fixAlpha, fixBeta, fixW0, redshift, fourPiLumDistSquared))
            lPeak = pm.Deterministic('lPeak', lambdaPeak(norm1, Tdust0, fixAlpha, fixBeta, fixW0))
            
        else: 
            LIR = pm.Deterministic('LIR', IRLum(norm1, Tdust, fixAlpha, fixBeta, fixW0, redshift, fourPiLumDistSquared))
            lPeak = pm.Deterministic('lPeak', lambdaPeak(norm1, Tdust, fixAlpha, fixBeta, fixW0))
        
        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Normal('Y_obs', mu=mu, sd=errFlux, observed=dataFlux)

    if platform.system() == 'Windows':
        with mod:
            step = pm.NUTS(target_accept=0.9)
            trace = pm.sample(draws=MCSamples, step=step, tune=tune, discard_tuned_samples=discardTunedSamples, cores=1)
    else:
         with mod:
            step = pm.NUTS(target_accept=0.9)
            trace = pm.sample(draws=MCSamples, step=step, tune=tune, discard_tuned_samples=discardTunedSamples)
    print('')
    print('Summary of trace: ')
    print(pm.summary(trace).round(2))

    return trace
