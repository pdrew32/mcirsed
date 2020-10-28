import numpy as np
import pandas as pd
import mcirsed_ff
from astropy.cosmology import Planck15 as cosmo
from astropy import constants as c
from astropy import units as u
from matplotlib import pyplot as plt


'''
Analysis helper functions for mcirsed
'''


class h:
    """define constants and arrays we will use repeatedly"""
    hck = (c.h*c.c/c.k_B).to(u.micron*u.K).value  # units: micron*Kelvins
    fineRestWave = np.linspace(8, 1000, 5000)
    xHz = np.linspace((c.c/(8.*u.micron)).decompose().value,
                      (c.c/(1000.*u.micron)).decompose().value, 100000)[::-1]
    xWa = (c.c/xHz/u.Hz).decompose().to(u.um)[::-1].value
    deltaHz = xHz[1]-xHz[0]
    conversionFactor = 2.4873056783618645e-11  # mJy Hz Mpc^2 to lsol


def lpeak_mmpz(LIR, lam, eta):
    """mmpz function to return peak wavelength given lir, eta, lam.
    See Casey2020 for more detail (2020ApJ...900...68C)

    Parameters
    ----------
    LIR : float or array
        IR luminosity you're interested in
    
    lam : float
        wavelength of the relation at LIR = 1e12

    eta : float
        slope of the relation in log-log space

    Returns
    -------
    lpeak : float or array
        the peak wavelength of the SED along the lir-lpeak
        correlation at the given LIR, lam, and eta
    """
    return lam * (LIR/1e12)**eta


def log_lirtd_corr(logLIR, eta=-0.068, lam0=100):
    """same as lpeak_mmpz but with an offset """
    return eta * (logLIR - 12.0) + lam0


def returnFitParamArrays(trace, fixAlphaValue, fixBetaValue, fixW0Value):
    """Return arrays filled with fixed and or fitted parameter values.

    Parameters
    ----------
    trace : object
       The trace object containing the samples collected.

    fixAlphaValue : Float or None
       The float or null value of Alpha inputted into the pymc3 model.

    fixBetaValue : Float or None
        The float or null value of Beta inputted into the pymc3 model.

    fixW0Value : Float or None
        The float or null value of w0 inputted into the pymc3 model.

    Returns
    -------
    norm1Array : ndarray
        numpy array filled with norm1 values for each successful chain.

    TdustArray : ndarray
        numpy array filled with Tdust values for each successful chain.

    alphaArray : ndarray
        numpy array filled with alpha fit values for each successful chain or
        the fixed alpha value if fixAlphaValue is a float.

    betaArray : ndarray
        numpy array filled with beta fit values for each successful chain or
        the fixed beta value if fixBetaValue is a float.

    w0Array : ndarray
        numpy array filled with w0 fit values for each successful chain or
        the fixed w0 value if fixW0Value is a float.

    LIRArray : ndarray
        numpy array filled with LIR values for each successful chain.

    lPeakArray : ndarray
        numpy array filled with LIR values for each successful chain.
    """

    norm1Array = trace['norm1']
    TdustArray = trace['Tdust']
    LIRArray = trace['LIR']
    lPeakArray = trace['lPeak']
    if fixAlphaValue is None:
        alphaArray = trace['alpha']
    else:
        alphaArray = np.ones(len(norm1Array))*fixAlphaValue
    if fixBetaValue is None:
        betaArray = trace['Beta']
    else:
        betaArray = np.ones(len(norm1Array))*fixBetaValue
    if fixW0Value is None:
        w0Array = trace['w0']
    else:
        w0Array = np.ones(len(norm1Array))*fixW0Value

    return (norm1Array, TdustArray, alphaArray, betaArray, w0Array, LIRArray,
            lPeakArray)


def returnMedianParams(trace,fixAlphaValue,fixBetaValue,fixW0Value):
    """Return median fit parameters
    
    Parameters:
    -----------
    trace : pymc3 trace
        output of pymc3 sampling
    fixAlphaValue : None or float
        value alpha is fixed to or None if allowed to be sampled
    fixBetaValue : None or float
        value beta is fixed to or None if allowed to be sampled
    fixW0Value : None or float
        value w0 is fixed to or None if allowed to be sampled
    
    Returns:
    -----------
    medianNorm1 : float
        median of norm1 samples or value param was fixed to
    medianTdust : float
        median of tdust samples or value param was fixed to
    medianAlpha : float
        median of alpha samples or value param was fixed to
    medianBeta : float
        median of beta samples or value param was fixed to
    medianW0 : float
        median of w0 samples or value param was fixed to
    """
    norm1Array = trace['norm1']
    TdustArray = trace['Tdust']
    if fixAlphaValue == None:
        alphaArray = trace['alpha']
    else: alphaArray = np.ones(len(norm1Array))*fixAlphaValue
    if fixBetaValue == None: 
        betaArray = trace['Beta']
    else: betaArray = np.ones(len(norm1Array))*fixBetaValue
    if fixW0Value == None:
        w0Array = trace['w0']
    else: w0Array = np.ones(len(norm1Array))*fixW0Value
        
    return np.median(norm1Array), np.median(TdustArray), np.median(alphaArray), np.median(betaArray), np.median(w0Array)


def returnMedianParamsAndErrorsFromFitFrame(fitFrame):
    """Adds best values and 1sigma errors to the fit dataframe."""

    fitFrame['measuredLIR'] = fitFrame['trace_LIR'].map(medFunc)
    fitFrame['measuredLIRlosig'] = fitFrame['trace_LIR'].map(lowSigma)
    fitFrame['measuredLIRhisig'] = fitFrame['trace_LIR'].map(higSigma)

    fitFrame['measuredLPeak'] = fitFrame['trace_lPeak'].map(medFunc)
    fitFrame['measuredLPeaklosig'] = fitFrame['trace_lPeak'].map(lowSigma)
    fitFrame['measuredLPeakhisig'] = fitFrame['trace_lPeak'].map(higSigma)

    fitFrame['measuredAlpha'] = fitFrame['trace_alpha'].map(medFunc)
    fitFrame['measuredAlphalosig'] = fitFrame['trace_alpha'].map(lowSigma)
    fitFrame['measuredAlphahisig'] = fitFrame['trace_alpha'].map(higSigma)

    fitFrame['measuredBeta'] = fitFrame['trace_beta'].map(medFunc)
    fitFrame['measuredBetalosig'] = fitFrame['trace_beta'].map(lowSigma)
    fitFrame['measuredBetahisig'] = fitFrame['trace_beta'].map(higSigma)

    fitFrame['measuredw0'] = fitFrame['trace_w0'].map(medFunc)
    fitFrame['measuredw0losig'] = fitFrame['trace_w0'].map(lowSigma)
    fitFrame['measuredw0hisig'] = fitFrame['trace_w0'].map(higSigma)

    fitFrame['measuredTdust'] = fitFrame['trace_Tdust'].map(medFunc)
    fitFrame['measuredTdustlosig'] = fitFrame['trace_Tdust'].map(lowSigma)
    fitFrame['measuredTdusthisig'] = fitFrame['trace_Tdust'].map(higSigma)

    fitFrame['measuredNorm1'] = fitFrame['trace_Norm1'].map(medFunc)
    fitFrame['measuredNorm1losig'] = fitFrame['trace_Norm1'].map(lowSigma)
    fitFrame['measuredNorm1hisig'] = fitFrame['trace_Norm1'].map(higSigma)
        
    return fitFrame


def medFunc(x):
    """return median. for vectorization of operations in pandas"""
    return np.median(x)
    

def lowSigma(x):
    """return 16th percentile for vectorization of operations in pandas"""
    return np.median(x) - np.percentile(x, 16)


def higSigma(x):
    """return 84th percentile for vectorization of operations in pandas"""
    return np.percentile(x, 84) - np.median(x)


def fixedValueReturns1(x):
    """return 1 for fixed fitting params for vectorization in pandas"""
    n = np.empty_like(x)
    nones = x == n
    return nones


def cornerHelper(trace, fixAlphaValue, fixBetaValue, fixW0Value):
    """Return labels and data in a format for easy use with corner.py

    Returns
    -------
    arrList : list
        numpy array filled with norm1 values for each successful chain.

    labels : list
        numpy array filled with float labels of each successful parameter
        allowed to vary.
    """
    labels = ['norm1', 'log(LIR)', 'lPeak', 'Tdust']
    arrList = [trace['norm1'], trace['LIR'], trace['lPeak'],
               trace['Tdust']]

    if fixAlphaValue is None:
        arrList.append(trace['alpha'])
        labels.append('alpha')

    if fixBetaValue is None:
        arrList.append(trace['Beta'])
        labels.append('beta')

    if fixW0Value is None:
        arrList.append(trace['w0'])
        labels.append('lambda_0')

    return np.array(arrList).transpose(), labels


def create_tdust_lpeak_grid(beta, w0, path):
    """save a conversion grid between dust temps and peak waves to csv at path
       to avoid having to calculate it every time we want to convert
    
    Parameters:
    -----------
    beta : float
        spectral emissivity index to generate SEDs

    w0 : float
        wavelength where opacity of modified blackbody is 1. used to generate
        seds

    path : string
        where to save the csv output
    """
    tdusts = np.logspace(np.log10(3), np.log10(300), 100000)
    lpeak = np.vectorize(mcirsed_ff.lambdaPeak)
    lpeaks = lpeak(2., tdusts, 2., beta, w0)
    t_l = pd.DataFrame(data=np.array([lpeaks, tdusts]).transpose(), columns=['lpeak', 'Tdust'])
    t_l.to_csv(path)
    return
