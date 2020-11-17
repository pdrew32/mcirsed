import numpy as np
import pandas as pd
import mcirsed_ff
from astropy.cosmology import Planck15 as cosmo
from astropy import constants as c
from astropy import units as u
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde


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
    arbitraryNorm1 = 6
    fixAlphaValue = 2.0
    fixBetaValue = 2.0
    fixW0Value = 200.0


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
    """same as lpeak_mmpz but with an offset"""
    return eta * (logLIR - 12.0) + lam0


def log_lirtd_corr_ODR(beta, logLIR):
    """same as log_lirtd_corr but organized for use with scipy odr"""
    return beta[0] * (logLIR - 12.0) + beta[1]


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


def create_tdust_lpeak_grid(tdusts, beta, w0, path):
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
    lpeak = np.vectorize(mcirsed_ff.lambdaPeak)
    lpeaks = lpeak(2., tdusts, 2., beta, w0)
    t_l = pd.DataFrame(data=np.array([lpeaks, tdusts]).transpose(), columns=['lpeak', 'Tdust'])
    t_l.to_csv(path)
    return


'''def scaling_factor(wave, fluxLimit, z_list, genF, fixAlphaValue, fixBetaValue, fixW0Value, plot_it=True, verbose=True):
    """calculate the scaling factor to apply to arbitrary snu to get the correct final snu

    Parameters:
    -----------
    wave : float
        wavelength of selection
    
    fluxLimit : float
        flux limit at wavelength of selection
    
    z_list : list or pandas column
        list of real redshifts from sample

    genF : dataFrame
        frame of generated galaxies

    fixAlphaValue : float
        alpha to fix

    fixBetaValue : float
        beta to fix

    fixW0Value : float
        w0 to fix

    plot_it : bool
        should we plot 10 seds to check that the scaling is working appropriately?
    
    Returns:
    --------
    scaled_norm1 : ndarray
        scaling factor that sets the appropriate flux limit
    """

    log_s_unscaled = np.zeros([len(z_list), len(genF.gen_loglpeak)])
    arbitraryNorm1 = 6
    for i in list(range(len(genF.gen_loglpeak))):
        if verbose is True:
            print(str(i) + '/' + str(len(genF.gen_loglpeak)))
        log_s_unscaled[:, i] = np.log10(mcirsed_ff.SnuNoBump(arbitraryNorm1, genF.loc[i, 'Tdust'], fixAlphaValue, fixBetaValue, fixW0Value, wave/(1+z_list)))
    # scaling factor is the value to add to arbitraryNorm1 to achieve the proper scaling
    scaled_norm1 = np.log10(fluxLimit) - log_s_unscaled

    if plot_it is True:
        galnum = 2
        gengalnum = list(range(10))
        x = np.logspace(np.log10(8), np.log10(1000), 1000)
        for i in gengalnum:
            y = np.log10(mcirsed_ff.SnuNoBump(arbitraryNorm1 + scaled_norm1[galnum, gengalnum[i]], genF.loc[gengalnum[i], 'Tdust'], fixAlphaValue, fixBetaValue, fixW0Value, x))

            plt.scatter(x, 10**y)
            plt.yscale('log')
            plt.xlabel('restframe wavelength (um)')
            plt.ylabel('S (mJy)')
            plt.axhline(fluxLimit, color='k')
            plt.axvline(wave/(1 + z_list[galnum]), color='k')
            plt.xscale('log')
            plt.show()

    return scaled_norm1'''


def detec_frac(wave, fitF, genF, scalingFactor, plot_it=True):
    """calculate the fraction of galaxies in the given LIR bin that
       would be detectable given the noise of the sample

    wave : float
        wavelength of selection
    fitF : pandas DataFrame
        dataframe of fits
    genF : pandas DataFrame
        dataframe of generated galaxies
    scalingFactor : ndarray
        scaling factor from scaling_factor
    plot_it : bool
        whether or not to show plots of the first 10 limits
    """

    # set ind j to 0. will plot from j=0 to 9 if plot_it is True
    j = 0
    for i in list(fitF.index):
        # for every galaxy:
        # 1) get lir bin the gal belongs to
        my_lir_bin_ind = genF.lir_bin_ind == fitF.loc[i, 'lir_bin_ind']
        # 2) get lirs at detec limit, scaled to z of gal
        lim_lirs = genF.loc[my_lir_bin_ind, 'log_lir_limit_unscaled'] + scalingFactor[j, my_lir_bin_ind]
        # 3) if there are galaxies in the LIR bin:
        if len(lim_lirs) > 0:
            # detec frac is sum of limit lirs less than those in uniform irlf over # gals in limit lirs
            fitF.loc[i, 'detec_frac_' + str(wave)] = sum(lim_lirs.values < np.log10(genF.loc[my_lir_bin_ind, 'gen_lir'].values))/len(lim_lirs)
        if (j < 10) & (plot_it is True):
            plt.scatter(lim_lirs.values, genF.loc[my_lir_bin_ind, 'gen_loglpeak'])
        j += 1
    if plot_it is True:
        plt.scatter(np.log10(genF.gen_lir), genF.gen_loglpeak, zorder=-1, color='k')
        plt.show()
    
    return fitF


'''def scaling_factor_dlim_curve(selec_wave, fluxLimit, z_list, td_lpF, fixAlphaValue, fixBetaValue, fixW0Value):
    """calculate the scaling factor to apply to arbitrary snu to get the correct final snu. Similar to scaling_factor() except will use a much wider range of tdusts for plotting purposes

    Parameters:
    -----------
    selec_wave : float
        wavelength of selection
    
    fluxLimit : float
        flux limit at wavelength of selection
    
    td_lpF : pandas dataframe
        pandas dataframe containing lpeaks and corresponding tdusts to plot
    
    z_list : list or pandas column
        list of real redshifts from sample

    fixAlphaValue : float
        alpha to fix

    fixBetaValue : float
        beta to fix

    fixW0Value : float
        w0 to fix
    
    Returns:
    --------
    scalingFactor_lir : ndarray
        scaling factors for lirs that set the appropriate flux limit
    """
    
    log_s_unscaled = np.zeros([len(z_list), len(td_lpF.lpeak)])
    arbitraryNorm1 = 6
    for i in list(range(len(td_lpF.lpeak))):
        print(str(i) + '/' + str(len(td_lpF.lpeak)))
        log_s_unscaled[:, i] = np.log10(mcirsed_ff.SnuNoBump(arbitraryNorm1, td_lpF.loc[i, 'Tdust'], fixAlphaValue, fixBetaValue, fixW0Value, selec_wave/(1+z_list)))
    # scaling factor is the difference between unscaled s60 and the flux limit
    scalingFactor_snu = np.log10(fluxLimit) - log_s_unscaled

    log4pidlsq = np.log10((4*np.pi*cosmo.luminosity_distance(z_list)**2.).value * h.conversionFactor / (1+z_list))
    log4pidlsq = np.tile(log4pidlsq, (np.shape(scalingFactor_snu)[1], 1))
    log4pidlsq = np.transpose(log4pidlsq)
    # adjust scaling factor to account for log4pidlsq
    scalingFactor_lir = scalingFactor_snu + log4pidlsq
    return scalingFactor_lir'''


def scaled_norm1(detec_wave, detec_lim, z_array, tdust_array, fixAlphaValue, fixBetaValue, fixW0Value, verbose=True):
    """calculate values to add to norm1 that will scale SEDs to the provided detection limit

    Parameters:
    -----------
    detec_wave : float
        wavelength of selection
    
    detec_lim : float
        flux limit at wavelength of selection
    
    z_array : list, array, or pandas column
        list of real redshifts from sample
    
    tdust_array : list, array, or pandas column
        array of temperatures corresponding to lpeaks to calculate limits at.

    fixAlphaValue : float
        alpha to fix

    fixBetaValue : float
        beta to fix

    fixW0Value : float
        w0 to fix
    
    Returns:
    --------
    scaled_norm1 : ndarray
        value to add to norm1 to scale SEDs to the provided detection limit
    """

    log_s_unscaled = np.zeros([len(z_array), len(tdust_array)])
    for i in list(range(len(tdust_array))):
        if verbose is True:
            print(str(i) + '/' + str(len(tdust_array)))
        log_s_unscaled[:, i] = np.log10(mcirsed_ff.SnuNoBump(h.arbitraryNorm1, tdust_array[i], fixAlphaValue, fixBetaValue, fixW0Value, detec_wave/(1+z_array)))
    # scaling factor is the difference between unscaled s60 and the flux limit
    scaled_norm1 = np.log10(detec_lim) - log_s_unscaled
    return scaled_norm1


def estimate_maxima(data):
    """script to find max of kde of hist of errors"""
    kde = gaussian_kde(data)
    # samples = np.linspace(np.floor(min(data)), np.floor(max(data)) + 1, int(np.floor(max(data)) + 1 - np.floor(min(data))))
    samples = np.linspace(np.floor(min(data)), np.floor(max(data)) + 1, 1000)
    probs = kde.evaluate(samples)
    maxima_index = probs.argmax()
    maxima = samples[maxima_index]
    return maxima
