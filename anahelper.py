import numpy as np
from numpy.random import Generator, SFC64
import pandas as pd
import mcirsed_ff
from astropy.cosmology import Planck15 as cosmo
from astropy import constants as c
from astropy import units as u
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from scipy import interpolate


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


def lpeak_mmpz(LIR, lam, eta, L_t=12):
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
    return lam * (LIR/10**L_t)**eta


def log_lirtd_corr(logLIR, eta=-0.068, lam0=100, L_t=12.0):
    """same as lpeak_mmpz but with an offset"""
    return eta * (logLIR - L_t) + lam0


def log_lirtd_corr_ODR(beta, logLIR, L_t=12.0):
    """same as log_lirtd_corr but organized for use with scipy odr
    NOTE: beta[0] = eta, beta[1] = log(lam_t)
    """
    return beta[0] * (logLIR - L_t) + beta[1]


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
    fitFrame['measuredLIRlosig'] = fitFrame['trace_LIR'].map(lowSigmaLIR)
    fitFrame['measuredLIRhisig'] = fitFrame['trace_LIR'].map(higSigmaLIR)

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


def lowSigmaLIR(x):
    """return 16th percentile for vectorization of operations in pandas"""
    a = 10**x
    b = 0.434 * ((np.percentile(a, 50) - np.percentile(a, 16))/np.median(a))
    return b
    # return np.median(a) - np.log10(np.percentile(a, 16))


def higSigmaLIR(x):
    """return 84th percentile for vectorization of operations in pandas"""
    a = 10**x
    b = 0.434 * ((np.percentile(a, 84) - np.percentile(a, 50))/np.median(a))
    return b
    # return np.log10(np.percentile(10**x, 84)) - np.median(x)


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


def IRLF(L, z=0.0):
    """IRLF from Casey+18a

    Parameters:
    -----------
    L : ndarray
        luminosities to calculate number counts for
    
    z : float
        redshift to calculate the irlf at. defaults to 0.
    
    Returns:
    --------
    irlf_arr : ndarray
        number per Mpc^3 per dex
    """
    irlf_arr = np.zeros_like(L)
    lstar = 10**log_lstar(z)
    phi_star = 10**log_phi_star(z)
    alpha_lf = -0.6
    beta_lf = -3.0
    irlf_arr[L < lstar] = phi_star * (L[L < lstar]/lstar) ** alpha_lf
    irlf_arr[L >= lstar] = phi_star * (L[L >= lstar]/lstar) ** beta_lf
    return irlf_arr


def log_lstar(z, gamma_1=2.8, gamma_2=1.0, z_turn=1.95, z_w=2.0, l_not = 1.3e11):
    """equation 8 from Casey+18a describing the z evolution of lstar

    For z_turn, taking the average of the two models, A and B. 1.95 is average of 2.1 and 1.8
    """
    x = np.log10(1+z)
    x_t = np.log10(1 + z_turn)
    x_w = z_w/(np.log(10) * (1+z_turn))

    a = (gamma_2 - gamma_1)/2.0 * x_w/np.pi
    b = np.log(np.cosh((x - x_t) * np.pi/x_w))
    c = np.log(np.cosh(-x_t * np.pi/x_w))
    d = (gamma_2 + gamma_1)/2 * x
    e = np.log10(l_not)

    return a * (b - c) + d + e

def log_phi_star(z, phi_1=0.0, phi_2=-4.2, z_turn=1.95, z_w=2.0, phi_0=3.2e-4):
    """equation 9 from Casey+18a describing the z evolution of phi_star
    
    For phi_2, taking the average of the two models, A and B. -4.2 is average of -5.9 and -2.5
    """
    x = np.log10(1+z)
    x_t = np.log10(1 + z_turn)
    x_w = z_w/(np.log(10) * (1+z_turn))

    a = (phi_2 - phi_1)/2.0 * x_w/np.pi
    b = np.log(np.cosh((x - x_t) * np.pi/x_w))
    c = np.log(np.cosh(-x_t * np.pi/x_w))
    d = (phi_2 + phi_1)/2.0 * x
    e = np.log10(phi_0)

    return a * (b - c) + d + e


def generate_galaxies_from_irlf(z_min, z_max, survey_area, lirs):
    """returns log_irlf from casey+18 in z bin over survey area at provided lirs. also randomly chooses numbers of galaxies in each lir bin based on the probability of observing them
    
    Parameters:
    -----------
    z_min : float
        minimum redshift to create z array from
    z_max : float
        maximum redshift to create z array from
    survey_area : float
        survey area in deg^2
    lirs : array
        array of lirs to calculate over. units of solar luminosities, not log10(lsol). Use np.logspace so they are evenly spaced in log10

    Returns:
    --------
    log_sky_phiarr : numpy array
        the expected number of galaxies between z_min and z_max across the survey area for the given lirs. fractions allowed.
    ngals_per_lir : numpy array
        the rounded of galaxies between z_min and z_max across the survey area for the given lirs. integers only.
    """
    # use a high quality RNG
    rg = Generator(SFC64())

    # calculate volume of cosmos field over the desired redshifts
    # units of Mpc^3
    vol_z_upper = cosmo.comoving_volume(z_max).value * survey_area
    vol_z_lower = cosmo.comoving_volume(z_min).value * survey_area

    # ir luminosity function from casey+18a
    phiarr = IRLF(lirs, z=np.mean([z_max, z_min]))

    # get delta log lir
    delta_log_LIR = np.log10(lirs[1]) - np.log10(lirs[0])

    # convert irlf to units on the sky: sources/deg^2 in given zbin and lbin
    log_sky_phiarr = np.log10(phiarr) + np.log10(vol_z_upper - vol_z_lower) + np.log10(delta_log_LIR)
    ngals_per_lir = 10**log_sky_phiarr

    # need to work with ngals_per_lir as probabilities
    # round to get floor version
    ngals_per_lir_floor = np.floor(ngals_per_lir)
    # subtract to find remainder
    ngals_per_lir_remainder = np.abs(ngals_per_lir - ngals_per_lir_floor)

    # generate a random uniform number. if lower than the remainder of ngals, round up, otherwise round down
    rand = rg.uniform(0, 1, len(ngals_per_lir_remainder))
    ngals_per_lir = np.where(ngals_per_lir_remainder > rand, np.ceil(ngals_per_lir), np.floor(ngals_per_lir))

    return log_sky_phiarr, ngals_per_lir


def simulate_galaxies(lirs, ngals_per_lir, dF):
    """returns lir and lpeak of simulated galaxies
    
    Parameters:
    -----------
    lirs : numpy array or list
        lirs to populate the galaxies over
    ngals_per_lir : numpy array or list
        the number of galaxies in the given lir bin

    Returns:
    --------
    simlir : numpy array
        the log lirs of simulated galaxies
    simlpeak : numpy array
        the log lpeaks of simulated galaxies
    """
    rg = Generator(SFC64()) # use a high-quality rng

    # interpolate over the lir-lpeak correlation from the provided lirs and lir-lpeak correlation fit params
    lirtd_interp = interpolate.interp1d(np.log10(lirs), np.log10(lpeak_mmpz(lirs, dF.lam_t.values[0], dF.eta.values[0])))

    simlir = []
    simlpeak = []
    # for every lir bin, for the number of galaxies in that lir bin randomly draw from the measured distribution of galaxies around the lir-lpeak correlation provided in dF
    for i in list(range(len(lirs))):
        for j in list(range(int(ngals_per_lir[i]))):
            simlpeak.append(rg.normal(lirtd_interp(np.log10(lirs[i])), dF.width)[0])
            simlir.append(np.log10(lirs[i]))
    simlir = np.array(simlir)
    simlpeak = np.array(simlpeak)
    return simlir, simlpeak
