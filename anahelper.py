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
    trace
    fixAlphaValue
    fixBetaValue
    fixW0Value
    
    Returns:
    -----------
    medianNorm1
    medianTdust
    medianAlpha
    medianBeta
    medianW0
    """
    norm1Array = trace['norm1']
    TdustArray = trace['Tdust']
    LIRArray = trace['LIR']
    lPeakArray = trace['lPeak']
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
    """
    Return labels and data in a format for easy use with corner.py

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
    tdusts = np.logspace(np.log10(3), np.log10(300), 100000)
    lpeak = np.vectorize(mcirsed_ff.lambdaPeak)
    lpeaks = lpeak(2., tdusts, 2., beta, w0)
    t_l = pd.DataFrame(data=np.array([lpeaks, tdusts]).transpose(), columns=['lpeak', 'Tdust'])
    t_l.to_csv(path)

    return



def checkTemperature(beta, lPeak, w0, TdustOutput, verbose=True):
    '''
    Make sure the output dust temperature and input beta returns the lPeak
    input.
    '''

    b = mcirsed_ff.lambdaPeak(2., TdustOutput, 2., beta, w0, h.xWa, h.fineRestWave, h.hck)
    
    if abs(b-lPeak) > 0.01:
        errorBool = 1
        if verbose is True:
            print('--> error, difference between input and output lPeaks is > 0.01' +
                'microns')
            print('diff:' + str(b-lPeak))
    else:
        errorBool = 0
        if verbose is True:
            print('--> passed LPeak-Temp conversion check')
    
    return errorBool


def checkLIR(norm1, Tdust, alpha, beta, w0, z, inputLIR, verbose=True):
    '''
    Make sure the output dust temperature and input beta returns the lPeak
    input.
    '''
    fourPiLumDistSquared = ((4*np.pi*cosmo.luminosity_distance(z)**2.).value * h.conversionFactor)
    lircheck = mcirsed_ff.IRLum(norm1, Tdust, alpha, beta, w0, h.xWa, h.fineRestWave, h.hck, h.deltaHz, fourPiLumDistSquared)
    if abs(lircheck / (1+z)-inputLIR)/inputLIR > 0.01:
        if verbose is True:
            print('error, LIR percent error is > 1%')
            print('diff:' + str(abs(lircheck / (1+z)-inputLIR)/inputLIR))
        errorBool = 1
    else:
        if verbose is True:
            print('passed input/output LIR conversion check')
            print('its: ' + str(abs(lircheck / (1+z)-inputLIR)/inputLIR))
            print(str(inputLIR) + ' / ' + str(lircheck / (1+z)))

        errorBool = 0
    return errorBool


def normalization(LIR, z, Tdust, alpha, beta, w0):
    """
    return the normalization for SnuNoBump, norm1, given the input parameters.
    """
    finerRestWave = np.linspace(8, 1000, 66800)
    fourPiLumDistSquared = ((4*np.pi*cosmo.luminosity_distance(z)**2.).value * h.conversionFactor)
    xHzfiner = np.linspace((c.c/(8.*u.micron)).decompose().value, (c.c/(1000.*u.micron)).decompose().value, 50000)[::-1]
    xWafiner = (c.c/xHzfiner/u.Hz).decompose().to(u.um)[::-1].value
    deltaHz = xHzfiner[1]-xHzfiner[0]
    hck = (c.h*c.c/c.k_B).to(u.micron*u.K).value  # units: micron*Kelvins
    return LIR * (1+z) / (fourPiLumDistSquared * deltaHz * np.sum(mcirsed_ff.SnuNoBumpNoNorm(Tdust, alpha, beta, w0, xWafiner, finerRestWave, hck)))


def normalizationForFluxLimit(LIR, z, Tdust, alpha, beta, w0):
    """
    return the normalization for SnuNoBump, norm1, given the input parameters.
    """
    finerRestWave = np.linspace(8, 1000, 66800)
    fourPiLumDistSquared = ((4*np.pi*cosmo.luminosity_distance(z)**2.).value * h.conversionFactor)
    xHzfiner = np.linspace((c.c/(8.*u.micron)).decompose().value, (c.c/(1000.*u.micron)).decompose().value, 50000)[::-1]
    xWafiner = (c.c / xHzfiner / u.Hz).decompose().to(u.um)[::-1].value
    deltaHzfiner = xHzfiner[1]-xHzfiner[0]
    return LIR * (1+z) / (fourPiLumDistSquared * deltaHzfiner * np.sum(mcirsed_ff.SnuNoBumpNoNorm(Tdust, alpha, beta, w0, xWafiner, finerRestWave, h.hck)))


def fluxLimit60umLIR_LPeak(fluxLimit, z, fixAlphaValue, fixBetaValue, fixW0Value, oldDataPath, newSavePath, plotLimit=False, saveAsDataFrame=True):
    # fluxLimit, z, fixAlphaValue, fixBetaValue, fixW0Value, nPoints
    """
    Find the parameters associated with the 60 um flux limit for the LIR_LP plot.
    The difference with the function fluxLimit60um is that this allows lPeak to vary.
    """

    frame = pd.read_csv(oldDataPath, index_col=0)
    possibleLPeaks = frame.lPeaks
    TdustList = frame.Tdusts
    nCurves = len(frame.index)
    fourPiLumDistSquared = ((4*np.pi*cosmo.luminosity_distance(z)**2.).value * h.conversionFactor)

    arbitraryNorm1 = 1e5
    logS60List = np.zeros(nCurves)
    for i in list(range(nCurves)):
        logS60List[i] = np.log10(mcirsed_ff.SnuNoBump(arbitraryNorm1, TdustList[i], fixAlphaValue, fixBetaValue, fixW0Value, 60.0/(1+z), h.fineRestWave, h.hck))
    scalingFactor = logS60List - np.log10(fluxLimit)

    LIRList = np.zeros(nCurves)
    for i in list(range(nCurves)):
        LIRList[i] = np.sum(10**(np.log10(mcirsed_ff.SnuNoBump(arbitraryNorm1, TdustList[i], fixAlphaValue, fixBetaValue, fixW0Value, h.xWa, h.fineRestWave, h.hck)) - scalingFactor[i])) * h.deltaHz * fourPiLumDistSquared

    if plotLimit is True:
        plt.plot(LIRList, possibleLPeaks)
        plt.xscale('log')
        plt.yscale('log')
        plt.show()

    if saveAsDataFrame is True:
        frame['LIRs'] = LIRList
        frame['z'] = np.ones(nCurves) * z
        frame.to_csv(newSavePath)

    return LIRList, possibleLPeaks


def fluxLimit250umLIR_LPeak(fluxLimit, z, fixAlphaValue, fixBetaValue, fixW0Value, oldDataPath, newSavePath, plotLimit=False, saveAsDataFrame=True):
    """
    Find the parameters associated with the 250 um flux limit for the LIR_LP plot.
    The difference with the function fluxLimit60um is that this allows lPeak to vary.
    """
    frame = pd.read_csv(oldDataPath, index_col=0)
    possibleLPeaks = frame.lPeaks
    TdustList = frame.Tdusts
    nCurves = len(frame.index)
    fourPiLumDistSquared = ((4*np.pi*cosmo.luminosity_distance(z)**2.).value * h.conversionFactor)

    arbitraryNorm1 = 1e5
    logS250List = np.zeros(nCurves)
    for i in list(range(nCurves)):
        logS250List[i] = np.log10(mcirsed_ff.SnuNoBump(arbitraryNorm1, TdustList[i], fixAlphaValue, fixBetaValue, fixW0Value, 250.0/(1+z), h.fineRestWave, h.hck))
    scalingFactor = logS250List - np.log10(fluxLimit)

    LIRList = np.zeros(nCurves)
    for i in list(range(nCurves)):
        LIRList[i] = np.sum(10**(np.log10(mcirsed_ff.SnuNoBump(arbitraryNorm1, TdustList[i], fixAlphaValue, fixBetaValue, fixW0Value, h.xWa, h.fineRestWave, h.hck)) - scalingFactor[i])) * h.deltaHz * fourPiLumDistSquared

    if plotLimit is True:
        plt.plot(LIRList, possibleLPeaks)
        plt.xscale('log')
        plt.yscale('log')
        plt.show()

    if saveAsDataFrame is True:
        frame['LIRs'] = LIRList
        frame['z'] = np.ones(nCurves) * z
        frame.to_csv(newSavePath)

    return LIRList, possibleLPeaks


def saveLPeakTdustConversion(nCurves, limLPeakLow, limLPeakHigh, fixBetaValue, fixW0Value, savePath):
    """
    save LPeak and Tdust conversion for fixed alpha and beta to prevent having to calculate them fresh every time
    """
    LPeaks = np.logspace(np.log10(limLPeakLow), np.log10(limLPeakHigh), nCurves)
    TdustList = np.zeros(len(LPeaks))
    fixBetaValues = np.ones(len(LPeaks)) * fixBetaValue
    fixW0Values = np.ones(len(LPeaks)) * fixW0Value
    for i in list(range(len(LPeaks))):
        print('calculating grid dust temps '+ str(i) + '/' + str(nCurves-1))
        TdustList[i] = temperatureGivenBetaLPeak(fixBetaValue, LPeaks[i], fixW0Value)

    frame = pd.DataFrame(data=np.array([LPeaks, TdustList, fixBetaValues, fixW0Values]).transpose(), columns=['lPeaks', 'Tdusts', 'beta', 'w0'])
    frame.to_csv(savePath)
    
    return


def saveLPeakTdustConversionGeneratedGalaxies(nCurves, fixBetaValue, fixW0Value, savePath):
    """
    save LPeak and Tdust conversion for fixed alpha and beta to prevent having to calculate them fresh every time
    """
    LPeaks = np.logspace(np.log10(limLPeakLow), np.log10(limLPeakHigh), nCurves)
    TdustList = np.zeros(len(LPeaks))
    fixBetaValues = np.ones(len(LPeaks)) * fixBetaValue
    fixW0Values = np.ones(len(LPeaks)) * fixW0Value
    for i in list(range(len(LPeaks))):
        print('calculating grid dust temps '+ str(i) + '/' + str(nCurves-1))
        TdustList[i] = temperatureGivenBetaLPeak(fixBetaValue, LPeaks[i], fixW0Value)

    frame = pd.DataFrame(data=np.array([LPeaks, TdustList, fixBetaValues, fixW0Values]).transpose(), columns=['lPeaks', 'Tdusts', 'beta', 'w0'])
    frame.to_csv(savePath)
    
    return




def normGiven60umFlux(flux, z, LIR, lPeak, fixAlphaValue, fixBetaValue, fixW0Value):
    """
    Find the parameters associated with the 60 um flux limit
    """

    Tdust = temperatureGivenBetaLPeak(fixBetaValue, lPeak, fixW0Value)
    norm1 = normalization(LIR, z, Tdust, fixAlphaValue, fixBetaValue, fixW0Value)
    Snu60Calculated = mcirsed_ff.SnuNoBump(norm1, Tdust, fixAlphaValue, fixBetaValue, fixW0Value, 60.0, h.fineRestWave, h.hck)

    # print('delta Snu:')
    # print(abs(flux-Snu60Calculated))
    # print('calculated Snu:')
    # print(Snu60Calculated)

    return norm1, Tdust


def LIRCurve(norm1, Tdust, fixAlphaValue, fixBetaValue, fixW0Value, inputZList):
    """
    return the LIR detection limit as a function of z
    """
    fourPiLumDistSquaredList = np.zeros(len(inputZList))
    lumArrays = np.zeros(len(inputZList))
    for i in list(range(len(fourPiLumDistSquaredList))):
        fourPiLumDistSquaredList[i] = ((4*np.pi*cosmo.luminosity_distance(inputZList[i])**2.).value * h.conversionFactor)
        lumArrays[i] = mcirsed_ff.IRLum(norm1, Tdust, fixAlphaValue, fixBetaValue, fixW0Value, h.xWa, h.fineRestWave, h.hck, h.deltaHz, fourPiLumDistSquaredList[i])

    return lumArrays
