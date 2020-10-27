import mcirsed_ff
import numpy as np
import pandas as pd
import findmyindex
from astropy.cosmology import Planck15 as cosmo
from astropy import constants as c
from astropy import units as u
from matplotlib import pyplot as plt


'''
Analysis helper functions for mcirsed
'''


class h:
    '''
    define constants and arrays we will use repeatedly
    '''
    hck = (c.h*c.c/c.k_B).to(u.micron*u.K).value  # units: micron*Kelvins
    fineRestWave = np.linspace(8, 1000, 5000)
    xHz = np.linspace((c.c/(8.*u.micron)).decompose().value,
                      (c.c/(1000.*u.micron)).decompose().value, 100000)[::-1]
    xWa = (c.c/xHz/u.Hz).decompose().to(u.um)[::-1].value
    deltaHz = xHz[1]-xHz[0]
    conversionFactor = 2.4873056783618645e-11  # mJy Hz Mpc^2 to lsol


def lineLIRTD(x, lam, eta):
    return lam * (x/1e12)**eta


def medFunc(x):
    '''
    return median. for vectorization of operations in pandas
    '''
    return np.median(x)
    

def lowSigma(x):
    '''
    return 16th percentile for vectorization of operations in pandas
    '''
    return np.median(x) - np.percentile(x, 16)


def higSigma(x):
    '''
    return 84th percentile for vectorization of operations in pandas
    '''
    return np.percentile(x, 84) - np.median(x)


def fixedValueReturns1(x):
    '''
    return 84th percentile for vectorization of operations in pandas
    '''
    n = np.empty_like(x)
    nones = x == n

    return nones


def return_LIR_LPeak_Arrays(fitFrame):
    """
    Return arrays of LIR, LPeak and their errors from a fit pandas dataframe

    Parameters
    ----------
    fitFrame : pandas DataFrame
        Pandas pymc3 fit dataFrame

    Returns
    -------
    fitLIR : ndarray
        numpy array filled with median LIR values from fit.

    lofitLIR : ndarray
        numpy array filled with 16th percentile LIR values from fit.

    hifitLIR : ndarray
        numpy array filled with 84th percentile LIR values from fit.

    errfitLIR : ndarray
        numpy array filled with 16th and 84th (1 sigma errors) LIR values from
        fit.

    fitLPeak : ndarray
        numpy array filled with median LPeak values from fit.

    lofitLPeak : ndarray
        numpy array filled with 16th percentile LPeak values from fit.

    hifitLPeak : ndarray
        numpy array filled with 84th percentile LPeak values from fit.

    errfitLPeak : ndarray
        numpy array filled with 16th and 84th (1 sigma errors) LPeak values
        from fit.

    """
    fitLIR = []
    lofitLIR = []
    hifitLIR = []
    fitLPeak = []
    lofitLPeak = []
    hifitLPeak = []
    indList = list(fitFrame.index)
    for i in indList:
        fitLIR.append(np.median(fitFrame.loc[i].trace_LIR))
        lofitLIR.append(np.percentile(fitFrame.loc[i].trace_LIR, 50) -
                        np.percentile(fitFrame.loc[i].trace_LIR, 16))
        hifitLIR.append(np.percentile(fitFrame.loc[i].trace_LIR, 84) -
                        np.percentile(fitFrame.loc[i].trace_LIR, 50))
        fitLPeak.append(np.median(fitFrame.loc[i].trace_lPeak))
        lofitLPeak.append(np.percentile(fitFrame.loc[i].trace_lPeak, 50) -
                          np.percentile(fitFrame.loc[i].trace_lPeak, 16))
        hifitLPeak.append(np.percentile(fitFrame.loc[i].trace_lPeak, 84) -
                          np.percentile(fitFrame.loc[i].trace_lPeak, 50))
    fitLIR = np.array(fitLIR)
    lofitLIR = np.array(lofitLIR)
    hifitLIR = np.array(hifitLIR)
    errfitLIR = np.array(list(zip(lofitLIR, hifitLIR))).transpose()
    fitLPeak = np.array(fitLPeak)
    lofitLPeak = np.array(lofitLPeak)
    hifitLPeak = np.array(hifitLPeak)
    errfitLPeak = np.array(list(zip(lofitLPeak, hifitLPeak))).transpose()

    return (fitLIR, lofitLIR, hifitLIR, errfitLIR, fitLPeak, lofitLPeak,
            hifitLPeak, errfitLPeak)


def return_alpha_distribution(fitFrame):
    """
    Return arrays of alpha and errors from a fit pandas dataframe

    Parameters
    ----------
    fitFrame : pandas DataFrame
        Pandas pymc3 fit dataFrame

    Returns
    -------
    alpha : ndarray
        numpy array filled with median alpha values from fit.

    lofitAlpha : ndarray
        numpy array filled with 16th percentile LIR values from fit.

    hifitAlpha : ndarray
        numpy array filled with 84th percentile LIR values from fit.

    errfitAlpha : ndarray
        numpy array filled with 16th and 84th (1 sigma errors) LIR values from
        fit.
    """
    alpha = []
    lofitAlpha = []
    hifitAlpha = []
    indList = list(fitFrame.index)
    for i in indList:
        alpha.append(np.median(fitFrame.loc[i].trace_alpha))
        lofitAlpha.append(np.percentile(fitFrame.loc[i].trace_alpha, 50) -
                        np.percentile(fitFrame.loc[i].trace_alpha, 16))
        hifitAlpha.append(np.percentile(fitFrame.loc[i].trace_alpha, 84) -
                        np.percentile(fitFrame.loc[i].trace_alpha, 50))
    alpha = np.array(alpha)
    lofitAlpha = np.array(lofitAlpha)
    hifitAlpha = np.array(hifitAlpha)
    errfitAlpha = np.array(list(zip(lofitAlpha, hifitAlpha))).transpose()

    return (alpha, lofitAlpha, hifitAlpha, errfitAlpha)


def return_beta_distribution(fitFrame):
    """
    Return arrays of beta and errors from a fit pandas dataframe

    Parameters
    ----------
    fitFrame : pandas DataFrame
        Pandas pymc3 fit dataFrame

    Returns
    -------
    beta : ndarray
        numpy array filled with median beta values from fit.

    lofitBeta : ndarray
        numpy array filled with 16th percentile beta values from fit.

    hifitBeta : ndarray
        numpy array filled with 84th percentile LIR values from fit.

    errfitBeta : ndarray
        numpy array filled with 16th and 84th (1 sigma errors) beta values from
        fit.
    """
    beta = []
    lofitBeta = []
    hifitBeta = []
    indList = list(fitFrame.index)
    for i in indList:
        beta.append(np.median(fitFrame.loc[i].trace_beta))
        lofitBeta.append(np.percentile(fitFrame.loc[i].trace_beta, 50) -
                        np.percentile(fitFrame.loc[i].trace_beta, 16))
        hifitBeta.append(np.percentile(fitFrame.loc[i].trace_beta, 84) -
                        np.percentile(fitFrame.loc[i].trace_beta, 50))
    beta = np.array(beta)
    lofitBeta = np.array(lofitBeta)
    hifitBeta = np.array(hifitBeta)
    errfitBeta = np.array(list(zip(lofitBeta, hifitBeta))).transpose()

    return (beta, lofitBeta, hifitBeta, errfitBeta)


def return_tdust_distribution(fitFrame):
    """
    Return arrays of Tdust and errors from a fit pandas dataframe

    Parameters
    ----------
    fitFrame : pandas DataFrame
        Pandas pymc3 fit dataFrame

    Returns
    -------
    Tdust : ndarray
        numpy array filled with median tdust values from fit.

    lofitTdust : ndarray
        numpy array filled with 16th percentile tdust values from fit.

    hifitTdust : ndarray
        numpy array filled with 84th percentile LIR values from fit.

    errfitTdust : ndarray
        numpy array filled with 16th and 84th (1 sigma errors) tdust values from
        fit.
    """
    Tdust = []
    lofitTdust = []
    hifitTdust = []
    indList = list(fitFrame.index)
    for i in indList:
        Tdust.append(np.median(fitFrame.loc[i].trace_tdust))
        lofitTdust.append(np.percentile(fitFrame.loc[i].trace_tdust, 50) -
                        np.percentile(fitFrame.loc[i].trace_tdust, 16))
        hifitTdust.append(np.percentile(fitFrame.loc[i].trace_tdust, 84) -
                        np.percentile(fitFrame.loc[i].trace_tdust, 50))
    tdust = np.array(tdust)
    lofitTdust = np.array(lofitTdust)
    hifitTdust = np.array(hifitTdust)
    errfitTdust = np.array(list(zip(lofitTdust, hifitTdust))).transpose()

    return (tdust, lofitTdust, hifitTdust, errfitTdust)


def return_tdust_distribution(fitFrame):
    """
    Return arrays of Tdust and errors from a fit pandas dataframe

    Parameters
    ----------
    fitFrame : pandas DataFrame
        Pandas pymc3 fit dataFrame

    Returns
    -------
    Tdust : ndarray
        numpy array filled with median tdust values from fit.

    lofitTdust : ndarray
        numpy array filled with 16th percentile tdust values from fit.

    hifitTdust : ndarray
        numpy array filled with 84th percentile LIR values from fit.

    errfitTdust : ndarray
        numpy array filled with 16th and 84th (1 sigma errors) tdust values from
        fit.
    """
    (norm1Array, TdustArray, alphaArray, betaArray, w0Array, LIRArray,
            lPeakArray) = returnFitParamArrays(trace, fixAlphaValue, fixBetaValue, fixW0Value)
    Tdust = []
    lofitTdust = []
    hifitTdust = []
    indList = list(fitFrame.index)
    for i in indList:
        Tdust.append(np.median(fitFrame.loc[i].trace_tdust))
        lofitTdust.append(np.percentile(fitFrame.loc[i].trace_tdust, 50) -
                        np.percentile(fitFrame.loc[i].trace_tdust, 16))
        hifitTdust.append(np.percentile(fitFrame.loc[i].trace_tdust, 84) -
                        np.percentile(fitFrame.loc[i].trace_tdust, 50))
    tdust = np.array(tdust)
    lofitTdust = np.array(lofitTdust)
    hifitTdust = np.array(hifitTdust)
    errfitTdust = np.array(list(zip(lofitTdust, hifitTdust))).transpose()

    return (tdust, lofitTdust, hifitTdust, errfitTdust)


def returnFitParamArrays(trace, fixAlphaValue, fixBetaValue, fixW0Value):
    """
    Return arrays filled with fixed and or fitted parameter values.

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
    '''
    Return median fit parameters
    
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
    '''
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
    '''
    Adds best values and 1sigma errors to the fit dataframe.
    '''

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


def temperatureGivenBetaLPeak_old(beta, lPeak, w0):
    '''
    Function to return a temperature given beta and peak wavelength
    '''

    TdustArrayCoarse = np.linspace(3, 300, 500)
    lPeakArrayCoarse = np.ones(len(TdustArrayCoarse))
    for i in list(range(len(TdustArrayCoarse))):
        # print(i)
        '''
        lambdaPeak doesn't depend on norm1 or alpha, so setting them to 2.
        '''
        lPeakArrayCoarse[i] = mcirsed_ff.lambdaPeak(2., TdustArrayCoarse[i], 2.,
                                               beta, w0)
    closeInd = findmyindex.index(lPeakArrayCoarse, lPeak)
    lPeakArrayFine = np.logspace(np.log10(lPeakArrayCoarse[closeInd-1]),
                                 np.log10(lPeakArrayCoarse[closeInd+1]),
                                 10000)[::-1]
    TdustArrayFine = np.logspace(np.log10(TdustArrayCoarse[closeInd-1]),
                                 np.log10(TdustArrayCoarse[closeInd+1]),
                                 10000)[::-1]

    closesrInd = findmyindex.index(lPeakArrayFine, lPeak)
    lPeakArrayFiner = np.logspace(np.log10(lPeakArrayFine[closesrInd-1]),
                                  np.log10(lPeakArrayFine[closesrInd+1]),
                                  10000)
    TdustArrayFiner = np.logspace(np.log10(TdustArrayFine[closesrInd-1]),
                                  np.log10(TdustArrayFine[closesrInd+1]),
                                  10000)

    closestInd = findmyindex.index(lPeakArrayFiner, lPeak)
    return TdustArrayFiner[closestInd]


"""def temperatureGivenBetaLPeak(beta, lPeak, w0):
    '''
    Function to return a temperature given beta and peak wavelength
    '''
    tdusts = np.logspace(np.log10(3), np.log10(300), 100000)
    lpeak = np.vectorize(mcirsed_ff.lambdaPeak)
    lpeaks = lpeak(2., tdusts, 2., beta, w0)
    ind = np.searchsorted()
    return """


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


def fluxLimit60um(fluxLimit, z, lPeak, fixAlphaValue, fixBetaValue, fixW0Value):
    """
    Find the parameters associated with the 60 um flux limit.
    The difference with the function fluxLimit60umLIR_LPeak is that this function
    does not allow lPeak to vary.
    """

    Tdust = temperatureGivenBetaLPeak(fixBetaValue, lPeak, fixW0Value)
    possibleLIRs = np.logspace(1, 14, 500)
    Snu60List = np.zeros([len(possibleLIRs)])
    norm1List = np.zeros(len(possibleLIRs))
    for i in list(range(len(possibleLIRs))):
        norm1List[i] = normalization(possibleLIRs[i], z, Tdust, fixAlphaValue, fixBetaValue, fixW0Value)
        Snu60List[i] = mcirsed_ff.SnuNoBump(norm1List[i], Tdust, fixAlphaValue, fixBetaValue, fixW0Value, 60.0/(1+z), h.fineRestWave, h.hck)
    closeInd = findmyindex.index(Snu60List, fluxLimit)

    possibleLIRsFine = np.linspace(possibleLIRs[closeInd-1], possibleLIRs[closeInd+1], 500)
    Snu60ListFine = np.zeros([len(possibleLIRsFine)])
    norm1ListFine = np.zeros(len(possibleLIRsFine))
    for i in list(range(len(possibleLIRsFine))):
        norm1ListFine[i] = normalization(possibleLIRsFine[i], z, Tdust, fixAlphaValue, fixBetaValue, fixW0Value)
        Snu60ListFine[i] = mcirsed_ff.SnuNoBump(norm1ListFine[i], Tdust, fixAlphaValue, fixBetaValue, fixW0Value, 60.0/(1+z), h.fineRestWave, h.hck)
    closerInd = findmyindex.index(Snu60ListFine, fluxLimit)
    # print('delta Snu:')
    # print(min(abs(fluxLimit-Snu60ListFine)))
    # print('calculated Snu:')
    # print(Snu60ListFine[closerInd])

    return norm1ListFine[closerInd], Tdust


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
'''

    possibleLIRs = np.logspace(6, 13, nPoints) # 8, 13
    possibleLPeaks = np.logspace(np.log10(70), np.log10(300), nPoints) # 70, 210 # 30, 300 # 

    Snu60List = np.zeros([nPoints, nPoints])
    norm1List = np.zeros(nPoints)  # np.zeros([nPoints, nPoints])
    possibleTdusts = np.zeros(nPoints)
    closeInd = np.zeros(nPoints, dtype=int)
    norm1ListFinal = np.zeros(nPoints)
    for i in list(range(nPoints)):
        print('calculating possible dust temps '+ str(i) + '/' + str(nPoints-1))
        possibleTdusts[i] = temperatureGivenBetaLPeak(fixBetaValue, possibleLPeaks[i], fixW0Value)
        norm1List[i] = normalizationForFluxLimit(possibleLIRs[i], z, possibleTdusts[i], fixAlphaValue, fixBetaValue, fixW0Value)
    for i in list(range(nPoints)):
        print('calculating flux densities: ' + str(i) + '/'+str(nPoints))
        for j in list(range(nPoints)):
            Snu60List[i,j] = mcirsed_ff.SnuNoBump(norm1List[i], possibleTdusts[j], fixAlphaValue, fixBetaValue, fixW0Value, 60.0/(1+z), h.fineRestWave, h.hck)
    for i in list(range(nPoints)):
        print('finding index of best normalization: ' + str(i) + '/'+str(nPoints))
        closeInd[i] = findmyindex.index(Snu60List[:,i], fluxLimit)
        norm1ListFinal[i] = norm1List[closeInd[i]]

    return norm1ListFinal, possibleTdusts
'''

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


def fluxLimit250umLIR_LPeakOldest(fluxLimit, z, fixAlphaValue, fixBetaValue, fixW0Value, nPoints, limLIRLow, limLIRHigh, limLPeakLow, limLPeakHigh):
    """
    Find the parameters associated with the 250 um flux limit for the LIR_LP plot.
    The difference with the function fluxLimit60um is that this allows lPeak to vary.
    """

    possibleLIRs = np.logspace(np.log10(limLIRLow), np.log10(limLIRHigh), nPoints) # np.logspace(6, 13, nPoints)
    possibleLPeaks = np.logspace(np.log10(limLPeakLow), np.log10(limLPeakHigh), nPoints) # np.logspace(np.log10(70), np.log10(300), nPoints)
    xx, yy = np.meshgrid(possibleLIRs, possibleLPeaks)

    # Snu250List = np.zeros([nPoints, nPoints])
    Snu250List = xx * 0
    norm1List = np.zeros(nPoints)  # np.zeros([nPoints, nPoints])
    possibleTdusts = np.zeros(nPoints)
    closeInd = np.zeros(nPoints, dtype=int)
    norm1ListFinal = np.zeros(nPoints)
    for i in list(range(nPoints)):
        print('calculating possible dust temps '+ str(i) + '/' + str(nPoints-1))
        possibleTdusts[i] = temperatureGivenBetaLPeak(fixBetaValue, possibleLPeaks[i], fixW0Value)
        norm1List[i] = normalizationForFluxLimit(possibleLIRs[i], z, possibleTdusts[i], fixAlphaValue, fixBetaValue, fixW0Value)
    for i in list(range(nPoints)):
        print('calculating flux densities: ' + str(i) + '/'+str(nPoints))
        for j in list(range(nPoints)):
            Snu250List[i,j] = mcirsed_ff.SnuNoBump(norm1List[i], possibleTdusts[j], fixAlphaValue, fixBetaValue, fixW0Value, 250.0/(1+z), h.fineRestWave, h.hck)
    for i in list(range(nPoints)):
        print('finding index of best normalization: ' + str(i) + '/'+str(nPoints))
        closeInd[i] = findmyindex.index(Snu250List[:,i], fluxLimit)
        norm1ListFinal[i] = norm1List[closeInd[i]]

    return norm1ListFinal, possibleTdusts


def fluxLimit250umLIR_LPeakOld(fluxLimit, z, fixAlphaValue, fixBetaValue, fixW0Value, nPoints, limLIRLow, limLIRHigh, limLPeakLow, limLPeakHigh):
    """
    Find the parameters associated with the 250 um flux limit for the LIR_LP plot.
    The difference with the function fluxLimit60um is that this allows lPeak to vary.
    """

    possibleLIRs = np.logspace(np.log10(limLIRLow), np.log10(limLIRHigh), nPoints) # np.logspace(6, 13, nPoints)
    possibleLPeaks = np.logspace(np.log10(limLPeakLow), np.log10(limLPeakHigh), nPoints) # np.logspace(np.log10(70), np.log10(300), nPoints)
    xx, yy = np.meshgrid(possibleLIRs, possibleLPeaks)
    # fourPiLumDistSquared = ((4*np.pi*cosmo.luminosity_distance(z)**2.).value * h.conversionFactor)

    TdustList = np.zeros(np.shape(yy[:, 0]))
    TdustError = np.zeros(np.shape(yy[:, 0]))
    for i in list(range(len(yy[:, 0]))):
        print('calculating grid dust temps '+ str(i) + '/' + str(nPoints-1))
        TdustList[i] = temperatureGivenBetaLPeak(fixBetaValue, yy[i, 0], fixW0Value)
        TdustError[i] = checkTemperature(fixBetaValue, yy[i, 0], fixW0Value, TdustList[i], verbose=False)


    testnorm = TdustList * 0
    testCheckLIR = testnorm * 0
    testLIRVals = testnorm * 0
    for i in list(range(len(xx[0, :]))): 
        testInputLIR = 1e10
        testInputz = 0.05
        fourPiLumDistSquaredtest = ((4*np.pi*cosmo.luminosity_distance(testInputz)**2.).value * h.conversionFactor)
        testnorm[i] = normalizationForFluxLimit(testInputLIR, testInputz, TdustList[i], fixAlphaValue, fixBetaValue, fixW0Value)
        testCheckLIR[i] = checkLIR(testnorm[i], TdustList[i], fixAlphaValue, fixBetaValue, fixW0Value, testInputz, testInputLIR, verbose=True)
        testLIRVals[i] = mcirsed_ff.IRLum(testnorm[i], TdustList[i], fixAlphaValue, fixBetaValue, fixW0Value, h.fineRestWave, h.fineRestWave, h.hck, h.deltaHz, fourPiLumDistSquaredtest) / (1+testInputz)

        plt.plot(h.fineRestWave, mcirsed_ff.SnuNoBump(testnorm[i], TdustList[i], fixAlphaValue, fixBetaValue, fixW0Value, h.fineRestWave, h.fineRestWave, h.hck))
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    norm1List = xx * 0
    checkLIRList = xx * 0
    for i in list(range(len(xx[0, :]))):
        print('calculating normalizations to match LIRs ' + str(i) + '/' + str(nPoints-1))
        for j in list(range(len(yy[:, 0]))):
            norm1List[i, j] = normalizationForFluxLimit(xx[0, i], z, TdustList[j], fixAlphaValue, fixBetaValue, fixW0Value)  # use 0th index in xx[0, i] because xx[:, i] are all the same
            checkLIRList[i, j] = checkLIR(norm1List[i, j], TdustList[j], fixAlphaValue, fixBetaValue, fixW0Value, z, xx[i, j], verbose=False)

    testLum = np.zeros(np.shape(yy[:, 0]))
    for i in list(range(len(xx[0, :]))):
        # for j in list(range(len(yy[:, 0]))):
        #     print(str(i) + ' ' + str(j))
        plt.plot(h.fineRestWave, mcirsed_ff.SnuNoBump(norm1List[i, i], TdustList[i], fixAlphaValue, fixBetaValue, fixW0Value, h.fineRestWave, h.fineRestWave, h.hck))
        testLum[i] = mcirsed_ff.IRLum(norm1List[0, i], TdustList[i], fixAlphaValue, fixBetaValue, fixW0Value, h.fineRestWave, h.fineRestWave, h.hck, h.deltaHz, fourPiLumDistSquared)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    Snu250List = norm1List * 0
    for i in list(range(nPoints)):
        print('calculating flux densities: ' + str(i) + '/'+str(nPoints))
        for j in list(range(nPoints)):
            Snu250List[i,j] = mcirsed_ff.SnuNoBump(norm1List[i, j], TdustList[j], fixAlphaValue, fixBetaValue, fixW0Value, 250.0/(1+z), h.fineRestWave, h.hck)

    norm1ListFinal = TdustList * 0
    closeInd = np.zeros(nPoints, dtype=int)
    for i in list(range(nPoints)):
        print('finding index of best normalization: ' + str(i) + '/'+str(nPoints))
        closeInd[i] = findmyindex.index(Snu250List[:,i], fluxLimit)
        norm1ListFinal[i] = norm1List[closeInd[i]]

    '''liroutputList = np.zeros(np.shape(xx[0, :]))
    for i in list(range(len(xx[0, :]))):
        liroutputList[i] = mcirsed_ff.IRLum(norm1List[i], TdustList[i], fixAlphaValue, fixBetaValue, fixW0Value, h.xWa, h.fineRestWave, h.hck, h.deltaHz, fourPiLumDistSquared)'''



    # mcirsed_ff.SnuNoBump(norm1List, TdustList, fixAlphaValue, fixBetaValue, fixW0Value, 250.0/(1+z), h.fineRestWave, h.hck)

    return norm1ListFinal, TdustList


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
