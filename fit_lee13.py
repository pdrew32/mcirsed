import numpy as np
import pymc3 as pm
import anahelper as ah
import mcirsed
import corner
import mcirsed_ff
import pandas as pd
import sys
from matplotlib import pyplot as plt

'''
Script to fit Lee+13 galaxies
'''

freshRun = False
runNum = 0
plotOutput = False
inflateErrors = False

write_file_sed_fit = 'lee13_fit.pkl'

read_file_data = '../data/lee13data.csv'
dataF = pd.read_csv(read_file_data, index_col=0)

whichind = 920 # 450
endInd = 1000 # whichind + 1 # 105 # 

if freshRun is True:
    contin = input(
        'you will overwrite/create new file. Type yes to continue \n')
    if contin != 'yes':
        sys.exit('Stopping Code')

###############################################################################
intList = dataF.index[whichind:endInd] # refit_inds[whichind:endInd] # 
###############################################################################

upTdust = 150

fixBetaValue = None # 2.0 # 
fixAlphaValue = None # 2.0 # 
flat_alpha_prior = False
flat_beta_prior = False
fixW0Value = 200
tune = 3000
MCSamples = 5000

dataLIR = None
errLIR = None
CMBCorrection = False

###############################################################################

###############################################################################

counter = 0
print(counter)

k = whichind
for i in intList:
    print('\n')
    print('galaxy '+str(i+1)+'/'+str(max(intList)+1))
    print(str(k) + '/' + str(endInd-1))
    k += 1

    dWave = np.array([24.0, 70.0, 100.0, 160.0, 250.0, 350.0, 500.0, 850.0, 1100.0, 1200.0])
    dataFlux = np.array([dataF.loc[i].F24, dataF.loc[i].F70, dataF.loc[i].F100, dataF.loc[i].F160, dataF.loc[i].F250, dataF.loc[i].F350, dataF.loc[i].F500, dataF.loc[i].s850, dataF.loc[i].F11mm, dataF.loc[i].F12mm])
    dataErr = np.abs([dataF.loc[i].E24, dataF.loc[i].E70, dataF.loc[i].E100, dataF.loc[i].E160, dataF.loc[i].E250, dataF.loc[i].E350, dataF.loc[i].E500, dataF.loc[i].e850, dataF.loc[i].E11mm, dataF.loc[i].E12mm])

    # if 100 or 160 flux density is 0 fix alpha and beta
    '''if np.any(dataFlux[np.where((dWave == 100.0) | (dWave == 160.0))] == 0):
        fixAlphaValue = 2.0
        fixBetaValue = 2.0'''

    dataFlux = dataFlux[dataErr > 0]
    dWave = dWave[dataErr > 0]
    dataErr = dataErr[dataErr > 0]

    # cap snr of 24um point to 10
    if dataF.loc[i].F24 / dataF.loc[i].E24 > 10:
        dataErr[np.where(dWave == 24.0)[0]] = dataF.loc[i].F24/10

    if inflateErrors is True:
        dataErr[np.abs(dataErr/dataFlux) < 0.12] = 0.12*dataFlux[np.abs(dataErr/dataFlux) < 0.12]  # Where flux is very certain, inflate the errors to 12%

    # upper limit for norm1, based on the max of the dataflux.
    upperLim = np.log10(np.round(max(dataFlux)*1e9))

    z = dataF.loc[i].z_use

    '''# if 350 and 500 microns are low snr fix beta to 2
    if fixBetaValue is None:
        if (dataFlux[dWave == 350.0]/dataErr[dWave == 350.0] < 5.0) & (dataFlux[dWave == 500.0]/dataErr[dWave == 500.0] < 5.0):
            fixBetaValue = 2.0
        else: 
            fixBetaValue = None

        if (dataFlux[dWave == 350] < dataFlux[dWave == 500]):
            fixBetaValue = 2.0'''
    
    tr = mcirsed.mcirsed(dWave, dataFlux, dataErr, z,
                                 fixAlpha=fixAlphaValue, fixBeta=fixBetaValue,
                                 fixW0=fixW0Value,
                                 CMBCorrection=CMBCorrection,
                                 MCSamples=MCSamples, tune=tune,
                                 upNorm1=upperLim, loNorm1=upperLim - 4,
                                 upTdust=upTdust, flat_alpha_prior=flat_alpha_prior,
                                 flat_beta_prior=flat_beta_prior)

    ###########################################################################
    # plot it

    if plotOutput is True:
        inp = ah.returnMedianParams(tr, fixAlphaValue,
                                                fixBetaValue, fixW0Value)
        best = mcirsed_ff.SnuNoBump(inp[0], inp[1], inp[2], inp[3], inp[4], ah.h.xWa)

        for j in list(range(100)):
            n = np.random.randint(0, len(tr['norm1']))
            nrand = tr['norm1'][n]
            trand = tr['Tdust'][n]
            if fixAlphaValue is None:
                arand = tr['alpha'][n]
            else:
                arand = inp[2]
            if fixBetaValue is None:
                brand = tr['Beta'][n]
            else:
                brand = inp[3]
            if fixW0Value is None:
                wrand = tr['w0'][n]
            else:
                wrand = inp[-1]
            ychains = mcirsed_ff.SnuNoBump(nrand, trand, arand, brand, wrand, ah.h.xWa)
            plt.plot(ah.h.xWa*(1+z), ychains, color='grey', alpha=0.15, zorder=-2)

        plt.plot(ah.h.xWa*(1+z), best, color='black')
        plt.xlabel('observed wavelength (microns)')
        plt.errorbar(dWave, dataFlux, yerr=dataErr, fmt='o', color='red',
                     zorder=1)
        plt.xscale('log')
        plt.yscale('log')
        plt.show(block=True)

        #######################################################################
        arr, lab = ah.cornerHelper(tr, fixAlphaValue, fixBetaValue, fixW0Value)
        figure2 = corner.corner(arr, labels=lab, quantiles=[0.16, 0.5, 0.84],
                                show_titles=True, title_kwargs={"fontsize": 12}
                                )
        plt.show(block=True)
    ###########################################################################

    # Make sure the matrix containing the fit params is broadcasted to be the
    # same size no matter the free parameters.
    traceMatrix = ah.returnFitParamArrays(tr, fixAlphaValue, fixBetaValue,
                                     fixW0Value)

    if freshRun is True and counter == 0:
        print('im in the first one')
        dataFrame = pd.DataFrame(
            data=[[z, fixAlphaValue, fixBetaValue, fixW0Value, tune,
                   MCSamples, CMBCorrection, traceMatrix[0],
                   traceMatrix[1], traceMatrix[2], traceMatrix[3],
                   traceMatrix[4], traceMatrix[5], traceMatrix[6], dWave, dataFlux, dataErr, runNum]],
            index=[i],
            columns=['z', 'fixAlphaValue', 'fixBetaValue', 'fixW0Value',
                     'tune', 'MCSamples',
                     'CMBCorrection', 'trace_Norm1', 'trace_Tdust',
                     'trace_alpha', 'trace_beta', 'trace_w0', 'trace_LIR',
                     'trace_lPeak', 'dataWave', 'dataFlux', 'dataErr', 'runNum']
                                 )
        counter += 1

        print(dataFrame.tail(n=5))

        dataFrame.to_pickle(
            '../data/' + write_file_sed_fit
            )

    else:
        print('im in the second one')
        dFrame = pd.read_pickle(
            '../data/' + write_file_sed_fit
            )

        counter += 1

        newEntry = pd.DataFrame(data=[[z, fixAlphaValue, fixBetaValue,
                                       fixW0Value, tune, MCSamples, CMBCorrection, traceMatrix[0],
                                       traceMatrix[1], traceMatrix[2],
                                       traceMatrix[3], traceMatrix[4],
                                       traceMatrix[5], traceMatrix[6], dWave, dataFlux, dataErr, runNum]
                                      ], index=[i],
                                columns=['z', 'fixAlphaValue', 'fixBetaValue',
                                         'fixW0Value', 'tune', 'MCSamples',
                                         'CMBCorrection', 'trace_Norm1',
                                         'trace_Tdust', 'trace_alpha',
                                         'trace_beta', 'trace_w0', 'trace_LIR',
                                         'trace_lPeak', 'dataWave', 'dataFlux', 'dataErr', 'runNum'])

        if i in dFrame.index:
            dFrame.update(newEntry)
            dFrame.to_pickle('../data/' +
                             write_file_sed_fit)
        else:
            tempFrame = dFrame.append(newEntry)
            tempFrame.sort_index(inplace=True)
            tempFrame.to_pickle('../data/' +
                                write_file_sed_fit)

        print(dFrame.tail(n=5))
