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
Example script for fitting with MCIRSED
'''

###############################################################################
# NOTE: you should only need to edit lines between these comment bars (the line aboove)

# True for new file or to overwrite an old one. False to append new fits to an old file. 
# NOTE: new runs must be run with freshRun = True
freshRun = True 

# do you want to see a plot of the output?
plotOutput = True 

# name the output file. must be a .pkl file
write_file_sed_fit = 'sample_sed_fits.pkl'

# location of data file
read_file_data = '../data/sample_sed_data.csv'
dataF = pd.read_csv(read_file_data, index_col=0)

# which indices in the data file do you want to fit?
# e.g. startind = 0, endind = 100; or startind = 0, endind = startind + 1
startind = 0
endind = startind + 1

# upper limit for dust temperature. If failing to converge, lower this limit
upTdust = 300

# put the list of wavelengths of observation here
dWave_master = np.array([24.0, 70.0, 100.0, 160.0, 250.0, 350.0, 500.0, 850.0, 1100.0]) 

# fit parameter settings:
fixBetaValue = None # value to fix dust emissivity, beta to or None if a free parameter
fixAlphaValue = None # value to fix power law slope, alpha to or None if a free parameter
fixW0Value = 200 # wavelength to fix lambda_0 to, the wavelength where opacity = 1 or None if a free parameter. NOTE: beta, dust temp (equivalently, peak wavlength), and lambda_0 are highly degenerate so it is our practice to fix fixW0Value with the understanding that the dust temperature, beta, and lambda_0 derived from this code are not physical unless this code is being used for data that truly constrains these parameters. Values other than 200 microns may be preferable, depending on your use case.
CMBCorrection = False # do you want to correct for effects from CMB heating? applicable when dust temperature is expected to be close to Tcmb at the redshift of the galaxy

tune = 3000 # how many mcmc steps to tune for. Essentially the burn in equivalent for pymc3
MCSamples = 5000 # how many steps to sample for

# There are more user inputs between comment lines below if changing the default way data is stored
###############################################################################

if freshRun is True:
    contin = input(
        'you will overwrite/create new file. Type yes to continue \n')
    if contin != 'yes':
        sys.exit('Stopping Code')

intList = dataF.index[startind:endind]

counter = 0
k = startind
for i in intList:
    print('\n')
    print('galaxy '+str(i+1)+'/'+str(max(intList)+1))
    print(str(k) + '/' + str(endind-1))
    k += 1

    ########################################################################################
    # NOTE: edit here if changing anything in input data file
    dWave = dWave_master
    dataFlux = np.array([dataF.loc[i].s24, dataF.loc[i].s70, dataF.loc[i].s100, dataF.loc[i].s160, dataF.loc[i].s250, dataF.loc[i].s350, dataF.loc[i].s500, dataF.loc[i].s850, dataF.loc[i].s1100])
    dataErr = np.abs([dataF.loc[i].e24, dataF.loc[i].e70, dataF.loc[i].e100, dataF.loc[i].e160, dataF.loc[i].e250, dataF.loc[i].e350, dataF.loc[i].e500, dataF.loc[i].e850, dataF.loc[i].e1100])
    z = dataF.loc[i].z
    # 
    ########################################################################################
    
    # remove wavelengths with negative errors or nan errors
    dataFlux = dataFlux[dataErr > 0]
    dWave = dWave[dataErr > 0]
    dataErr = dataErr[dataErr > 0]

    # upper limit for norm1, based on the max of the dataflux.
    upperLim = np.log10(np.round(max(dataFlux)*1e9))
    
    # do the SED fitting
    tr = mcirsed.mcirsed(dWave, dataFlux, dataErr, z,
                                 fixAlpha=fixAlphaValue, fixBeta=fixBetaValue,
                                 fixW0=fixW0Value,
                                 CMBCorrection=CMBCorrection,
                                 MCSamples=MCSamples, tune=tune,
                                 upNorm1=upperLim, loNorm1=upperLim - 4,
                                 upTdust=upTdust)

    # plot it
    if plotOutput is True:
        inp = ah.returnMedianParams(tr, fixAlphaValue,
                                                fixBetaValue, fixW0Value)
        best = mcirsed_ff.SnuNoBump(inp[0], inp[1], inp[2], inp[3], inp[4], ah.h.xWa)

        # plot 100 random mcmc chains in grey
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

        # plot best fit
        plt.plot(ah.h.xWa*(1+z), best, color='black')
        plt.xlabel('observed wavelength (microns)')
        plt.errorbar(dWave, dataFlux, yerr=dataErr, fmt='o', color='red',
                     zorder=1)
        plt.xscale('log')
        plt.yscale('log')
        plt.show(block=True)

        arr, lab = ah.cornerHelper(tr, fixAlphaValue, fixBetaValue, fixW0Value)
        figure2 = corner.corner(arr, labels=lab, quantiles=[0.16, 0.5, 0.84],
                                show_titles=True, title_kwargs={"fontsize": 12}
                                )
        plt.show(block=True)

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
                   traceMatrix[4], traceMatrix[5], traceMatrix[6], np.median(traceMatrix[0]), np.median(traceMatrix[1]), np.median(traceMatrix[2]), np.median(traceMatrix[3]), np.median(traceMatrix[4]), np.median(traceMatrix[5]), np.median(traceMatrix[6]), np.percentile(traceMatrix[0], 15.87), np.percentile(traceMatrix[1], 15.87), np.percentile(traceMatrix[2], 15.87), np.percentile(traceMatrix[3], 15.87), np.percentile(traceMatrix[4], 15.87), np.percentile(traceMatrix[5], 15.87), np.percentile(traceMatrix[6], 15.87), np.percentile(traceMatrix[0], 84.13), np.percentile(traceMatrix[1], 84.13), np.percentile(traceMatrix[2], 84.13), np.percentile(traceMatrix[3], 84.13), np.percentile(traceMatrix[4], 84.13), np.percentile(traceMatrix[5], 84.13), np.percentile(traceMatrix[6], 84.13), dWave, dataFlux, dataErr]],
            index=[i],
            columns=['z', 'fixAlphaValue', 'fixBetaValue', 'fixW0Value',
                     'tune', 'MCSamples',
                     'CMBCorrection', 'trace_Norm1', 'trace_Tdust',
                     'trace_alpha', 'trace_beta', 'trace_w0', 'trace_LIR',
                     'trace_lPeak', 'median_Norm1', 'median_Tdust','median_alpha', 'median_beta', 'median_w0', 'median_LIR',
                     'median_lPeak', 'Norm1_16th', 'Tdust_16th', 'alpha_16th', 'beta_16th', 'w0_16th', 'LIR_16th',
                     'lPeak_16th', 'Norm1_84th', 'Tdust_84th', 'alpha_84th', 'beta_84th', 'w0_84th', 'LIR_84th',
                     'lPeak_84th', 'dataWave', 'dataFlux', 'dataErr']
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

        newEntry = pd.DataFrame(data=[[z, fixAlphaValue, fixBetaValue, fixW0Value, tune, MCSamples, CMBCorrection, traceMatrix[0], traceMatrix[1], traceMatrix[2], traceMatrix[3], traceMatrix[4], traceMatrix[5], traceMatrix[6], np.median(traceMatrix[0]), np.median(traceMatrix[1]), np.median(traceMatrix[2]), np.median(traceMatrix[3]), np.median(traceMatrix[4]), np.median(traceMatrix[5]), np.median(traceMatrix[6]), np.percentile(traceMatrix[0], 15.87), np.percentile(traceMatrix[1], 15.87), np.percentile(traceMatrix[2], 15.87), np.percentile(traceMatrix[3], 15.87), np.percentile(traceMatrix[4], 15.87), np.percentile(traceMatrix[5], 15.87), np.percentile(traceMatrix[6], 15.87), np.percentile(traceMatrix[0], 84.13), np.percentile(traceMatrix[1], 84.13), np.percentile(traceMatrix[2], 84.13), np.percentile(traceMatrix[3], 84.13), np.percentile(traceMatrix[4], 84.13), np.percentile(traceMatrix[5], 84.13), np.percentile(traceMatrix[6], 84.13), dWave, dataFlux, dataErr]
                                      ], index=[i],
                                columns=['z', 'fixAlphaValue', 'fixBetaValue', 'fixW0Value', 'tune', 'MCSamples', 'CMBCorrection', 'trace_Norm1', 'trace_Tdust', 'trace_alpha', 'trace_beta', 'trace_w0', 'trace_LIR', 'trace_lPeak', 'median_Norm1', 'median_Tdust','median_alpha', 'median_beta', 'median_w0', 'median_LIR', 'median_lPeak', 'Norm1_16th', 'Tdust_16th', 'alpha_16th', 'beta_16th', 'w0_16th', 'LIR_16th', 'lPeak_16th', 'Norm1_84th', 'Tdust_84th', 'alpha_84th', 'beta_84th', 'w0_84th', 'LIR_84th', 'lPeak_84th', 'dataWave', 'dataFlux', 'dataErr'])

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
