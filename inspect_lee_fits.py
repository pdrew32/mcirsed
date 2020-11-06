import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

"""
Inspect different correlations between fit parameters for different regimes of lirtd for lee+13 sample
"""

read_file_fits = '../data/lee13_fit_best_fit_params_added.pkl'
fitF = pd.read_pickle(read_file_fits)

read_file_fits = '../data/drew_fits_params_added.pkl'
dF = pd.read_pickle(read_file_fits)

# look at the population of galaxies at lpeaks < 70 that don't lie on the correlation
ins_inds = fitF.measuredLPeak < 70

# plot histograms
plt.hist(fitF.loc[ins_inds, 'z'], alpha=0.7, label='lpeak < 70', hatch='/', zorder=99)
plt.hist(fitF.loc[~ins_inds, 'z'], alpha=0.7, label='lpeak > 70', hatch='\ ')
plt.legend(fontsize='x-small')
plt.show()


# z vs alpha
plt.scatter(fitF.z, fitF.measuredAlpha)
plt.xlabel('redshift')
plt.ylabel(r'$\alpha$')
plt.show()


# z vs beta
plt.scatter(fitF.z, fitF.measuredBeta)
plt.xlabel('redshift')
plt.ylabel(r'$\beta$')
plt.show()


# beta vs alpha
plt.scatter(fitF.measuredBeta, fitF.measuredAlpha)
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\alpha$')
plt.show()


# alpha vs lpeak
plt.scatter(fitF.measuredAlpha, fitF.measuredLPeak)
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\lambda_{peak}$')
plt.show()


# alpha vs LIR
plt.scatter(fitF.measuredAlpha, 10**fitF.measuredLIR)
plt.yscale('log')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'LIR')
plt.show()

