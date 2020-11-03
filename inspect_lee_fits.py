import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

"""
Inspect different correlations between fit parameters for different regimes of lirtd for lee+13 sample
"""

read_file_fits = '../data/lee13_fit_best_fit_params_added.pkl'
fitF = pd.read_pickle(read_file_fits)

# look at the population of galaxies at lpeaks < 70 that don't lie on the correlation
ins_inds = fitF.measuredLPeak < 70

# plot histograms
plt.hist(fitF.loc[ins_inds, 'z'], alpha=0.7, label='lpeak < 70', hatch='/', zorder=99)
plt.hist(fitF.loc[~ins_inds, 'z'], alpha=0.7, label='lpeak > 70', hatch='\ ')
plt.legend(fontsize='x-small')
plt.show()
