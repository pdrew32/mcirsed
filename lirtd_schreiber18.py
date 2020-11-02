import numpy as np
import pandas as pd
import anahelper as ah
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import ticker


"""
plot schreiber+18 stacked galaxies on the lirtd correlation from the iras sample
"""

# load schreiber fits
read_file_data = '../data/schreiber18_fit_best_fit_params_added.pkl'
fitF = pd.read_pickle(read_file_data)

read_file_data_min_z = '../data/schreiber18_fit_min_z_best_fit_params_added.pkl'
fitF_min_z = pd.read_pickle(read_file_data_min_z)

# load correlation params for iras
read_file_corr_drew = 'lir_lpeak_corr_best_fit_drew.csv'
dF = pd.read_csv('../data/' + read_file_corr_drew, index_col=0)

# create array of lirs
x = np.logspace(8, 14)
# calculate lirtd correlation lpeaks
dy = ah.lpeak_mmpz(x, dF.lam0.values, dF.eta.values)

plt.plot(x, dy, label='IRAS best fit')
plt.xscale('log')
plt.yscale('log')
plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter())
plt.gca().yaxis.set_minor_formatter(ticker.ScalarFormatter())
plt.scatter(10**fitF.measuredLIR, fitF.measuredLPeak, marker='.', color='k', alpha=0.7, label='schreiber+18')
plt.scatter(10**fitF_min_z.measuredLIR, fitF_min_z.measuredLPeak, marker='.', color='r', alpha=0.7, label='schreiber+18 min z')
plt.fill_between(x, 10**(np.log10(dy) - dF.gauss_width.values), 10**(np.log10(dy) + dF.gauss_width.values), color=sns.color_palette('mako')[1], alpha=0.3, label=r'$\pm$1$\sigma$ IRAS')
plt.fill_between(x, 10**(np.log10(dy) - 2*dF.gauss_width.values), 10**(np.log10(dy) + 2*dF.gauss_width.values), color=sns.color_palette('mako')[1], alpha=0.1, label=r'$\pm$2$\sigma$ IRAS')
plt.xlabel('LIR')
plt.ylabel(r'$\lambda_{peak}$')
plt.legend(fontsize='x-small')
plt.show()
