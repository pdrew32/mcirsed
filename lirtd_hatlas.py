import numpy as np
import pandas as pd
import anahelper as ah
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import ticker


"""
plot HATLAS DR1 galaxies on the lirtd correlation from the iras sample
"""

write_file = True
write_file_plot = '../plots/raw_lirtd_hatlas.pdf'

# load fits
read_file_data = '../data/hatlas_dr1_fits_portion_of_sky_best_params_added.pkl'
fitF = pd.read_pickle(read_file_data)

# load correlation params for iras
read_file_corr_drew = 'lir_lpeak_corr_best_fit_drew.csv'
dF = pd.read_csv('../data/' + read_file_corr_drew, index_col=0)

# create array of lirs
x = np.logspace(8, 14)
# calculate lirtd correlation lpeaks
dy = ah.lpeak_mmpz(x, dF.lam0.values, dF.eta.values)

fig, ax = plt.subplots()
plt.plot(x, dy, label='IRAS best fit')
plt.xscale('log')
plt.yscale('log')
plt.title('HATLAS DR1', fontsize=12)
plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter())
plt.gca().yaxis.set_minor_formatter(ticker.ScalarFormatter())
scat = plt.scatter(10**fitF.loc[fitF.measuredLIR > 7, 'measuredLIR'], fitF.loc[fitF.measuredLIR > 7, 'measuredLPeak'], c=fitF.loc[fitF.measuredLIR > 7, 'z'], marker='.', alpha=0.7) # color='k', 
cb = fig.colorbar(scat, ax=ax)
cb.set_label('z')
plt.fill_between(x, 10**(np.log10(dy) - dF.gauss_width.values), 10**(np.log10(dy) + dF.gauss_width.values), color=sns.color_palette('mako')[1], alpha=0.3, label=r'$\pm$1$\sigma$ IRAS')
plt.fill_between(x, 10**(np.log10(dy) - 2*dF.gauss_width.values), 10**(np.log10(dy) + 2*dF.gauss_width.values), color=sns.color_palette('mako')[1], alpha=0.1, label=r'$\pm$2$\sigma$ IRAS')
plt.xlabel('LIR')
plt.ylabel(r'$\lambda_{peak}$')
plt.legend(fontsize='x-small')
if write_file is True:
    plt.savefig(write_file_plot, dpi=300)
plt.show()
