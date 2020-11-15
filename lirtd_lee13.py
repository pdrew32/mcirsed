import numpy as np
import pandas as pd
import anahelper as ah
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import ticker


"""
plot lee+13 galaxies on the lirtd correlation from the iras sample. Marks galaxies removed in red. 
To add detection limit curves?
"""

write_file = True
write_file_plot = '../plots/lirtd_lee_detected.pdf' # '../plots/raw_lirtd_lee13.pdf'

# load fits
read_file_data = '../data/lee13_fit_detec_frac_added.pkl' # 
fitF = pd.read_pickle(read_file_data)

# load correlation params for iras
read_file_corr = 'lir_lpeak_corr_best_fit_all.csv' # 'lir_lpeak_corr_best_fit_drew.csv'
dF = pd.read_csv('../data/' + read_file_corr, index_col=0)

# create array of lirs
x = np.logspace(8, 14)
# calculate lirtd correlation lpeaks
dy = ah.lpeak_mmpz(x, dF.lam0.values, dF.eta.values)


detec_cutoff = 0.7
# get inds where fraction of galaxies detected is above the cutoff
detec_100_inds = fitF.detec_frac_100 > detec_cutoff
detec_160_inds = fitF.detec_frac_160 > detec_cutoff
detec_250_inds = fitF.detec_frac_250 > detec_cutoff
detec_350_inds = fitF.detec_frac_350 > detec_cutoff
detec_500_inds = fitF.detec_frac_500 > detec_cutoff

fitF.loc[(fitF.above_meas_100_limit == 1) & detec_100_inds, 'use_flag_100'] = 1
fitF.loc[(fitF.above_meas_160_limit == 1) & detec_160_inds, 'use_flag_160'] = 1
fitF.loc[(fitF.above_meas_250_limit == 1) & detec_250_inds, 'use_flag_250'] = 1
fitF.loc[(fitF.above_meas_350_limit == 1) & detec_350_inds, 'use_flag_350'] = 1
fitF.loc[(fitF.above_meas_500_limit == 1) & detec_500_inds, 'use_flag_500'] = 1

# sum the galaxies flagged as meeting our criteria
fitF['use_flag_sum'] = np.nansum([fitF.use_flag_100, fitF.use_flag_160, fitF.use_flag_250, fitF.use_flag_350, fitF.use_flag_500], axis=0)

# set up cut inds to plot. cut those without 5 sigma above xid error in 2 bands as well as those not above detec_frac cutoff in at least 2 bands
cut_inds = (fitF.sum_above_5_sigma_xid_error < 2) & (fitF.use_flag_sum < 2)

fig, ax = plt.subplots()
plt.plot(x, dy, label='best fit, all samples')
plt.xscale('log')
plt.yscale('log')
plt.title('Lee+13 galaxies with z_COSMOS2020', fontsize=12)
plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter())
plt.gca().yaxis.set_minor_formatter(ticker.ScalarFormatter())
plt.scatter(10**fitF.loc[cut_inds, 'measuredLIR'], fitF.loc[cut_inds, 'measuredLPeak'], c='r', marker='.', alpha=1, label='removed', zorder=99)
scat = plt.scatter(10**fitF.measuredLIR, fitF.measuredLPeak, c=fitF.z, marker='.', alpha=0.7)
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
