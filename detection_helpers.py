import numpy as np
import mcirsed_ff
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde


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