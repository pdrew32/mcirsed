import numpy as np
import anahelper as ah
from scipy.stats import logistic
from astropy import constants as c
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo


"""
Fast versions of the functions in mccmcirsed (ones that don't use theano)
"""


def derivativeLogBB(Tdust, beta, w0):
    """Solve for the (approximate) derivatives of the BB function."""
    extra_fine_rest_wave = np.logspace(np.log10(20), np.log10(200), 1000)
    log_bb = np.log10(BB(10.0, Tdust, beta, w0, extra_fine_rest_wave))
    delta_y = log_bb[1:] - log_bb[:-1]
    delta_x = np.log10(extra_fine_rest_wave[1:]) - np.log10(extra_fine_rest_wave[:-1])
    return delta_y / delta_x


def eqWave(alpha, Tdust, beta, w0):
    """Compute the wavelength where the derivative of the log of BB equals the slope of the power law"""
    der_bb_reverse = derivativeLogBB(Tdust, beta, w0)[::-1]
    fineRestWave_reverse = np.logspace(np.log10(20), np.log10(200), 1000)[::-1]
    return fineRestWave_reverse[np.searchsorted(der_bb_reverse, alpha)]


def powerLaw(npl, restWave, alpha):
    """Equation of the power law portion of SED"""
    return npl * restWave**alpha


def sigmoid(x, a, b):
    """sigmoid function"""
    y = 1 / (1 + np.exp(-b*(x-a)))
    return y


def SnuNoBump(norm1, Tdust, alpha, beta, w0, restWave):
    """Combined MBB and Power Law functional form to fit with MCMC"""
    eq_w = eqWave(alpha, Tdust, beta, w0)
    bb = BB(norm1, Tdust, beta, w0, restWave)
    n = BB(norm1, Tdust, beta, w0, eq_w) * eq_w**-alpha
    pl = powerLaw(n, restWave, alpha)
    sig = sigmoid(restWave, eq_w, 200)
    return (1-sig) * pl + sig * bb


def lambdaPeak(norm1, Tdust, alpha, beta, w0):
    """Calculate wavelength where SED peaks"""
    return ah.h.xWa[np.argmax(SnuNoBump(norm1, Tdust, alpha, beta, w0, ah.h.xWa))]


def IRLum(norm1, Tdust, alpha, beta, w0, z, fourPiLumDistSquared):
    """Calculate LIR"""
    return np.sum(SnuNoBump(norm1, Tdust, alpha, beta, w0, ah.h.xWa)) * ah.h.deltaHz/(1+z) * fourPiLumDistSquared


def BB(nbb, Tdust, beta, w0, restWave):
    '''
    Modified blackbody function
    '''
    return 10**nbb * (1.0-np.exp(-(w0/restWave)**beta)) * restWave**(-3.0) / (np.exp(ah.h.hck/restWave/Tdust)-1.0)
