
import numpy as np
import helper_functions as ah
import theano as T
import theano.tensor as tt

"""
Core blackbody and SED calculations.
"""

def BB(nbb, Tdust, beta, w0, restWave):
    """Modified blackbody function

    Math takes the form:
    10**nbb * (1.0-np.exp(-(w0/restWave)**beta)) * restWave**(-3.0) / (np.exp(ah.h.hck/restWave/Tdust)-1.0)
    """
    a = tt.pow(10, nbb)
    b = tt.sub(1.0, tt.exp(-tt.pow(tt.true_div(w0, restWave), beta)))
    c = tt.pow(restWave, -3.0)
    d = tt.sub(tt.exp(tt.true_div(tt.true_div(ah.h.hck, restWave), Tdust)), 1.0)
    return tt.true_div(tt.mul(tt.mul(a, b), c), d)


def powerLaw(npl, restWave, alpha):
    """Equation of the power law portion of SED"""
    return tt.mul(npl, tt.pow(restWave, alpha))


def derivativeLogBB(Tdust, beta, w0):
    """Solves for the (approx) derivatives of the BB function."""
    extra_fine_rest_wave = np.logspace(np.log10(20), np.log10(200), 1000)
    log_bb = np.log10(BB(10.0, Tdust, beta, w0, extra_fine_rest_wave))
    delta_y = log_bb[1:] - log_bb[:-1]
    delta_x = np.log10(extra_fine_rest_wave[1:]) - np.log10(extra_fine_rest_wave[:-1])
    return delta_y / delta_x


def eqWave(alpha, Tdust, beta, w0):
    """Compute the wavelength where the derivative of the log of BB equals the slope of the power law"""
    der_bb_reverse = derivativeLogBB(Tdust, beta, w0)[::-1]
    # only search 20um to 200um because the eqWave is definitely between there
    extra_fine_rest_wave = T.shared(np.logspace(np.log10(20), np.log10(200), 1000)[::-1])
    return extra_fine_rest_wave[tt.extra_ops.searchsorted(der_bb_reverse, alpha)]


def SnuNoBump(norm1, Tdust, alpha, beta, w0, restWave):
    """Combined MBB and Power Law functional form to fit with MCMC
    For speed of computation, uses a sigmoid function with a sharp cutoff
    times BB and 1-sigmoid times pl to achieve the piecewise function.
    """
    eq_w = eqWave(alpha, Tdust, beta, w0)
    bb = BB(norm1, Tdust, beta, w0, restWave)
    n = BB(norm1, Tdust, beta, w0, eq_w) * eq_w**-alpha
    pl = powerLaw(n, restWave, alpha)
    sig = tt.nnet.sigmoid(200*(restWave-eq_w))
    return (1-sig) * pl + sig * bb