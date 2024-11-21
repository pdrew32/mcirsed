import numpy as np
from numpy.random import Generator, SFC64
from astropy.cosmology import Planck15 as cosmo
from scipy import interpolate


def IRLF(L, z=0.0):   
    """IRLF from Casey+18a

    Parameters:
    -----------
    L : ndarray
        luminosities to calculate number counts for
    
    z : float
        redshift to calculate the irlf at. defaults to 0.
    
    Returns:
    --------
    irlf_arr : ndarray
        number per Mpc^3 per dex
    """
    irlf_arr = np.zeros_like(L)
    lstar = 10**log_lstar(z)
    phi_star = 10**log_phi_star(z)
    alpha_lf = -0.6
    beta_lf = -3.0
    irlf_arr[L < lstar] = phi_star * (L[L < lstar]/lstar) ** alpha_lf
    irlf_arr[L >= lstar] = phi_star * (L[L >= lstar]/lstar) ** beta_lf
    return irlf_arr


def log_lstar(z, gamma_1=2.8, gamma_2=1.0, z_turn=1.95, z_w=2.0, l_not = 1.3e11):
    """equation 8 from Casey+18a describing the z evolution of lstar

    For z_turn, taking the average of the two models, A and B. 1.95 is average of 2.1 and 1.8
    """
    x = np.log10(1+z)
    x_t = np.log10(1 + z_turn)
    x_w = z_w/(np.log(10) * (1+z_turn))

    a = (gamma_2 - gamma_1)/2.0 * x_w/np.pi
    b = np.log(np.cosh((x - x_t) * np.pi/x_w))
    c = np.log(np.cosh(-x_t * np.pi/x_w))
    d = (gamma_2 + gamma_1)/2 * x
    e = np.log10(l_not)

    return a * (b - c) + d + e

def log_phi_star(z, phi_1=0.0, phi_2=-4.2, z_turn=1.95, z_w=2.0, phi_0=3.2e-4):
    """equation 9 from Casey+18a describing the z evolution of phi_star
    
    For phi_2, taking the average of the two models, A and B. -4.2 is average of -5.9 and -2.5
    """
    x = np.log10(1+z)
    x_t = np.log10(1 + z_turn)
    x_w = z_w/(np.log(10) * (1+z_turn))

    a = (phi_2 - phi_1)/2.0 * x_w/np.pi
    b = np.log(np.cosh((x - x_t) * np.pi/x_w))
    c = np.log(np.cosh(-x_t * np.pi/x_w))
    d = (phi_2 + phi_1)/2.0 * x
    e = np.log10(phi_0)

    return a * (b - c) + d + e


def generate_galaxies_from_irlf(z_min, z_max, survey_area, lirs):
    """returns log_irlf from casey+18 in z bin over survey area at provided lirs. also randomly chooses numbers of galaxies in each lir bin based on the probability of observing them
    
    Parameters:
    -----------
    z_min : float
        minimum redshift to create z array from
    z_max : float
        maximum redshift to create z array from
    survey_area : float
        survey area in deg^2
    lirs : array
        array of lirs to calculate over. units of solar luminosities, not log10(lsol). Use np.logspace so they are evenly spaced in log10

    Returns:
    --------
    log_sky_phiarr : numpy array
        the expected number of galaxies between z_min and z_max across the survey area for the given lirs. fractions allowed.
    ngals_per_lir : numpy array
        the rounded of galaxies between z_min and z_max across the survey area for the given lirs. integers only.
    """
    # use a high quality RNG
    rg = Generator(SFC64())

    # calculate volume of cosmos field over the desired redshifts
    # units of Mpc^3
    vol_z_upper = cosmo.comoving_volume(z_max).value * survey_area
    vol_z_lower = cosmo.comoving_volume(z_min).value * survey_area

    # ir luminosity function from casey+18a
    phiarr = IRLF(lirs, z=np.mean([z_max, z_min]))

    # get delta log lir
    delta_log_LIR = np.log10(lirs[1]) - np.log10(lirs[0])

    # convert irlf to units on the sky: sources/deg^2 in given zbin and lbin
    log_sky_phiarr = np.log10(phiarr) + np.log10(vol_z_upper - vol_z_lower) + np.log10(delta_log_LIR)
    ngals_per_lir = 10**log_sky_phiarr

    # need to work with ngals_per_lir as probabilities
    # round to get floor version
    ngals_per_lir_floor = np.floor(ngals_per_lir)
    # subtract to find remainder
    ngals_per_lir_remainder = np.abs(ngals_per_lir - ngals_per_lir_floor)

    # generate a random uniform number. if lower than the remainder of ngals, round up, otherwise round down
    rand = rg.uniform(0, 1, len(ngals_per_lir_remainder))
    ngals_per_lir = np.where(ngals_per_lir_remainder > rand, np.ceil(ngals_per_lir), np.floor(ngals_per_lir))

    return log_sky_phiarr, ngals_per_lir


def simulate_galaxies(lirs, ngals_per_lir, dF):
    """returns lir and lpeak of simulated galaxies
    
    Parameters:
    -----------
    lirs : numpy array or list
        lirs to populate the galaxies over
    ngals_per_lir : numpy array or list
        the number of galaxies in the given lir bin

    Returns:
    --------
    simlir : numpy array
        the log lirs of simulated galaxies
    simlpeak : numpy array
        the log lpeaks of simulated galaxies
    """
    rg = Generator(SFC64()) # use a high-quality rng

    # interpolate over the lir-lpeak correlation from the provided lirs and lir-lpeak correlation fit params
    lirtd_interp = interpolate.interp1d(np.log10(lirs), np.log10(lpeak_mmpz(lirs, dF.lam_t.values[0], dF.eta.values[0])))

    simlir = []
    simlpeak = []
    # for every lir bin, for the number of galaxies in that lir bin randomly draw from the measured distribution of galaxies around the lir-lpeak correlation provided in dF
    for i in list(range(len(lirs))):
        for j in list(range(int(ngals_per_lir[i]))):
            simlpeak.append(rg.normal(lirtd_interp(np.log10(lirs[i])), dF.width)[0])
            simlir.append(np.log10(lirs[i]))
    simlir = np.array(simlir)
    simlpeak = np.array(simlpeak)
    return simlir, simlpeak
