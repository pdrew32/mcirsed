## Readme
For each of the three samples from Drew and Casey 2022, IRAS, HATLAS, and COSMOS, there are two files. One, called [x]_sample_data.csv contains redshifts, flux densities, uncertainties on flux densities, and positions. The other file, called [x]_sample_fit_params.csv contains redshifts, IR luminosities, peak wavelengths, alpha, beta, dust temperature measurements, and their uncertainties. See below for a short description of each column. The IRAS_sample_fit_params.csv file has one additional column called sigma_clipped. This is 1 if the galaxy was sigma clipped during the fitting procedure described in section 4.1 of Drew and Casey 2022 or else it is nan.

### IRAS Sample Data Columns
z: redshift

IRAS_S12: Flux density at 12um in mJy

WISE_S12: Flux density at 12um in mJy

WISE_S22: Flux density at 22um in mJy

IRAS_S25: Flux density at 25um in mJy

IRAS_S60: Flux density at 60um in mJy

IRAS_S100: Flux density at 100um in mJy

HAT_S100: Flux density at 100um in mJy

HAT_S160: Flux density at 160um in mJy

HAT_S250: Flux density at 250um in mJy

HAT_S350: Flux density at 350um in mJy

HAT_S500: Flux density at 500um in mJy

IRAS_E12: Uncertainty in flux density at 12um in mJy

WISE_E12: Uncertainty in flux density at 12um in mJy

WISE_E22: Uncertainty in flux density at 22um in mJy

IRAS_E25: Uncertainty in flux density at 25um in mJy

IRAS_E60: Uncertainty in flux density at 60um in mJy

IRAS_E100: Uncertainty in flux density at 100um in mJy

HAT_E100: Uncertainty in flux density at 100um in mJy

HAT_E160: Uncertainty in flux density at 160um in mJy

HAT_E250: Uncertainty in flux density at 250um in mJy

HAT_E350: Uncertainty in flux density at 350um in mJy

HAT_E500: Uncertainty in flux density at 500um in mJy

RA_RIFSCz: RA from RIFSCz catalog

DEC_RIFSCz: Dec from RIFSCz catalog


### IRAS Sample Fit Parameter Columns
z: redshift

LIR_med: median LIR

LIR_16th: 16th percentile of LIR

LIR_84th: 84th percentile of LIR

LPeak_med: median peak Wavelength

LPeak_16th: 16th percentile of peak wavelength

LPeak_84th: 84th percentile of peak wavelength

alpha_med: median alpha parameter from equation 1 in Drew and Casey 2022

alpha_16th: 16th percentile of alpha

alpha_84th: 84th percentile of alpha

beta_med: median beta parameter from equation 1 in Drew and Casey 2022

beta_16th: 16th percentile of beta

beta_84th: 84th percentile of beta

tdust_med: median dust temperature paramter from equation 1 in Drew and Casey 2022

tdust_16th: 16th percentile of dust temperature

tdust_84th: 84th percentile of dust temperature

sigma_clipped: Whether or not the galaxy was sigma clipped out of the fitting procedure described in section 4.1 of Drew and Casey 2022


### COSMOS sample data columns

z: redshift

F24: 24um flux density in mJy

F70: 70um flux density in mJy

F100: 100um flux density in mJy

F160: 160um flux density in mJy

F250: 250um flux density in mJy

F350: 350um flux density in mJy

F500: 500um flux density in mJy

F850: 850um flux density in mJy

F1point1mm: 1100um flux density in mJy

F1point2mm: 1200um flux density in mJy

E24: Uncertainty in flux density at 24um in mJy

E70: Uncertainty in flux density at 70um in mJy

E100: Uncertainty in flux density at 100um in mJy

E160: Uncertainty in flux density at 160um in mJy

E250: Uncertainty in flux density at 250um in mJy

E350: Uncertainty in flux density at 350um in mJy

E500: Uncertainty in flux density at 500um in mJy

E850: Uncertainty in flux density at 850um in mJy

E1point1mm: Uncertainty in flux density at 1100um in mJy

E1point2mm: Uncertainty in flux density at 1200um in mJy

RA24: RA of 24um emission

DEC24: Dec of 24um emission


### COSMOS sample fit parameter columns
These are the same columns as IRAS Sample Fit Parameter Columns above, without the sigma_clipped column.

### HATLAS sample data columns

z: redshift

F22: flux density at 22um in mJy

F100: flux density at 100um in mJy

F160: flux density at 160um in mJy

F250: flux density at 250um in mJy

F350: flux density at 350um in mJy

F500: flux density at 500um in mJy

E22: Uncertainty in flux density at 22um in mJy

E100: Uncertainty in flux density at 100um in mJy

E160: Uncertainty in flux density at 160um in mJy

E250: Uncertainty in flux density at 250um in mJy

E350: Uncertainty in flux density at 350um in mJy

E500: Uncertainty in flux density at 500um in mJy

ra: HATLAS ra

dec: HATLAS dec


### HATLAS sample fit parameter columns
These are the same columns as IRAS Sample Fit Parameter Columns above, without the sigma_clipped column.
