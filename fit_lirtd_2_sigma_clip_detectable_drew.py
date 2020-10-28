import numpy as np
import pandas as pd
import anahelper as ah
import lirtd_corr_func as l
from matplotlib import pyplot as plt
from astropy.modeling import models, fitting
from scipy.odr import ODR, Model, Data, RealData

'''
Fit line to LIRTD correlation while sigma clipping
'''

read_file_fit = 'drew_fits_detec_frac_added.pkl'
fitF = pd.read_pickle('../data/' + read_file_fit)

# choose a detection limit fraction cutoff for 60 and 250
detec_cutoff = 0.95
fitF = fitF.loc[(fitF.detec_frac_60 > detec_cutoff) & (fitF.detec_frac_250 > detec_cutoff)]

# initialize a linear fitter
fit = fitting.LinearLSQFitter()
# fit the data with the fitter
line_init = models.Linear1D()
fitted_line = fit(line_init, fitF.measuredLIR.values, np.log10(fitF.measuredLPeak.values))

'''# fit the data with our custom model
cust_mod = l.lirtd_corr(1, eta=-0.068, lam0=100)
fitted_line = fit(cust_mod, fitF.measuredLIR.values, np.log10(fitF.measuredLPeak.values))'''

x = np.linspace(min(fitF.measuredLIR.values), max(fitF.measuredLIR.values))

# plot values and fit
plt.figure()
plt.subplot(211)
plt.errorbar(10**fitF.measuredLIR, fitF.measuredLPeak, fmt='o')
plt.plot(10**x, 10**fitted_line(x), 'k-', label='Fitted Model', linewidth=2, zorder=99)
plt.xlabel('log LIR')
plt.ylabel('log LPeak')
plt.xscale('log')
plt.yscale('log')
# plot with residuals 
data_minus_model = np.log10(fitF.measuredLPeak) - fitted_line(fitF.measuredLIR)
plt.subplot(212)
plt.scatter(10**fitF.measuredLIR, data_minus_model)
plt.axhline(0, linewidth=2, color='k')
plt.xscale('log')
plt.xlabel('log LIR')
plt.ylabel('log LPeak - model (microns)')
plt.show()


model = Model(ah.lineLIRTD)
# data = RealData(x[concatF.actual_outlier!=1], y[concatF.actual_outlier!=1], errX[concatF.actual_outlier!=1], errY[concatF.actual_outlier!=1])
data = RealData(x, y, errX, errY)
odr = ODR(data, model, beta0=[100.0, -0.06])
odr.set_job(fit_type=2)
output = odr.run()
print(output.beta)

fitter = fitting.LevMarLSQFitter()
model = models.Gaussian1D(mean=0.0, stddev=0.04)
hist = np.histogram(data_minus_model, bins=50)
binMeans = []
for i in list(range(len(hist[1])-1)):
    binMeans.append(np.mean([hist[1][i], hist[1][i+1]]))
fitted_model = fitter(model, x=binMeans, y=hist[0])
print(fitted_model)

# what sigma to clip at?
sigmaClip = 2.5 * fitted_model.stddev.value
removeInd = (data_minus_model < fitted_model.mean.value - sigmaClip) | (data_minus_model > fitted_model.mean.value + sigmaClip)
print(fitF.index[removeInd])
print(sum(removeInd))

# set up gaussian for plotting the fit
x2 = np.linspace(min(data_minus_model), max(data_minus_model))
y = fitted_model.amplitude.value * np.exp(-(x2-fitted_model.mean.value)**2.0/(2 * fitted_model.stddev.value**2.0))

# plot hist of residuals
plt.hist(data_minus_model, bins=50)
plt.plot(x2, y)
plt.xlabel('measured - model (microns)')
plt.ylabel('count')
plt.show()


# remove cut inds and do everything again
fitF_cut = fitF[~removeInd]

fit = fitting.LinearLSQFitter()
# fit the data with the fitter
line_init = models.Linear1D()
fitted_line = fit(line_init, fitF_cut.measuredLIR.values, np.log10(fitF_cut.measuredLPeak.values))

# plot values and fit
plt.figure()
plt.subplot(211)
plt.errorbar(10**fitF_cut.measuredLIR, fitF_cut.measuredLPeak, fmt='o')
plt.plot(10**x, 10**fitted_line(x), 'k-', label='Fitted Model', linewidth=2, zorder=99)
plt.xlabel('log LIR')
plt.ylabel('log LPeak')
plt.xscale('log')
plt.yscale('log')
# plot with residuals 
data_minus_model = np.log10(fitF_cut.measuredLPeak) - fitted_line(fitF_cut.measuredLIR)
plt.subplot(212)
plt.scatter(10**fitF_cut.measuredLIR, data_minus_model)
plt.axhline(0, linewidth=2, color='k')
plt.xscale('log')
plt.xlabel('log LIR')
plt.ylabel('log LPeak - model (microns)')
plt.show()

nbins = 25
fitter = fitting.LevMarLSQFitter()
model = models.Gaussian1D(mean=0.0, stddev=0.04)
hist = np.histogram(data_minus_model, bins=nbins)
binMeans = []
for i in list(range(len(hist[1])-1)):
    binMeans.append(np.mean([hist[1][i], hist[1][i+1]]))
fitted_model = fitter(model, x=binMeans, y=hist[0])
print(fitted_model)

x = np.linspace(min(data_minus_model), max(data_minus_model))
y = fitted_model.amplitude.value * np.exp(-(x-fitted_model.mean.value)**2.0/(2 * fitted_model.stddev.value**2.0))

# plot hist of residuals
plt.hist(data_minus_model, bins=nbins)
plt.plot(x, y)
plt.xlabel('measured - model (microns)')
plt.ylabel('count')
plt.show()


