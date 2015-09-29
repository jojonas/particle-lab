import math

import numpy as np

import matplotlib as mpl
mpl.rcParams['savefig.bbox'] = 'tight'
#mpl.rcParams['font.size'] = 20

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from scipy.optimize import curve_fit
from scipy.signal import deconvolve

from uncertainties import ufloat, umath
from uncertainties import unumpy as unp

from tka import TkaFile
from jhist import Hist
from ufit import curve_fit as uncertainty_curve_fit, plot_fit

def annotate(text, loc=1, ax=None, frameon=True, **kwargs):
    if not ax:
        ax = plt.gca()
    at = AnchoredText(text, loc=loc, frameon=frameon, prop=kwargs)
    ax.add_artist(at)
    return at
	
def hist_from_tka(filename, frequencies=False):
	data = TkaFile(filename)
	channel_count = len(data)
	hist = Hist(1, channel_count + 1, channel_count)
	hist.histogram = data.data.copy()
	if frequencies:
		hist.rescale(1/data.live_time)
	return hist
	
def peak_finder(hist, seed, step=5, background_fraction=0.1):
	data = hist.histogram
	seed_index = hist.bin_index(seed)
	while True:
		subset = data[seed_index-step:seed_index+step]
		new_seed_index = np.argmax(subset)+(seed_index-step)
		if new_seed_index == seed_index:
			break
		else:
			seed_index = new_seed_index
	
	threshold = data[seed_index] * background_fraction
	lower = seed_index 
	while lower > 0 and data[lower] > threshold:
		lower -= 1
	upper = seed_index
	while upper < len(data)-1 and data[upper] > threshold:
		upper += 1
	subset = hist.slice(lower, upper, index=True)
	return subset
	
def find_68_interval(hist, center):
	total = hist.count()
	center_index = hist.bin_index(center)
	width = 0
	sum = hist.histogram[center_index]
	goal = total * 0.68
	while sum < goal and center_index-width > 0 and center_index+width < hist.nbins:
		width += 1
		sum += hist.histogram[center_index-width] + hist.histogram[center_index+width]
	return width * hist.bin_width


def mca_calibration():
	filename = "data/Electronical_resolution_delay0_moredata.TKA"
	
	hist = hist_from_tka(filename, frequencies=False)
	
	seeds = (2250, 2644, 3050, 3484, 3926, 4373, 4820, 5157, 5594, 6004, 6445, 6942, 7369, 7785, 8223, 8678, 9122, 9561, 9976, 10410, 10740, 12920, 15061)
	times = (   0,    1,    2,    3,    4,    5,    6,    7,    8,    9,   10,   11,   12,   13,   14,   15,   16,   17,   18,    19,    20,    25,    30)
	
	
	#hist.rebin(hist.nbins/4)
	
	seeds = np.array(seeds)
	times = np.array(times)
	
	peaks = np.zeros(len(seeds), dtype=object)
	
	plt.figure(figsize=(12*math.sqrt(2), 12))
	for i, seed in enumerate(seeds):
		peak = peak_finder(hist, seed-1, step=3, background_fraction=0.02)
		peak.normalize()
		peak.steps(color="black")
		center = peak.centroid()
		width = find_68_interval(peak, center)
		peaks[i] = ufloat(center, width)
		y = np.max(peak.histogram)
		plt.gca().annotate(
			"%d ns" % times[i], 
			xy=(center, y), 
			xytext=(center, y), 
			xycoords="data", 
			textcoords="data",
			horizontalalignment='center', 
			verticalalignment='bottom',
		)
		#plt.axvline(peaks[i].n, color="red")
		#plt.axvline(peaks[i].n-peaks[i].s, color="red", alpha=0.1)
		#plt.axvline(peaks[i].n+peaks[i].s, color="red", alpha=0.1)
	plt.xlabel("Channel")
	plt.ylabel("Normalized number of Events")
	plt.yscale("log")
	plt.xlim(2000, 16e3)
	plt.ylim(0.002, 0.04)
	plt.savefig("images/calibration_peaks.pdf", fontsize=10)
	
	plt.clf()
	plt.figure(figsize=(8*math.sqrt(2), 8))
	
	linear = lambda x, a, b: a*x + b
	params, chisqndof = uncertainty_curve_fit(linear, peaks, times, x0=(0.002, -5), epsfcn=1e-5, B=1000)
	
	ax1, ax2 = plot_fit(peaks, times, linear, params, xlabel="Channel", ylabel="Delay [ns]", range=(2000, 16000))
	
	text = ""
	text += r"$\chi^2/\mathrm{{ndof}} = {:.3g}$".format(chisqndof) + "\n"
	text += r"$\left({:.L}\right) x + \left({:.L}\right)$".format(*params)
	
	annotate(text, loc=2, ax=ax1)
	plt.savefig("images/calibration.pdf")
	
	average_width = ufloat(unp.std_devs(peaks).mean(), unp.std_devs(peaks).std())
	resolution = average_width * params[0] * 1000
	print("Resolution: {:.3P} ps".format(resolution))
	
def windows():
	filenames = [
		"data/Na22_antikoin_start_.tka", 
		"data/Na22_antikoin_stop_.tka", 
	]
	ranges = [(6000, 16000), (0, 8000)]
	
	for i, filename in enumerate(filenames):
		hist = hist_from_tka(filename)
		hist.rebin(hist.nbins/160)
		filtered = hist.slice(int(ranges[i][0]/160), int(ranges[i][1]/160), index=True)
		
		filtered.steps(fill=True, color="red", hatch="/", facecolor="none")
		hist.steps(label=("Start circuit", "Stop circuit")[i], color="black")
		
		plt.xlabel("Channel (Energy)")
		plt.ylabel("Number of Events")
		plt.yscale("log")
		plt.ylim(1, hist.histogram.max()*1.2)
		plt.legend()
		plt.savefig("images/windows_%d.pdf" % i)
		plt.clf()
	
def all_cobalt():
	filenames = [
		#"data/Co60_1900_16nsDelay_.TKA",
		#"data/Co60_Fr9_11_resolution.TKA",
		#"data/Co60_Fr1240_resolution_short.TKA",
		#"data/Co60_Mo0945.TKA",
		"data/Co60_duenn_links.TKA",
		"data/Co60_duenn_rechts.TKA",
		"data/Co60_delay_vom_anfang.TKA",
	]
	
	channel2time = lambda c: 0.002329*c - 5.16
	
	for filename in filenames:
		hist = hist_from_tka(filename, frequencies=False)
		hist.normalize()
		hist.rebin(500)			
		hist.min = channel2time(hist.min)
		hist.max = channel2time(hist.max)
		hist.steps(label=filename)
		print(filename, "Centroid:", hist.mean())
		
	plt.legend()
	plt.xlabel("Delay [ns]")
	plt.ylabel("Number of Events")
	#plt.yscale("log")
	plt.xlim(hist.min, hist.max)
	#plt.ylim(np.min(hist.histogram[hist.histogram!=0]), np.max(hist.histogram))
	plt.show()
	
def resolution_and_aluminum_lifetime():	
	signal_filename = "data/Na22_aluminum_overnight.TKA"
	resolution_filename = "data/Co60_Fr9_11_resolution.TKA"
	
	channel2time = lambda c: 0.002329*c - 5.16

	resolution_hist = hist_from_tka(resolution_filename, frequencies=False)
	resolution_hist.rebin(500)
	resolution_hist.min = channel2time(resolution_hist.min)
	resolution_hist.max = channel2time(resolution_hist.max)
	
	plt.clf()
	resolution_hist.steps(color="black")
	resolution_background = resolution_hist.slice(20, 30)
	resolution_background.steps(color="blue")
	resolution_background_level = np.mean(resolution_background.histogram)
	plt.axhline(resolution_background_level, color="blue")
	plt.xlabel("Delay [ns]")
	plt.ylabel("Number of Events")
	plt.yscale("log")
	plt.xlim(resolution_hist.min, resolution_hist.max)
	plt.ylim(200, 1e5)	
	plt.savefig("images/resolution_background.pdf")
	
	plt.clf()
	resolution_hist.histogram = np.maximum(0, resolution_hist.histogram-resolution_background_level)
	resolution_hist.steps(color="black")
	resolution_peak = resolution_hist.slice(-0.55, 2.1)
	resolution_peak.errorbar(fmt=",", color="red")
	
	gauss = lambda x, mu, sigma, N: N*np.exp(-np.power(x-mu, 2)/np.power(sigma, 2))
	popt, pcov = curve_fit(gauss, resolution_peak.bin_centers, resolution_peak.histogram, sigma=np.sqrt(resolution_peak.histogram), p0=(0.54, 0.5, 1e5), absolute_sigma=True)
	X = np.linspace(resolution_peak.min, resolution_peak.max, 1000)
	plt.plot(X, gauss(X, *popt), color="blue")
	params = list(ufloat(v, e) for v, e in zip(popt, np.sqrt(np.diag(pcov))))
	print("Gauss fit:", params)
	
	chisq = np.power(resolution_peak.histogram - gauss(resolution_peak.bin_centers, *popt), 2).sum() / resolution_peak.histogram.sum() 
	chisqndof = chisq / (resolution_peak.nbins - len(popt))
	
	text = ""
	text += r"$\chi^2$/ndof = " + "%.3g\n" % chisqndof
	text += r"$\mu$ = " + "%.3g ps, " % (1000*popt[0]) + r"$\sigma$ = " + "%.3g ps" % (1000*popt[1])
	
	annotate(text)
	
	plt.xlabel("Delay [ns]")
	plt.ylabel("Number of Events")
	plt.yscale("log")
	plt.xlim(-2, 5)
	plt.ylim(2, 1e5)	
	plt.savefig("images/resolution_peak.pdf")
	
	resolution_centroid = resolution_peak.centroid()
	
	
	signal_hist = hist_from_tka(signal_filename, frequencies=False)
	signal_hist.rebin(500)
	signal_hist.min = channel2time(signal_hist.min)
	signal_hist.max = channel2time(signal_hist.max)
	
	signal_background = signal_hist.slice(20, 30)
	signal_background_level = np.mean(signal_background.histogram)
	signal_hist.histogram = np.maximum(0, signal_hist.histogram-signal_background_level)
	
	signal_hist.normalize()
	resolution_hist.normalize()
	
	signal_peak = signal_hist.slice(-5, 12)
	signal_centroid = signal_peak.centroid()
	
	resolution_ucentroid = ufloat(resolution_peak.centroid(), resolution_peak.centroid_error())
	signal_ucentroid = ufloat(signal_peak.centroid(), signal_peak.centroid_error())
	
	print("PMT delay:", resolution_ucentroid, "ns")
	print("Aluminium delay:", signal_ucentroid, "ns")
	print("Aluminium lifetime:", signal_ucentroid - resolution_ucentroid, "ns")
	
	plt.clf()
	plt.gca().annotate(
		"$^{60}$Co, %.3g ns" % resolution_centroid, 
		xy=(resolution_centroid-0.4, 0.1), 
		xytext=(resolution_centroid-0.4, 0.1), 
		xycoords="data", 
		textcoords="data",
		horizontalalignment='right', 
		verticalalignment='bottom',
		color="blue"
	)
	plt.gca().annotate(
		"$^{22}$Na in Al, %.4g ns" % signal_centroid, 
		xy=(signal_centroid+0.4, 0.1), 
		xytext=(signal_centroid+0.4, 0.1), 
		xycoords="data", 
		textcoords="data",
		horizontalalignment='left', 
		verticalalignment='bottom',
		color="red",
	)
	
	#resolution_hist.steps(color="gray")
	resolution_peak = resolution_hist.slice(-0.55, 2.1)
	resolution_peak.steps(color="blue")
	
	signal_hist.steps(color="black")
	signal_peak.steps(color="red")
	plt.axvline(signal_centroid, color="red")
	
	plt.axvline(resolution_centroid, color="blue")
	
	plt.xlabel("Delay [ns]")
	plt.ylabel("Normalized Number of Events")
	plt.yscale("log")
	plt.xlim(-5, 20)
	plt.ylim(0.00001, 0.2)
	
	
	plt.savefig("images/na22_aluminum.pdf")
	
	
def lifetime_polyethylen():
	signal_filename = "data/Na22_plastik.TKA"
	resolution_filename = "data/Co60_Fr9_11_resolution.TKA"
	
	channel2time = lambda c: 0.002329*c - 5.16
	
	signal_hist = hist_from_tka(signal_filename, frequencies=False)
	signal_hist.rebin(500)
	signal_hist.min = channel2time(signal_hist.min)
	signal_hist.max = channel2time(signal_hist.max)
	
	signal_background = signal_hist.slice(20, 30)
	signal_background_level = np.mean(signal_background.histogram)
	signal_hist.histogram = np.maximum(0, signal_hist.histogram-signal_background_level)
	
	peak = (1.4, 12)
	
	N = signal_hist.count()
	signal_peak = signal_hist.slice(*peak) # for unnormalized errrors
	errors = np.sqrt(signal_peak.histogram) / N
	signal_hist.normalize()
	
	signal_peak = signal_hist.slice(*peak) # for real
	
	
	resolution_hist = hist_from_tka(resolution_filename, frequencies=False)
	resolution_hist.rebin(500)
	resolution_hist.min = channel2time(resolution_hist.min)
	resolution_hist.max = channel2time(resolution_hist.max)
	resolution_background = resolution_hist.slice(20, 30)
	resolution_background.steps(color="blue")
	resolution_background_level = np.mean(resolution_background.histogram)
	resolution_hist.histogram = np.maximum(0, resolution_hist.histogram-resolution_background_level)
	resolution_hist.normalize()

	f = lambda x, N1, N2, tau1, tau2: N1*np.exp(-x/tau1) + N2*np.exp(-x/tau2)
	popt = (0.8, 0.005, 1/2.5, 1/0.5)
	popt, pcov = curve_fit(f, signal_peak.bin_centers, signal_peak.histogram, p0=popt, sigma=errors, absolute_sigma=True)
	X = np.linspace(signal_peak.min, signal_peak.max, 1000)
	
	chisq = np.power(signal_peak.histogram - f(signal_peak.bin_centers, *popt), 2).sum() / np.power(errors, 2).sum() 
	chisqndof = chisq / (signal_peak.nbins - len(popt))
	
	params = list(abs(e/v)*100 for v, e in zip(popt, np.sqrt(np.diag(pcov))))
	print("Fit:", params)
	
	plt.clf()
	text = ""
	text += r"$\chi^2$/ndof = " + "%.3g\n" % chisqndof
	text += r"$N_1 / N2$ = " + "%.3g\n" % (popt[0]/popt[1])
	text += r"$\tau_1$ = " + "%.3g ps, " % (1000*popt[2]) + r"$\tau_2$ = " + "%.4g ps" % (1000*popt[3])
	
	annotate(text)
	
	#resolution_hist.steps(color="gray")
	resolution_peak = resolution_hist.slice(-0.55, 2.1)
	resolution_peak.steps(color="blue")
	signal_hist.steps(color="gray")
	signal_peak.errorbar(errors=errors, fmt=",", color="red")
	plt.plot(X, f(X, *popt), color="black", linewidth=3)
	plt.xlabel("Delay [ns]")
	plt.ylabel("Normalized number of Events")
	plt.yscale("log")
	plt.xlim(signal_hist.min, signal_hist.max)
	plt.ylim(0.000001, 0.2)
	plt.savefig("images/na22_polyethylen.pdf")
	
	#centroid = peak.centroid()
	#print("Aluminum Delay:", centroid, "ns")
	
if __name__=="__main__":
	#windows()
	#mca_calibration()
	#all_cobalt()
	#sdeconvolution_test()
	#resolution_and_aluminum_lifetime()
	lifetime_polyethylen()