import numpy as np
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
	
	
	hist.rebin(hist.nbins/4)
	hist.steps(color="black")
	#hist.steps(color="gray")
	
	seeds = np.array(seeds)
	times = np.array(times)
	
	peaks = np.zeros(len(seeds), dtype=object)
	for i, seed in enumerate(seeds):
		peak = peak_finder(hist, seed-1, step=3, background_fraction=0.02)
		center = peak.mean()
		width = find_68_interval(peak, center)
		peaks[i] = ufloat(center, width)
		plt.axvline(peaks[i].n, color="red")
		plt.axvline(peaks[i].n-peaks[i].s, color="red", alpha=0.1)
		plt.axvline(peaks[i].n+peaks[i].s, color="red", alpha=0.1)
	plt.xlabel("Channel")
	plt.ylabel("Number of Events")
	plt.show()
	
	plt.clf()
	linear = lambda x, a, b: a*x + b
	params, chisqndof = uncertainty_curve_fit(linear, peaks, times, x0=(0.002, -5), epsfcn=1e-5)
	
	ax1, ax2 = plot_fit(peaks, times, linear, params, xlabel="Channel", ylabel="Delay [ns]")
	
	text = ""
	text += r"$\chi^2/\mathrm{{ndof}} = {:.3g}$".format(chisqndof) + "\n"
	text += r"$\left({:.L}\right) x + \left({:.L}\right)$".format(*params)
	
	annotate(text, loc=2, ax=ax1)
	plt.show()
	
	average_width = ufloat(unp.std_devs(peaks).mean(), unp.std_devs(peaks).std())
	resolution = average_width * params[0] * 1000
	print("Resolution: {:.3P} ps".format(resolution))
	
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
	
	channel2time = lambda c: 0.0023293*c - 5.163
	
	for filename in filenames:
		hist = hist_from_tka(filename, frequencies=False)
		hist.normalize()
		hist.rebin(hist.nbins/4)			
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
	
def resolution_curve():	
	filename = "data/Co60_Fr9_11_resolution.TKA"
	channel2time = lambda c: 0.0023293*c - 5.163

	hist = hist_from_tka(filename, frequencies=False)
	hist.rebin(hist.nbins/4)
	
	hist.min = channel2time(hist.min)
	hist.max = channel2time(hist.max)
	peak = hist.slice(-0.55, 2.1)
	
	plt.clf()
	hist.steps(color="black")
	peak.steps(color="red")
	plt.xlabel("Delay [ns]")
	plt.ylabel("Number of Events")
	plt.yscale("log")
	plt.xlim(hist.min, hist.max)
	plt.ylim(np.min(hist.histogram[hist.histogram!=0]), np.max(hist.histogram))
	plt.show()
	
	centroid = peak.mean()
	print("PMT Delay:", centroid, "ns")
	
def deconvolution_test():
	channel2time = lambda c: 0.0023293*c - 5.163

	resolution_filename = "data/Co60_Fr9_11_resolution.TKA"
	resolution = hist_from_tka(resolution_filename, frequencies=False)
	resolution.rebin(resolution.nbins/20)
	resolution.normalize()
	resolution.min = channel2time(resolution.min)
	resolution.max = channel2time(resolution.max)
	#resolution = resolution.slice(-0.55, 2.1)
	
	signal_filename = "data/Co60_Fr1240_resolution_short.TKA"
	signal = hist_from_tka(signal_filename, frequencies=False)
	signal.rebin(signal.nbins/20)
	#signal.normalize()
	signal.min = channel2time(signal.min)
	signal.max = channel2time(signal.max)
	
	result, _ = deconvolve(signal.histogram, resolution.histogram)
	X = np.arange(len(result))
	#print(len(signal.histogram)-len(resolution.histogram), len(result))
	#plt.plot(X, result)
	#signal.steps(color="red")
	#resolution.steps(color="black")
	plt.plot(X, result)
	#plt.ylim(0, 100)
	plt.yscale("log")
	plt.show()
	
def lifetime_aluminum():
	filename = "data/Na22_aluminum_overnight.TKA"
	channel2time = lambda c: 0.0023293*c - 5.163
	
	hist = hist_from_tka(filename, frequencies=False)
	hist.rebin(hist.nbins/4)
	hist.min = channel2time(hist.min)
	hist.max = channel2time(hist.max)

	peak = hist.slice(-1.6, 8.9)
	
	plt.clf()
	hist.steps(color="black")
	peak.steps(color="red")
	plt.xlabel("Delay [ns]")
	plt.ylabel("Number of Events")
	plt.yscale("log")
	plt.xlim(hist.min, hist.max)
	plt.ylim(np.min(hist.histogram[hist.histogram!=0]), np.max(hist.histogram))
	plt.show()
	
	centroid = peak.mean()
	print("Aluminum Delay:", centroid, "ns")
	
if __name__=="__main__":
	#mca_calibration()
	all_cobalt()
	#sdeconvolution_test()
	#resolution_curve()
	#lifetime_aluminum()