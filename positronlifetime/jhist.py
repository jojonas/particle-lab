import sys
import math
from contextlib import contextmanager

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

def round_signifcant_digits(x, digits=2, string=False):
	#sign = -1 if x < 0 else 1
	magnitude = -int(math.ceil(math.log(abs(x))/math.log(10)))
	total_digits = magnitude + digits
	number = round(x, total_digits)
	if not string:
		return number
	else:
		if x==0 or x > 1:
			fmt = "%%g"
		else:
			fmt = "%%.0%df" % total_digits
		return fmt % number

class Hist:
	def __init__(self, min, max, nbins):
		if min >= max:
			raise ValueError("min must be smaller than max")
		
		if nbins < 1:
			raise ValueError("Number of bins must be >= 1")
		

		self.min = float(min)
		self.max = float(max)
		self.nbins = int(nbins)
		
		self.histogram = np.zeros(self.nbins, dtype=float)
					   
		self.underflow = 0.
		self.overflow = 0.

	def fill(self, value, weight=1.0):
		if value < self.min:
			self.underflow += weight
		elif value >= self.max:
			self.overflow += weight
		else:
			index = self.bin_index(value)
			if index < 0 or index > self.nbins-1:
				print("Warning: invalid index %d!" % index)
			else:
				self.histogram[index] += weight
				
	def bin_index(self, value):
		return int(math.floor((value-self.min)/(self.max-self.min) * self.nbins))
		
	def bin_edge(self, index):
		return (self.max-self.min)*index/self.nbins + self.min
				
	def slice(self, lower, upper, index=False):
		if upper < lower:
			raise ValueError("Cannot slice if max < min")
			
		if not index:
			lower_index = self.bin_index(lower)
			upper_index = self.bin_index(upper)
		else:
			lower_index = lower
			upper_index = upper
			lower = self.bin_edge(lower)
			upper = self.bin_edge(upper)
			
		nbins = upper_index - lower_index
		hist = Hist(lower, upper, nbins)
		hist.histogram = self.histogram[lower_index:upper_index].copy()
		assert(hist.nbins == len(hist.histogram))
		return hist

	def rescale(self, factor):
		self.histogram *= factor
		self.overflow *= factor
		self.underflow *= factor

	def std(self):
		return math.sqrt(np.power(self.histogram*(self.bin_centers-self.mean()), 2).sum() / self.histogram.sum())

	def mean(self):
		return (self.histogram*self.bin_centers).sum() / self.histogram.sum()

	def count(self):
		return self.histogram.sum() + self.overflow + self.underflow

	@property
	def bin_width(self):
		return float(self.max-self.min)/self.nbins
		
	@property
	def bin_edges(self):
		return np.arange(self.nbins) * self.bin_width + self.min
		
	@property
	def bin_centers(self):
		return self.bin_edges + self.bin_width/2

	def errorbar(self, fmt="s", **kwargs):
		with self._plot_context():
			values = self.histogram
			stat_errors = np.sqrt(values)
			return plt.errorbar(self.bin_centers[values!=0], values[values!=0], yerr=stat_errors[values!=0], fmt=fmt, **kwargs)
			
	def dots(self, markers=".", **kwargs):
		with self._plot_context():
			return plt.plot(self.bin_centers, self.histogram, markers, **kwargs)
			
	def lines(self, **kwargs):
		with self._plot_context():
			X = np.concatenate(([self.min], self.bin_centers, [self.max]))
			Y = np.concatenate(([0], self.histogram, [0]))
			return plt.plot(X, Y, "-", **kwargs)
			
	def steps(self, **kwargs):
		with self._plot_context():
			edges = np.concatenate((self.bin_edges, [self.bin_edges[-1]+self.bin_width]))
			X = np.repeat(edges, 2)
			Y = np.concatenate(([0], np.repeat(self.histogram, 2), [0]))
			return plt.plot(X, Y, "-", **kwargs)
			
	def bars(self, **kwargs):
		with self._plot_context():
			return plt.bar(self.bin_edges, self.histogram, self.bin_width, **kwargs)

	def stats_box(self, **kwargs):
		n = "%d" % self.count()
		m = "%.1f" % self.mean() #round_signifcant_digits(self.mean(), 2, string=True)
		s = "%.1f" % self.std() #round_signifcant_digits(self.std(), 2, string=True)
		text = "$N = %s$\n$\\bar{x} = %s$\n$\\sigma = %s$" % (n, m, s)
		self.annotate(text, **kwargs)

	def annotate(self, text, loc=1, frameon=True, **kwargs):
		at = AnchoredText(text, loc=loc, frameon=frameon, prop=kwargs)
		plt.gca().add_artist(at)
		return at

	def adjust_axes(self):		  
		plt.xlim(self.min, self.max)
		plt.ylim(0, plt.ylim()[1])

	def normalize(self):
		factor = 1.0 / self.count()
		self.rescale(factor)
		
	def rebin(self, nbins):
		if nbins < 1:
			raise ValueError("Number of bins must be >= 1")
			
		factor = math.floor(self.nbins/nbins)
		new_nbins = int(self.nbins/factor)
		if new_nbins != int(nbins):
			print("Cannot rebin to %d bins, will use %d bins instead." % (nbins, new_nbins), file=sys.stderr)
			
		new_histogram = np.zeros(new_nbins, dtype=float)
		for i in range(new_nbins):
			new_histogram[i] = self.histogram[factor*i:factor*(i+1)].sum()
		overflow_bin_count = self.nbins - factor*new_nbins
		overflow_entry_count = self.histogram[factor*new_nbins:].sum()
		if overflow_entry_count != 0:
			print("Warning: %g entries in %d bins could not be accomodated while rebinning, they will be inserted into overflow." % (overflow_entry_count, overflow_bin_count), file=sys.stderr)
		old_sum = self.count()
		self.overflow += overflow_entry_count
		self.histogram = new_histogram
		self.nbins = new_nbins
		new_sum = self.count()
		if (old_sum - new_sum)/new_sum > 1e-5:
			raise ValueError("Rebinning went wrong. Now %g instead of %g entries." % (new_sum, old_sum))

	@contextmanager
	def _plot_context(self):
		yield
		self.adjust_axes()
