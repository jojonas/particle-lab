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

        self.min = float(min)
        self.max = float(max)
        self.nbins = int(nbins)
        
        self.histogram = np.zeros(self.nbins, dtype=float)
        self.bin_width = float(self.max-self.min)/self.nbins
        self.bin_edges = np.arange(self.nbins) * self.bin_width + self.min
                       
        self.underflow = 0.
        self.overflow = 0.

    def fill(self, value, weight=1.0):
        if value < self.min:
            self.underflow += weight
        elif value >= self.max:
            self.overflow += weight
        else:
            index = int(math.floor((value-self.min)/(self.max-self.min) * self.nbins))
            if index < 0 or index > self.nbins-1:
                print("Warning: invalid index %d!" % index)
            else:
                self.histogram[index] += weight

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
    def bin_centers(self):
        return self.bin_edges + self.bin_width/2

    def errorbar(self, fmt="s", **kwargs):
        with self._plot_context():
            values = self.histogram
            stat_errors = np.sqrt(values)
            return plt.errorbar(self.bin_centers[values!=0], values[values!=0], yerr=stat_errors[values!=0], fmt=fmt, **kwargs)
            
    def dots(self, **kwargs):
        with self._plot_context():
            return plt.plot(self.bin_centers, self.histogram, "s", **kwargs)
            
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

    @contextmanager
    def _plot_context(self):
        yield
        self.adjust_axes()
