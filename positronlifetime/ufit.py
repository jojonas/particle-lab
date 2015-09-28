import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from uncertainties import ufloat, unumpy as unp

def _derivative(func, x, *params, h=1e-6):
		return (-func(x+2*h, *params) + 8*func(x+h, *params) - 8*func(x-h, *params) + func(x-2*h, *params))/(12*h)
		
def _chi(params, func, xvalue, yvalue, xerror, yerror):
	difference = yvalue - func(xvalue, *params)
	error = np.sqrt( np.power(_derivative(func, xvalue, *params) * xerror, 2) + np.power(yerror, 2) )
	chi = difference / error
	return chi

def curve_fit(func, x, y, x0, B=1000, **kwargs):
	xerror = unp.std_devs(x)
	yerror = unp.std_devs(y)
	
	means = []
	for i in range(B):
		indices = np.random.randint(0, len(x), size=len(x))
		
		shifts1 = np.random.normal(loc=0, scale=1, size=len(x))
		x_simulated = unp.nominal_values(x) + xerror*shifts1
		
		shifts2 = np.random.normal(loc=0, scale=1, size=len(y))
		y_simulated = unp.nominal_values(y) + yerror*shifts2
			
		popt, pcov, infodict, mesg, ier = leastsq(
			_chi, 
			x0=tuple(x0),
			args=(func, x_simulated, y_simulated, xerror, yerror),
			full_output=True, 
			**kwargs
		)
		if ier in (1,2,3,4):
			means.append(popt)
	
	popt, pcov, infodict, mesg, ier = leastsq(
		_chi, 
		x0=tuple(x0),
		args=(func, unp.nominal_values(x), unp.nominal_values(y), xerror, yerror),
		full_output=True, 
		**kwargs
	)

	errors = np.std(means, axis=0)
	results = tuple(ufloat(a, b) for a, b in zip(popt, errors))
	
	chisqndof = np.power(_chi(popt, func, unp.nominal_values(x), unp.nominal_values(y), unp.std_devs(x), unp.std_devs(y)), 2).sum() / (len(x)-len(x0))
	
	return results, chisqndof
	
	
def plot_fit(x, y, func, params, xlabel="", ylabel=""):
	gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])
	
	xvalue = unp.nominal_values(x)
	yvalue = unp.nominal_values(y)
	xerror = unp.std_devs(x)
	yerror = unp.std_devs(y)
	
	X = np.linspace(xvalue.min(), xvalue.max(), 100)
	Y = func(X, *unp.nominal_values(params))
	Y1 = func(X, *(x.n + x.s for x in params))
	Y2 = func(X, *(x.n - x.s for x in params))
	
	residual = yvalue - func(xvalue, *unp.nominal_values(params))
	
	ax1 = plt.subplot(gs[0])
	ax1.errorbar(xvalue, yvalue, xerr=xerror, yerr=yerror, fmt='s', color="black")
	line, = ax1.plot(X, Y, "-", color="red")
	ax1.fill_between(X, np.minimum(Y1, Y2), np.maximum(Y1, Y2), color=line.get_color(), alpha=0.1)

	ax2 = plt.subplot(gs[1], sharex=ax1)
	combined_error = np.sqrt(np.power(xerror*_derivative(func, xvalue, *unp.nominal_values(params)),2) + np.power(yerror, 2))
	ax2.errorbar(xvalue, residual, yerr=combined_error, fmt='s', color="black")
	line = ax2.axhline(0, color="red")
	ax2.fill_between(X, np.minimum(Y1, Y2) - Y, np.maximum(Y1, Y2) - Y, color=line.get_color(), alpha=0.1)
	
	ax2.set_xlabel(xlabel)
	ax1.set_ylabel(ylabel)
	ax2.set_ylabel(r"$\Delta$" + ylabel)
	
	ax1.grid(True)
	ax2.grid(True)
	
	return ax1, ax2
	