from uncertainties import ufloat

r0 = ufloat(0.0305, 0.0005) / 2 # m
K = 8

ux_correction = ufloat(0.78, 0.02)
uy_correction = ufloat(0.76, 0.03)
uz_correction = ufloat(0.81, 0.03)
ug_correction = ufloat(0.61, 0.01)
uw_correction = ufloat(0.17, 0.03)

ux_error = 14
uy_error = 16
uz_error = 7
uw_error = 46
ug_error = 26

stability_uncertainty_air = 0.10
stability_uncertainty_vacuum = 0.10