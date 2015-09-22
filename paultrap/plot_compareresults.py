import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText


air_m1=np.array([1.,1.2,1.3])
err_air_m1=np.array([1.,1.2,1.3])/10.
air_m2=np.array([1.1,1.1,1.2])
err_air_m2=np.array([1.,1.2,1.3])/10.
air_m3=np.array([0.9,1.3,1.5])
err_air_m3=np.array([1.,1.2,1.3])/10.
air_ns=np.array([n+1 for n in range(len(air_m1))])
print air_ns

vac_m1=np.array([1.,1.2,1.3])
err_vac_m1=np.array([1.,1.2,1.3])/10.
vac_m2=np.array([1.1,1.1,1.2])
err_vac_m2=np.array([1.,1.2,1.3])/10.
vac_m3=np.array([0.9,1.3,1.5])
err_vac_m3=np.array([1.,1.2,1.3])/10.
vac_ns=np.array([n+1 for n in range(len(vac_m1))])
print vac_ns

h,axes=plt.subplots(1,2,sharey=True)
#axes[0].grid(True)
axes[0].errorbar(air_ns,air_m1,yerr=err_air_m1,fmt='^',label="Stability Study")
axes[0].errorbar(air_ns,air_m2,yerr=err_air_m2,fmt='o',label="Z-Compensation")
axes[0].errorbar(air_ns,air_m3,yerr=err_air_m3,fmt='p',label="Resonance Study")

axes[0].set_ylabel(r'Specific Charge $[\frac{\mu C}{kg}]$', fontsize=13)
axes[0].set_xlabel(r'Particle',fontsize=13)
axes[0].set_xticks(air_ns)
axes[0].set_xticklabels(air_ns)
axes[0].set_xlim(air_ns[0]-0.5,air_ns[-1]+0.5,)
#axes[0].set_ylim(min(air_m1[0],air_m2[0],air_m3[0])-30,max(air_m1[-1],air_m2[-1],air_m3[-1])+50)
axes[0].set_yscale('log')
axes[0].grid(axis="y")


axes[1].errorbar(vac_ns,vac_m1,yerr=err_vac_m1,fmt='^',label="Stability Study")
axes[1].errorbar(vac_ns,vac_m2,yerr=err_vac_m2,fmt='o',label="Z-Compensation")
axes[1].errorbar(vac_ns,vac_m3,yerr=err_vac_m3,fmt='p',label="Resonance Study")

axes[1].set_xlabel(r'Particle',fontsize=13)
axes[1].set_xticks(vac_ns)
axes[1].set_xticklabels(vac_ns)
axes[1].set_xlim(vac_ns[0]-0.5,vac_ns[-1]+0.5,)
#axes[1].set_ylim(min(vac_m1[0],vac_m2[0],vac_m3[0])-10,max(vac_m1[-1],vac_m2[-1],vac_m3[-1])+50)
axes[1].set_yscale('log')
axes[1].grid(axis="y")

plt.legend()
plt.show()

h.savefig('images/paul_compareresults.pdf')
