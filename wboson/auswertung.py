from __future__ import print_function
import os, os.path
import sys
import contextlib
import math
import random
#random.seed(33)

import ROOT
import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat, umath, correlated_values_norm
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

try:
    from scipy.optimize import curve_fit
except ImportError:
    from myoptimize import curve_fit


from rootwrapper import *
from jhist import Hist
import jmpl

def weight_to_mass(i): 
    return [
        79.9446,
        79.9946,
        80.0446,
        80.0946,
        80.1446,
        80.1946,
        80.2446,
        80.2946,
        80.3446,
        80.3946,
        80.4446,
        80.4946,
        80.5446,
        80.5946,
        80.6446,
        80.6946,
        80.7446,
        80.7946,
        80.8446
    ][i]

plot_colors = [
            "#396ab1",
            "#da7c30",
            "#3e9651",
            "#cc2529",
            "#535154",
            "#6b4c9a",
            "#922428",
            "#948b3d",
        ]

plot_linestyles = [
            "-",
            "--",
            "..",
            "-.",
        ]

def annotate(text, loc=1, frameon=True, **kwargs):
    at = AnchoredText(text, loc=loc, frameon=frameon, prop=kwargs)
    plt.gca().add_artist(at)
    return at

class MyEvent(object):
    def __init__(self, event, smear_energy=False, scale_energy=None):
        self._event = event

        pT = math.sqrt(self._event.el_px**2 + self._event.el_py**2)
        p = math.sqrt(self._event.el_px**2 + self._event.el_py**2 + self._event.el_pz**2)
        E = self.el_e
        self.el_e_t = pT/p * E

        self.met_e_t = math.sqrt(self.metx_calo**2 + self.mety_calo**2)

        if smear_energy:
            self.el_e_t = self.dice_energy_resolution(self.el_e_t)

        if scale_energy == "up":
            self.el_e_t += self.energy_scale(self.el_e_t)
        elif scale_energy == "down":
            self.el_e_t -= self.energy_scale(self.el_e_t)
        elif scale_energy is not None:
            raise ValueError("Unknown scale direction '%s'." % scale_energy)

        self.met_phi = math.atan2(self.metx_calo, self.mety_calo) + math.pi
        self.delta_z = self.el_track_z - self.met_vertex_z
        self.m_t = math.sqrt(self.el_e_t*self.met_e_t*(1-math.cos(self.el_met_calo_dphi)))

    def __getattr__(self, name):
        return getattr(self._event, name)
        
    def is_mc(self):
        return hasattr(self._event, "weight")

    def is_data(self):
        return not self.is_mc()

    def get_weight(self, i):
        if self.is_mc():
            return self._event.weight[i]
        else:
            return 1.0

    def passes_cuts(self):
        return (self.met_e_t > 20
            and self.el_e_t > 30
            and self.el_iso < 0.03 
            and self.el_met_calo_dphi > 2.85
            and abs(self.delta_z) < 0.2)

    @staticmethod
    def dice_energy_resolution(E):
        sigma = 0.015 * E + 0.13*math.sqrt(E) + 0.4
        return max(0, random.gauss(E, sigma))

    @staticmethod
    def energy_scale(E):
        # 0.0018 / 0.9514 ~ 0.2%
        return 0.002*E + 0.015


def task_1():    
    with root_open("data/mc_all.root") as mc_file:
        mc_events = mc_file.MCTree
        N_events = mc_events.GetEntries()

        interesting_weights = (0, 9, 18)

        hists = []
        for i in interesting_weights:
            hists.append( Hist(70, 90, 100) )
            
        for event_number, event in enumerate(mc_events):
            if event_number % 1000 == 0:
                print("\rProgress: %.1f %%" % (100*float(event_number)/float(N_events)), end="")
                sys.stdout.flush()
                
            for i, weight in enumerate(interesting_weights):
                hists[i].fill(event.mc_w_m, weight=event.weight[weight])

        plt.clf()
        for i, weight in enumerate(interesting_weights):
            hists[i].steps(label="m = %g GeV" % weight_to_mass(weight), color=plot_colors[i], ls=["--", "-", "-."][i], linewidth=1)
            #hists[i].stats_box(loc=3)
            
        plt.xlabel("Generated W Mass / GeV")
        plt.ylabel("Number of Events")
        plt.legend()
        plt.ylim(0, hists[0].histogram.max()*1.4)
        plt.savefig("images/mc_examples.pdf")

def task_2():    
    with contextlib.nested(root_open("data/mc_all.root"), root_open("data/d0.root")) as (mc_file, data_file):
        # NOTE: determine cuts on the "standard" weight, 9
        
        data_lumi = 198.0 # +- 20, pb^-1
        process_xsec = 2.58e3 # +- 0.09e3 pb
        
        magic_factor = 0.90 # given in manual
        mc_gen_count = 1164699 # also given in manual
        lumi_scale = data_lumi * process_xsec / mc_gen_count * magic_factor

        N_mc_tau_events = 33705

        all_hists = {
            "mc": {
                "E_T_el": Hist(0, 100, 100),
                "E_T_miss": Hist(0, 100, 100),
                "m_T": Hist(0, 100, 100),
                "eta_el": Hist(-1.5, 1.5, 100),
                "phi_el": Hist(0, 2*math.pi, 100),
                "phi_miss": Hist(0, 2*math.pi, 100),
                "delta_phi": Hist(0, 1*math.pi, 100),
                "delta_z": Hist(-1, 1, 100),
                "el_iso": Hist(0, 0.1, 100),
            },
            "data": {
                "E_T_el": Hist(0, 100, 100),
                "E_T_miss": Hist(0, 100, 100),
                "m_T": Hist(0, 100, 100),
                "eta_el": Hist(-1.5, 1.5, 100),
                "phi_el": Hist(0, 2*math.pi, 100),
                "phi_miss": Hist(0, 2*math.pi, 100),
                "delta_phi": Hist(0, 1*math.pi, 100),
                "delta_z": Hist(-1, 1, 100),
                "el_iso": Hist(0, 0.1, 100),
            },
            "mc_tau": {
                "E_T_el": Hist(0, 100, 100),
                "E_T_miss": Hist(0, 100, 100),
                "m_T": Hist(0, 100, 100),
                "eta_el": Hist(-1.5, 1.5, 100),
                "phi_el": Hist(0, 2*math.pi, 100),
                "phi_miss": Hist(0, 2*math.pi, 100),
                "delta_phi": Hist(0, 1*math.pi, 100),
                "delta_z": Hist(-1, 1, 100),
                "el_iso": Hist(0, 0.1, 100),
            }
        }
        
        for name, tree in zip(
            ("mc", "data", "mc_tau"),
            (mc_file.MCTree, data_file.MessTree, mc_file.MCTree)
            ):

            N_events = tree.GetEntries() 
            hists = all_hists[name]

            if name == "mc_tau":
                first_event = N_events - N_mc_tau_events
            else:
                first_event = 0
                
            for event_number, root_event in enumerate(tree):
                if event_number < first_event:
                    continue

                #if event_number % 100 != 0:
                #    continue
                                   
                if event_number % 1000 == 0:
                    print("\r", name, "- progress: %.1f %%           " % (100*float(event_number)/float(N_events)), end="")
                    sys.stdout.flush()

                event = MyEvent(root_event)
                
                #if not event.passes_cuts():
                #    continue
                
                weight = event.get_weight(9)
                hists["E_T_el"].fill(event.el_e_t, weight)
                hists["E_T_miss"].fill(event.met_e_t, weight)
                hists["m_T"].fill(event.m_t, weight)
                hists["eta_el"].fill(event.el_eta, weight)
                hists["phi_el"].fill(event.el_phi, weight)
                hists["phi_miss"].fill(event.met_phi, weight)
                hists["delta_phi"].fill(event.el_met_calo_dphi, weight)
                hists["delta_z"].fill(event.delta_z, weight)
                hists["el_iso"].fill(event.el_iso, weight)

        xlabels = {
            "E_T_el": "Electron Transverse Energy / GeV",
            "E_T_miss": "Missing Transverse Energy / GeV",
            "m_T": "Transverse Mass / GeV",
            "eta_el": "Electron Eta",
            "phi_el": "Electron Phi / rad",
            "phi_miss": "MET Phi / rad",
            "delta_phi": "Delta Phi / rad",
            "delta_z": "Delta z / mm",
            "el_iso": "Isolation",
        }

        upper_cuts = {
            "E_T_el": 30,
            "E_T_miss": 20,
            "delta_phi": 2.85,
            "delta_z": -0.2,
        }
        
        lower_cuts = {
            "el_iso": 0.03,
            "delta_z": 0.2,
        }

        draw_cuts = False

        for quantity in all_hists["data"].keys():
            plt.clf()

            data_hist = all_hists["data"][quantity]
            data_hist.errorbar(label="Data", color="black", fmt=".")
            
            mc_hist = all_hists["mc"][quantity]

            Nmc = mc_hist.count()
            
            mc_hist.rescale(lumi_scale)
            mc_hist.steps(label="MC (all)", color=plot_colors[0], linestyle="-")

            tau_hist = all_hists["mc_tau"][quantity]
            tau_hist.rescale(lumi_scale)
            tau_hist.steps(label="MC (tau only)", color=plot_colors[0], linestyle="-.")

            text = "Data: %d\nMC: %d\nMC raw: %d" % (data_hist.count(), mc_hist.count(), Nmc)
            data_hist.annotate(text, loc=2)

            if draw_cuts:
                if quantity in lower_cuts:
                    cut = lower_cuts[quantity]
                    plt.axvspan(cut, plt.xlim()[1], hatch="/", color="red", fill=False)
                    
                if quantity in upper_cuts:
                    cut = upper_cuts[quantity]
                    plt.axvspan(plt.xlim()[0], cut, hatch="\\", color="red", fill=False)
                
            plt.xlabel(xlabels[quantity])
            plt.ylabel("Number of Events")
            plt.legend()
            plt.savefig("images2/%s.pdf" % quantity)
                
def task_3():
    # CACHE
    mc_allcuts = 74072. 
    data = 67329.

    process_xsec = ufloat(2.58e3, 0.09e3)
    n_data = ufloat(data, math.sqrt(data))
    xsec = n_data / mc_allcuts * process_xsec

    print("XSec: {xsec:.2f} nb".format(xsec=xsec*1e-3))

def task_4():
    with contextlib.nested(root_open("data/mc_all.root"), root_open("data/d0.root")) as (mc_file, data_file):
        mc_hist_m_t = list( Hist(40, 70, 50) for _ in range(19) )
        mc_hist_el_e_t = list( Hist(30, 70, 50) for _ in range(19) )
        data_hist_m_t = Hist(40, 70, 50)
        data_hist_el_e_t = Hist(30, 70, 50)

        for name, tree in zip(("mc", "data"), (mc_file.MCTree, data_file.MessTree)):
            N_events = tree.GetEntries()
            for event_number, root_event in enumerate(tree):
                #if event_number % 50 != 0:
                #    continue
                    
                if event_number % 1000 == 0:
                    print("\rProgress: %.1f %%" % (100*float(event_number)/float(N_events)), end="")
                    sys.stdout.flush()

                event = MyEvent(root_event, scale_energy="down")
                
                if not event.passes_cuts():
                    continue

                if event.is_mc():
                    for i, weight in enumerate(event.weight):
                        mc_hist_m_t[i].fill(event.m_t, weight)
                        mc_hist_el_e_t[i].fill(event.el_e_t, weight)
                else:
                    data_hist_m_t.fill(event.m_t)
                    data_hist_el_e_t.fill(event.el_e_t)


    for name, mc_hist, data_hist in zip(("m_t", "el_e_t"), (mc_hist_m_t, mc_hist_el_e_t), (data_hist_m_t, data_hist_el_e_t)):
        X = np.zeros(19)
        Y = np.zeros(19)
        
        plt.clf()
        for weight_index in range(19):

            factor = data_hist.count() / mc_hist[weight_index].count()
            mc_hist[weight_index].rescale(factor)

            assert(mc_hist[weight_index].min == data_hist.min)
            assert(mc_hist[weight_index].max == data_hist.max)
            assert(mc_hist[weight_index].nbins == data_hist.nbins)
            
            Nmc = mc_hist[weight_index].histogram
            Ndata = data_hist.histogram
            
            error = np.sqrt(np.maximum(Ndata, 1))
            chisquare = (np.power(Nmc-Ndata, 2)/np.power(error, 2)).sum() 
            
            X[weight_index] = weight_to_mass(weight_index)
            Y[weight_index] = chisquare

            if weight_index in (0, 18):
                mc_hist[weight_index].steps(label="MC m=%.3f GeV" % weight_to_mass(weight_index))

        data_hist.errorbar(label="Data", fmt=",")
        #data_hist.annotate("$\\chi^2$ = %.1f" % chisquare, loc=2)

        if name == "m_t":
            plt.xlabel("Transverse Mass / GeV")
        elif name == "el_e_t":
            plt.xlabel("Electron Transverse Energy / GeV")
        plt.ylabel("Number of Events")
        plt.legend()
        plt.savefig("comparison_%s.pdf" % name)

        print()

        plt.clf()
        plt.plot(X,Y, 's', color="black")
        
        try:
            parabel = lambda x, w_mass, minimum, sigma: np.power((x-w_mass)/sigma, 2) + minimum
            min_mass = weight_to_mass(0)
            max_mass = weight_to_mass(18)
            window = 0.3 # GeV
            for i in range(2):
                f = np.logical_and(X>=min_mass, X<=max_mass)
                popt, pcov = curve_fit(parabel, X[f], Y[f], p0=((max_mass+min_mass)/2, data_hist.nbins, 0.05))
                min_mass = max(popt[0] - window, weight_to_mass(0))
                max_mass = min(popt[0] + window, weight_to_mass(18))
                
            w_mass, minimum, sigma = popt
            sigma = abs(sigma)

            xnew = np.linspace(X.min(), X.max(), 1000)
            ynew = parabel(xnew, *popt)
            
            plt.axvline(w_mass, ls="-", color="black")
            plt.axhline(minimum+1, ls="--", color="grey")
            plt.axvline(w_mass-sigma, ls="--", color="grey")
            plt.axvline(w_mass+sigma, ls="--", color="grey")
            plt.plot(xnew,ynew, '-', color="grey")
            print("W mass:", w_mass, "GeV")
            u_w_mass = ufloat(w_mass, sigma)
            annotate("Fitted W Mass: ${m:L}$ GeV".format(m=u_w_mass))
            plt.ylim(minimum-1, minimum+10)
            
        except Exception as e:
            print("Error:", e)
        
        plt.xlabel("W Boson Mass / GeV")
        plt.ylabel(r"$\chi^2$")
        #plt.grid()
        plt.savefig("chisquare_%s.pdf" % name)

def task_5():
    with root_open("data/d0.root") as data_file:
        X = []
        Y = []
        
        N_events = data_file.MessTree.GetEntries()
        for event_number, event in enumerate(data_file.MessTree):                
            if event_number % 1000 == 0:
                print("\rProgress: %.1f %%" % (100*float(event_number)/float(N_events)), end="")
                sys.stdout.flush()

            event = MyEvent(event)
            if not event.passes_cuts():
                continue

            X.append(event.m_t)
            Y.append(event.el_e_t)

        X = np.array(X)
        Y = np.array(Y)

        #correlation = (( X-X.mean() )*( Y-Y.mean() )).sum() / np.sqrt(np.power(X-X.mean(), 2).sum()*np.power(Y-Y.mean(), 2).sum())
        correlation = np.corrcoef(X, Y)[0][1]
        annotate("Correlation: %.3f" % correlation, loc=2)
        plt.xlabel("Transverse Mass / GeV")
        plt.ylabel("Electron Transverse Energy / GeV")
        plt.plot(X[::10], Y[::10], ",", color="black")
        plt.xlim(20,100)
        plt.ylim(20,70)
        plt.savefig("correlation.pdf")

def task_6():
    m_W = ufloat(80.46, 0.06)
    m_Z = ufloat(91.1876, 0.0021)

    theta_W = umath.acos(m_W/m_Z)

    print("W mass:", m_W, "GeV")
    print("Z mass:", m_W, "GeV")
    print("weinberg:", umath.degrees(theta_W), "degrees")
    
    sin_theta_W_2 = 1 - umath.pow(m_W/m_Z, 2)
    print("sin(weinberg)^2:", sin_theta_W_2)
    
    
                
if __name__=="__main__":
    task_2()
