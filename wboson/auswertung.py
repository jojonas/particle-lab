from __future__ import print_function
import os, os.path
import sys
import contextlib
import math

import ROOT
import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat

from myoptimize import curve_fit
#from scipy.optimize import curve_fit

from rootwrapper import *
from jhist import Hist
import jmpl

#curve_fit(lambda x, a: x*a, [1,0],[1,1])

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
            hists[i].steps(label="m = %g GeV" % weight_to_mass(weight), alpha=0.5)
            #hists[i].stats_box(loc=3)
            
        plt.xlabel("Generated W Mass / GeV")
        plt.ylabel("Number of Events")
        plt.legend()
        plt.show()

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
                
            for event_number, event in enumerate(tree):
                if event_number < first_event:
                    continue

                #if event_number % 10 != 0:
                #    continue
                                        
                #if hists["E_T_el"].count() > 1000:
                #    break
                    
                if event_number % 1000 == 0:
                    print("\r", name, "- progress: %.1f %%           " % (100*float(event_number)/float(N_events)), end="")
                    sys.stdout.flush()

                if hasattr(event, "weight"):
                    weight = event.weight[9]
                else:
                    weight = 1.0
                
                E_T_el = math.sqrt(event.el_e**2 - event.el_pz**2)
                E_T_miss = math.sqrt(event.metx_calo**2 + event.mety_calo**2)
                phi_miss = math.atan2(event.metx_calo, event.mety_calo) + math.pi
                delta_z = event.el_track_z - event.met_vertex_z
                m_T = math.sqrt(E_T_el*E_T_miss*(1-math.cos(event.el_met_calo_dphi)))

                cuts = True
                if cuts:
                    # GOOD CUTS!
                    if E_T_miss < 20 or E_T_el < 30 or event.el_iso > 0.03 \
                        or event.el_met_calo_dphi < 2.85 or abs(delta_z) > 0.2:
                        continue
                
                hists["E_T_el"].fill(E_T_el, weight)
                hists["E_T_miss"].fill(E_T_miss, weight)
                hists["m_T"].fill(m_T, weight)
                hists["eta_el"].fill(event.el_eta, weight)
                hists["phi_el"].fill(event.el_phi, weight)
                hists["phi_miss"].fill(phi_miss, weight)
                hists["delta_phi"].fill(event.el_met_calo_dphi, weight)
                hists["delta_z"].fill(delta_z, weight)

        xlabels = {
            "E_T_el": "Electron Transverse Energy / GeV",
            "E_T_miss": "Missing Transverse Energy / GeV",
            "m_T": "Transverse Mass / GeV",
            "eta_el": "Electron Eta",
            "phi_el": "Electron Phi",
            "phi_miss": "MET Phi",
            "delta_phi": "Delta Phi",
            "delta_z": "Delta z / mm",
        }

        for quantity in all_hists["data"].keys():
            plt.clf()

            data_hist = all_hists["data"][quantity]
            data_hist.errorbar(label="Data", color="black", fmt=".")
            
            mc_hist = all_hists["mc"][quantity]

            Nmc = mc_hist.count()
            
            mc_hist.rescale(lumi_scale)
            mc_hist.steps(label="MC")

            tau_hist = all_hists["mc_tau"][quantity]
            tau_hist.rescale(lumi_scale)
            tau_hist.steps(label="MC Tau", color="red")

            text = "Data: %d\nMC: %d\nMC raw: %d" % (data_hist.count(), mc_hist.count(), Nmc)
            data_hist.annotate(text, loc=2)
            
            plt.xlabel(xlabels[quantity])
            plt.ylabel("Number of Events")
            plt.legend()
            plt.savefig("images/%s.pdf" % quantity)
            plt.show()
                
def task_3():
    # CACHE
    mc_allcuts = 74072. 
    #mc_allcuts = 187647. 
    data = 67329.

    #mc_gen_count = 1164699.
    process_xsec = ufloat(2.58e3, 0.09e3)
    #correction = ufloat(0.9, 0.1)
    #luminosity = ufloat(198., 20.)

    n_data = ufloat(data, math.sqrt(data))

    #lumi_scale = luminosity * process_xsec / mc_gen_count * correction
    #mc_gen_count *= lumi_scale
    
    #efficiency = mc_allcuts / mc_gen_count
    #xsec = n_data / (luminosity * efficiency * correction)

    xsec = n_data / mc_allcuts * process_xsec

    #print("Efficiency: {eff:.1f} %%".format(eff=efficiency*100))
    print("XSec: {xsec:.2f} nb".format(xsec=xsec*1e-3))

def task_4():
    with contextlib.nested(root_open("data/mc_all.root"), root_open("data/d0.root")) as (mc_file, data_file):
        mc_hist = list( Hist(40, 70, 50) for _ in range(19) )
        data_hist = Hist(40, 70, 50)

        for name, tree in zip(("mc", "data"), (mc_file.MCTree, data_file.MessTree)):
            N_events = tree.GetEntries()
            for event_number, event in enumerate(tree):
                #if event_number % 5 != 0:
                #    continue
                    
                if event_number % 1000 == 0:
                    print("\rProgress: %.1f %%" % (100*float(event_number)/float(N_events)), end="")
                    sys.stdout.flush()
                    
                E_T_el = math.sqrt(event.el_e**2 - event.el_pz**2)
                E_T_miss = math.sqrt(event.metx_calo**2 + event.mety_calo**2)
                m_T = math.sqrt(E_T_el*E_T_miss*(1-math.cos(event.el_met_calo_dphi)))
                delta_z = event.el_track_z - event.met_vertex_z
                
                if E_T_miss < 20 or E_T_el < 30 or event.el_iso > 0.03 \
                    or event.el_met_calo_dphi < 2.85 or abs(delta_z) > 0.2:
                    continue

                if name == "data":
                    data_hist.fill(m_T)
                elif name == "mc":
                    for i, weight in enumerate(event.weight):
                        mc_hist[i].fill(m_T, weight)
                else:
                    raise ValueError("WTF")


    X = np.zeros(19)
    Y = np.zeros(19)
    for weight_index in range(19):

        factor = data_hist.count() / mc_hist[weight_index].count()
        mc_hist[weight_index].rescale(factor)
        
        Nmc = mc_hist[weight_index].histogram
        Ndata = data_hist.histogram
        
        error = np.sqrt(np.maximum(Ndata, 1))
        chisquare = (np.power(Nmc-Ndata, 2)/np.power(error, 2)).sum() 
        
        X[weight_index] = weight_to_mass(weight_index)
        Y[weight_index] = chisquare

        if weight_index in (0, 18):
            #plt.clf()
            mc_hist[weight_index].steps(label="MC")
            data_hist.errorbar(label="Data", fmt=",")
            data_hist.annotate("Chi^2 = %.1f" % chisquare, loc=2)

    plt.xlabel("Transverse Mass / GeV")
    plt.ylabel("Number of Events")
    plt.legend()
    plt.savefig("comparison.pdf")
    plt.show()

    print()
       
    parabel = lambda x, a, b, c: c*np.power(x-a, 2) + b
    min_mass = weight_to_mass(0)
    max_mass = weight_to_mass(18)
    window = 0.2 # GeV
    for i in range(5):
        f = np.logical_and(X>=min_mass, X<=max_mass)
        popt, pcov = curve_fit(parabel, X[f], Y[f], p0=(80, 100, 100))
        min_mass = popt[0] - window
        max_mass = popt[0] + window

    xnew = np.linspace(X.min(), X.max(), 1000)
    ynew = parabel(xnew, *popt)

    print("W mass:", popt[0], "GeV")
    
    plt.clf()
    plt.xlabel("W Boson Mass / GeV")
    plt.ylabel("Chi^2")
    plt.grid()
    plt.plot(X,Y, 's', color="black")
    plt.plot(xnew,ynew, '-', color="grey")
    plt.savefig("chisquare.pdf")
    plt.show()

    
                
                
                
if __name__=="__main__":
    task_3()
