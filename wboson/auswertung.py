from __future__ import print_function
import os, os.path
import sys
import contextlib
import math

import ROOT
import numpy as np
import matplotlib.pyplot as plt

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
        
        data_lumi = 198.0 # +- 20, pb
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
                    
                #if hists["E_T_el"].count() > 1000:
                #    break
                    
                if event_number % 1000 == 0:
                    print("\rProgress: %.1f %%" % (100*float(event_number)/float(N_events)), end="")
                    sys.stdout.flush()

                if hasattr(event, "weight"):
                    weight = event.weight[9]
                else:
                    weight = 1.0
                
                # transverse energy of the electron
                E_T_el = math.sqrt(event.el_e**2 - event.el_pz**2)
                hists["E_T_el"].fill(E_T_el, weight)

                # missing transverse energy 
                E_T_miss = math.sqrt(event.metx_calo**2 + event.mety_calo**2)
                hists["E_T_miss"].fill(E_T_miss, weight)
                
                m_T = math.sqrt(E_T_el*E_T_miss*(1-math.cos(event.el_met_calo_dphi)))
                hists["m_T"].fill(m_T, weight)

                hists["eta_el"].fill(event.el_eta, weight)
                hists["phi_el"].fill(event.el_phi, weight)

                phi_miss = math.atan2(event.metx_calo, event.mety_calo) + math.pi
                hists["phi_miss"].fill(phi_miss, weight)

                hists["delta_phi"].fill(event.el_met_calo_dphi, weight)

                delta_z = event.el_track_z - event.met_vertex_z
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
            mc_hist.rescale(lumi_scale)
            mc_hist.steps(label="MC")

            tau_hist = all_hists["mc_tau"][quantity]
            tau_hist.rescale(lumi_scale)
            tau_hist.steps(label="MC Tau", color="red")
            
            plt.xlabel(xlabels[quantity])
            plt.ylabel("Number of Events")
            plt.legend()
            plt.savefig("images/%s.pdf" % quantity)
            plt.show()
                
            
        
if __name__=="__main__":
    task_2()
