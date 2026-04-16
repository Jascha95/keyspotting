8#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 04:41:00 2020

@author: dzenn
"""

import toml
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class DynapseInfo():
    
    def __init__(self):
        
        self.dynapse_id = '00000'
        self.cores = _init_cores()
        # self.board_info = {}
        # self.board_info['overall_mismatch'] = None
        # self.board_info['board_spec'] = None
        
    
    def overview(self):
        """
            Prints an overview of what data is available for the given board
        """
        cores_data_avail = []
        ref_neurons_avail = []
        bias_stats_avail = []
        core_stats_avail = []
        bias_range_avail = []
        bias_osc_preset_avail = []
        bias_lut_avail = []
        bias_fit_avail = []
        
        for core in self.cores:
            
            
            # Check for ref neurons and stats per bias
            for bias in self.cores[core]:
                if self.cores[core][bias]['ref_neurons'] is not None:
                    ref_neurons_avail.append("Core " + str(core) + " " + str(bias))
                if self.cores[core][bias]['values'] is not None:
                    cores_data_avail.append("Core " + str(core) + " " + str(bias))
                if self.cores[core][bias]['stats'] is not None:
                    bias_stats_avail.append("Core " + str(core) + " " + str(bias))
                if self.cores[core][bias]['range'] is not None:
                    bias_range_avail.append("Core " + str(core) + " " + str(bias))
                if self.cores[core][bias]['osc_settings'] is not None:
                    bias_osc_preset_avail.append("Core " + str(core) + " " + str(bias))
                if self.cores[core][bias]['lut'] is not None:
                    bias_lut_avail.append("Core " + str(core) + " " + str(bias))
                        
                    
            # Check for stats per core
            if 'core_stats' in self.cores[core]:
                core_stats_avail.append(core)
                
        
        print("Value distribution available for cores:")
        print(cores_data_avail)
        
        print("Look-up tables available for biases:")
        print(bias_lut_avail)
        
        print("Comments:")
        
    
    def write_lut(self, core_id : int, bias_id : str, lut):
        """
        Method to write the usable range of the bias

        Parameters
        ----------
        core_id : int
        bias_id : str
        """
        
        self.cores[str(core_id)][bias_id]['lut'] = lut
        
    def write_values(self, core_id : int, bias_id : str, values):
        """
        Method to write the usable range of the bias

        Parameters
        ----------
        core_id : int
        bias_id : str

        """
        
        self.cores[str(core_id)][bias_id]['values'] = values
        
        
    def create_bias_tuning_curve_from_lut(self, core_id : int, bias_id : str, max_diff=None):
        """
        Performs fit with of the collected lut

        Parameters
        ----------
        core_id : int
        bias_id : str
        """
        
        if not max_diff:
            max_diff = 0.0015
                                          
        lut = self.cores[str(core_id)][bias_id]['lut']
        ids_to_fit = np.where(abs(np.diff(lut[:,0])) > max_diff)[0]
        popt, pcov = curve_fit(poly3, lut[ids_to_fit,0], lut[ids_to_fit,1],  p0=(1,1,1,1), method='dogbox')
        self.cores[str(core_id)][bias_id]['fit'] = popt
        
        
    def plot_bias_tuning_curve(self, core_id : int, bias_id : str):
        """
        Plot the tuning curve of a specific bias of the given core,
        both the captured lookup table and the fit

        Parameters
        ----------
        core_id : int
            Core of the bias to plot
        bias_id : str
            Bias to plot

        """
        
        lut = self.cores[str(core_id)][bias_id]['lut']
        fit = self.cores[str(core_id)][bias_id]['fit']
        
        plt.figure()
        plt.plot(lut[:,0], lut[:,1], 'o-', label="LUT")
        plt.plot(lut[:,0], poly3(lut[:,0],*fit), 'o-', label="fit")
        
        plt.title("Bias tuning curve for %s of the Core %d" % (bias_id, core_id))
        
        plt.xlabel("value")
        plt.ylabel("scaled bias")
        plt.legend()
        
        plt.show()
        
        
    def write_bias_range(self, core_id : int, bias_id : str, scaled_bias_min : float, scaled_bias_max : float):
        """
        Method to write the usable range of the bias

        Parameters
        ----------
        core_id : int
        bias_id : str
        scaled_bias_min : float
        scaled_bias_max : float
        """
        
        self.cores[str(core_id)][bias_id]['range'] = (scaled_bias_min, scaled_bias_max)
        
    
    def load_state(self, filename = None):
        """
        Loads the saved board info from file

        Parameters
        ----------
        filename : TYPE, optional
            The filename must end with ".toml" extension. If no filename is given,
            a opening of file with the serial number filename will be attempted.
        """
        
        if filename == None:
            filename = "dynapse_config/" + self.dynapse_id + ".toml"
        
        input_config_file = open(filename, "r")
        
        config_dict = toml.load(input_config_file)
        input_config_file.close()
                
        for core_id in range(16):
            for bias_id in self.cores[str(core_id)]:
                if bias_id in config_dict['cores'][str(core_id)].keys():
                    for param in config_dict['cores'][str(core_id)][bias_id]:
                        if param == 'lut' or param == 'fit' or param == 'values':
                            self.cores[str(core_id)][bias_id][param] = np.array(config_dict['cores'][str(core_id)][bias_id][param], dtype='float')
                        else:
                            self.cores[str(core_id)][bias_id][param] = config_dict['cores'][str(core_id)][bias_id][param]
                        
        self.dynapse_id = config_dict['board_info']['id']
        
        print("Load for Dynapse SN" + self.dynapse_id + " is complete")
        
    
    def export_state(self, filename = None):
        """
        Saves all internal values into the loadable configuration file

        Parameters
        ----------
        filename : TYPE, optional
            If no filename is given, board serial number is used.
        """
        
        if filename == None:
            filename = "dynapse_config/" + self.dynapse_id
        
        config_dict = {
            'board_info' : {'id' : self.dynapse_id},
            'cores' : self.cores            
            }
        
        output_config_file = open(filename + ".toml", "w")
        toml.dump(config_dict, output_config_file)
        output_config_file.close()
        

def _init_cores():
    
    init_dict = {}
    
    for core_id in range(16):
        init_dict[str(core_id)] = _init_core_data()
        
    return init_dict

def _init_core_data():
    
    init_dict = {
        'tau1' : _init_bias_data(),
        'tau2' : _init_bias_data(),
        'rfr' : _init_bias_data(),
        'fast_exc_tau' : _init_bias_data(),
        'fast_exc_weight' : _init_bias_data(),
        'slow_exc_tau' : _init_bias_data(),
        'slow_exc_weight' : _init_bias_data(),
        'fast_inh_tau' : _init_bias_data(),
        'fast_inh_weight' : _init_bias_data(),
        'slow_inh_tau' : _init_bias_data(),
        'slow_inh_weights' : _init_bias_data(),
        'pulse_width' : _init_bias_data()
        }
    
    return init_dict

def _init_bias_data():
    
    init_dict = {
        'ref_neurons' : None,
        'stats' : None,
        'range' : None,
        'osc_settings' : None,
        'lut' : None,
        'fit' : None,
        'values' : None
        }
        
    return init_dict
        
def poly3(x, a, b, c, d):
    return a*x*x*x + b*x*x + c*x + d
