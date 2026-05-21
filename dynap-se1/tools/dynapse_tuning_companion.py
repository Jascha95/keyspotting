#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import samna
import sys
sys.path.append('..')
import samna.dynapse1 as dyn1
import dynapse1utils as ut
from dynapse1constants import *
import netgen
from netgen import Neuron
import numpy as np
import matplotlib.pyplot as plt
from tools.notebook_gui import make_parameter_slider_box
from details.waveform_analysis_details import get_falling_times, get_falling_voltage_and_time, get_tau
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from time import sleep



class DynapseTuningCompanion(object):
    """ A class to assist with tuning DYNAP-SE1 chip parameters. It connects to a specified core of the chip and provides
    a number of easy-access simple functionalities, as well as some more complex automation if the external oscilloscope is attached.
    
    Simple functions (no oscilloscope needed):
        - DC pulses of specified length (by automating the DC bias of the core)
        - spiking regular rate input from the FPGA through one of the four synapses
        - quick bias manipulation to bring the board to the state that allows tuning of specific parameters,
            with the option to revert to the previous state
        
    Complex functions (PyScope object connection required):
        - extraction of neuron and synapse time constants using the oscilloscope;
        waveform fitting
        - extraction of neuron refractory period through sending input and processing the waveform
        - extraction of synaptic weights
        - parameter capture sweep across the core
        - IN DEVELOPMENT: extraction of parameter-value maps

    Args:
        model (Dynapse1Model) : DYNAP-SE1 model object
        chip_id (int) : target chip id [0..3]
        core_id (int) : targt core id [0..3]
        net_gen (NetworkGenerator, optional) : external network generator object, to be used to avoid overwriting previously created connections
        target_neurons (list([NetworkGenerator.Neuron]), optional) : a list of specific neurons the Dynapse Tuning Companion should track, instead of the whole core.
                                                                    to be used for tuning specific populations of neurons.
        scope (PyScope, optional) : external oscilloscope object for waveform extraction. To be used to avoid opening the same physical oscilloscope twice,
                                                                    in case it is used by other classes outside of DTC. 
    """
    
    def __init__(self, model, chip_id, core_id, net_gen=None, target_neurons=None, scope=None):
        
        self.model=model
        if net_gen is None:
            net_gen = netgen.NetworkGenerator()
        else:
            self.net_gen=net_gen
        self._spikegen_connected=False
        self._scope_connected=False
        self._loaded_state_available=False
        self.info=None
        self._target_chip_id=chip_id
        self._target_core_id=core_id
        self._source_neuron=Neuron(0,0,1, is_spike_gen=True)
        self._source_neuron_id=1
        self._tmp_param_group=None
        self.fpga_spike_gen=model.get_fpga_spike_gen()

        if target_neurons is None:
            self.neurons = [Neuron(self._target_chip_id, self._target_core_id, neuron_id) for neuron_id in range(256)]
        else:
            self.neurons = target_neurons
            
        self.event_graph, self.event_filter_node, self.event_sink_node = ut.create_neuron_select_graph(self.model, [(n.chip_id, n.core_id, n.neuron_id) for n in self.neurons])
        
    
    def connect_spikegen(self, source_neuron_id=1):
        
        if self._spikegen_connected:
            raise Warning("Spikegen already connected!")
        
        self._source_neuron_id=source_neuron_id
        self._source_neuron=Neuron(0, source_neuron_id//256,source_neuron_id%256,is_spike_gen=True)
        self.neurons=[Neuron(self._target_chip_id, self._target_core_id, neuron_id%256) for neuron_id in range(256)]

        for neuron in self.neurons:
            self.net_gen.add_connection(self._source_neuron, neuron, dyn1.Dynapse1SynType.AMPA)
            self.net_gen.add_connection(self._source_neuron, neuron, dyn1.Dynapse1SynType.NMDA)
            self.net_gen.add_connection(self._source_neuron, neuron, dyn1.Dynapse1SynType.GABA_A)
            self.net_gen.add_connection(self._source_neuron, neuron, dyn1.Dynapse1SynType.GABA_B)
    
        new_config = self.net_gen.make_dynapse1_configuration()
        ut.apply_biases_from_configuration(self.model.get_configuration(), new_config)
        self.model.apply_configuration(new_config)
        
        print(f"Spikegen connected to C{self._target_chip_id}c{self._target_core_id}\n")
        
        self._spikegen_connected = True
        return
    
    def disconnect_spikegen(self):
        if self._spikegen_connected:
            for neuron in self.neurons:
                self.net_gen.remove_connection(self._source_neuron, neuron, dyn1.Dynapse1SynType.AMPA)
                self.net_gen.remove_connection(self._source_neuron, neuron, dyn1.Dynapse1SynType.NMDA)
                self.net_gen.remove_connection(self._source_neuron, neuron, dyn1.Dynapse1SynType.GABA_A)
                self.net_gen.remove_connection(self._source_neuron, neuron, dyn1.Dynapse1SynType.GABA_B)
    
        else:
            print("Warning: spikegen not connected!")
            return
        
    
    def set_spikegen_rate(self, rate_hz):
        timestamps = np.linspace(0,1,rate_hz)
        indices = np.ones(len(timestamps))*self._source_neuron_id
        
        ut.set_fpga_spike_gen(self.fpga_spike_gen,
                              spike_times=timestamps,
                              indices=indices.astype(int),
                              target_chips=[self._target_chip_id]*len(timestamps), isi_base=900, repeat_mode=True)

        self.fpga_spike_gen.start()
        
        
    def detect_hot_neurons(self):
        #TODO: record 3 seconds of spiking activity, determine the mean and sigma of rates
        #      return extreme outliers
        pass
    
    
    def dc_steps(self, period_s, duration_s, bias_max_index=1600, bias_format='index'):
        ut.dc_steps(self.model, self._target_chip_id, self._target_core_id, period_s, duration_s, bias_max_index, bias_format)
    
    
    def monitor_neuron(self, neuron_id):
        ut.monitor_neuron(self.model, self._target_chip_id, self._target_core_id, neuron_id)
        
        
    def connect_pyscope(self, device_id_string=None):
        #TODO adapt this to use SPCM scope
        from pygetscope.py_scope import PyScope        
        self.scope = PyScope(device_id_string)
        self._scope_connected = True
        
    def connect_scope(self, scope):
        
        self.scope = scope
        self._scope_connected = True
        
    def load_info(self, dynapse_id : str, filename = None):
        """
            Load available dynapse calibration data
        """
        self.info.dynapse = dynapse_id
        self.info.load_state(filename)
        self._loaded_state_available = True
        
    def store_biases(self):
        tmp_config = self.model.get_configuration()
        self._tmp_param_group = tmp_config.chips[self._target_chip_id].cores[self._target_core_id].parameter_group
    
    def restore_biases(self):
        if self._tmp_param_group:
            self.model.update_parameter_group(self._tmp_param_group,
                                     self._target_chip_id, self._target_core_id)
        else:
            raise RuntimeError("No temporary biases saved to restore!")
            
    def disable_synapses(self):
        
        ut.set_ampa_tau(self.model, self._target_chip_id, self._target_core_id, 1811, bias_format='index')
        ut.set_nmda_tau(self.model, self._target_chip_id, self._target_core_id, 1811, bias_format='index')
        ut.set_gaba_a_tau(self.model, self._target_chip_id, self._target_core_id, 1811, bias_format='index')
        ut.set_gaba_b_tau(self.model, self._target_chip_id, self._target_core_id, 1811, bias_format='index')
        
        
    def set_state_measure_neuron_tau(self):
        
        ut.set_neuron_adaptation_tau(self.model, self._target_chip_id, self._target_core_id, 1811, bias_format='index')
        ut.set_neuron_tau2(self.model, self._target_chip_id, self._target_core_id, 1811, bias_format='index')
        ut.set_neuron_gain(self.model, self._target_chip_id, self._target_core_id, 0, bias_format='index')
        ut.set_nmda_enable(self.model, self._target_chip_id, self._target_core_id, 0, bias_format='index')
        ut.set_neuron_dc(self.model, self._target_chip_id, self._target_core_id, 0, bias_format='index')
        
    def set_state_measure_synapses(self):
        
        ut.set_neuron_adaptation_tau(self.model, self._target_chip_id, self._target_core_id, 1811, bias_format='index')
        ut.set_neuron_tau1(self.model, self._target_chip_id, self._target_core_id, 1811, bias_format='index')
        ut.set_neuron_tau2(self.model, self._target_chip_id, self._target_core_id, 1811, bias_format='index')
        ut.set_neuron_gain(self.model, self._target_chip_id, self._target_core_id, 0, bias_format='index')
        ut.set_nmda_enable(self.model, self._target_chip_id, self._target_core_id, 0, bias_format='index')
        ut.set_neuron_dc(self.model, self._target_chip_id, self._target_core_id, (7, 200), bias_format='tuple')
    
    def set_state_refractory_period(self):
        
        #TODO set parameters for neurons to fire with 10+ Hz across the whole core
        pass
    
    def show_neuron_parameter_controls(self):
        
        core_prefix = f"C{self._target_chip_id}c{self._target_core_id}:"
        
        parameter_bias_tags = [core_prefix + 'tau1',
                               core_prefix + 'rfr',
                               core_prefix + 'dc',
                               core_prefix + 'gain']
        
        return make_parameter_slider_box(self.model, parameter_bias_tags, "Neuron Parameters " + f"C{self._target_chip_id}c{self._target_core_id}")
    
    def show_synapse_parameter_controls(self, syn_type : str):
        
        if syn_type not in ['ampa', 'nmda', 'gaba_a', 'gaba_b']:
            raise ValueError("Invalid synapse type. Use \'ampa\', \'nmda\', \'gaba_a\' or \'gaba_b\'")
        
        core_prefix = f"C{self._target_chip_id}c{self._target_core_id}:"
        
        parameter_bias_tags = [core_prefix + syn_type + '_tau',
                               core_prefix + syn_type + '_gain',
                               core_prefix + syn_type + '_w']
        
        return make_parameter_slider_box(self.model, parameter_bias_tags, "Synapse Parameters " + f"C{self._target_chip_id}c{self._target_core_id}")
    
    
    def get_rfr_period(self, neuron_id : int, channel_id : int, instant_plot=False):
        
        if not self._scope_connected:
            raise RuntimeError("Oscilloscope is not connected! Connect an oscilloscope to analyse waveforms")
        
        event_filter = self.event_sink_node
        self.event_graph.start()
        event_filter.get_events()
        
        ut.monitor_neuron(self.model, self._target_chip_id, self._target_core_id, neuron_id)
        wf = self.scope.get_waveform(channel_id)
        
        num_events = len(event_filter.get_events())
        self.event_graph.stop()
        
        if num_events < 5:
            print("Warning: firing rate too low, no rfr measured!")
            return 100
        
        x = wf[:,0]
        y = wf[:,1]

        #time_range = x[-1] - x[0]
        #arr_range = len(x)
        pks_height = np.min(y) + 0.75*(np.max(y) - np.min(y))
        pks, _ = find_peaks(y, distance=100, height=pks_height)
        
        if len(pks) < 2:
            print("Error: too few peaks in waveform (%d peaks)" % len(pks))
            return 100
        
        pks_diff = np.diff(pks)[1]

        num_post_peak_points = pks_diff-20
        
        selected_y = y[pks[1]:pks[1]+num_post_peak_points]
        selected_x = x[pks[1]:pks[1]+num_post_peak_points] - x[pks[1]]
        
        y_sel_min = np.min(selected_y)
        y_sel_max = np.max(selected_y)
        y_thr = y_sel_min + 0.2*(y_sel_max-y_sel_min)
        
        rfr_start_idx = np.argwhere(selected_y<y_thr)[0][0]
        rfr_end_idx = rfr_start_idx + np.argwhere(selected_y[rfr_start_idx:]>y_thr)[0][0]
        
        rfr_period = selected_x[rfr_end_idx] - selected_x[rfr_start_idx]
        
        if instant_plot:
            plt.plot(wf[:,0], np.log(wf[:,1]))
            plt.xlabel("time, seconds")
            plt.ylabel("V")
        
        return wf, rfr_period
    
        
    def get_waveform_time_constant(self, channel_id):
        """
            Function to get a single waveform and a fall time measured by the scope
        """
        
        if not self._scope_connected:
            raise RuntimeError("Oscilloscope is not connected! Connect an oscilloscope to analyse waveforms")

        fall_time = float(self.scope.do_query_string(':MEAS:FALL? CHAN' + str(channel_id)))
        rise_time = float(self.scope.do_query_string(':MEAS:RIS? CHAN' + str(channel_id)))
        waveform = self.scope.get_waveform(channel_id)
        if fall_time>1 and rise_time>1:
            TC = fall_time
        else:
            try:
                TC = get_tau_from_waveform(waveform)
            except UnboundLocalError:
                print("Fitting failed, returning fall time instead")
                TC = fall_time
            
        return waveform, fall_time, rise_time, TC
    
    
    def get_waveform_amplitude(self, channel_id : int):
        """
        Returns aplitude of the signal

        Parameters
        ----------
        channel_id : TYPE
            Oscilloscope channel

        Returns
        -------
        weight : float
            Voltage delta in V

        """
        
        waveform = self.scope.get_waveform(channel_id)
        return waveform, waveform[:,1].max() - waveform[:,1].min()
    
    
        
    def get_neuron_time_constant(self, neuron_id : int, channel_id : int, dc_step_max_index=1000, dc_step_length=0.2, debug_plot=False,
                                 debug=False):
        """
        Returns neuron time constant and the wavefrom.
        The function manipulates the vertical scale of the oscilloscope for better defintion

        Parameters
        ----------
        neuron_id : int
            Index of the neuron to record from
        channel_id : int
            Oscilloscope channel to record from
        dc_step_length : int, optional
            Length of the DC pulse - might require adjustment for fast or slow values of observed neuron TC.
            Default value is 0.2.
        debug_plot : bool, optional
            Plot the waveform to confirm the returned TC is adequate
        debug : bool, optional
            Additional prints

        Returns
        -------
        waveform : np.array
            Acquired waveform from which the TC was extracted
        TC : float
            The measured time constant
        """
        
        _neuron_spikes_flag=False
        event_filter = self.event_sink_node
        self.event_graph.start()
        event_filter.get_events()
        ut.monitor_neuron(self.model, self._target_chip_id, self._target_core_id, neuron_id)
        # for attempt in range(3):            
        while True:
            
            if debug:
                print("Starting waveform recording...")
            
            self.scope.run()
            sleep(0.2)
            if debug:
                print("DC steps...")
            self.dc_steps(dc_step_length, dc_step_length*5, dc_step_max_index)
            if debug:
                print("Done.")
            self.scope.stop()
            
            # if fall_time < 1:
            #     break
            # else:
            #     self.scope.find_offset(channel_id)
            if debug:
                print("Recording stopped. Starting waveform retrival...")
            
            waveform, fall_time, rise_time, TC = self.get_waveform_time_constant(channel_id)
            
            y_min = np.min(waveform[:,1])
            y_max = np.max(waveform[:,1])
            scale = self.scope.get_channel_scale(channel_id)
            
            if debug:
                print("Waveform capture complete, checking if the neuron spiked...")
            
            # check if the measured neuron spikes and abort if it does
            evts = event_filter.get_events()
            
            if debug:
                print("Spikes recorded:", len(evts))
            if len(evts):
                for evt in evts:
                    if evt.neuron_id == neuron_id:
                        _neuron_spikes_flag=True
            
            if debug:
                print("Spike capture complete")
            
            if _neuron_spikes_flag:
                self.event_graph.stop()
                print("Neuron %d spikes, unable to measure the time constant!" % neuron_id)
                return waveform, np.NAN
            
            # autoscale if the trace is larger than 60% of the oscilloscope observable range
            if y_max - y_min>scale*6:
                print("Autoscaling!")
                self.scope.set_channel_scale(channel_id, scale*1.2)
                self.scope.find_offset(channel_id)
            else:
                break
    
        if debug_plot:
            plt.plot(waveform[:,0], np.log(waveform[:,1]))
            plt.xlabel("time, seconds")
            plt.ylabel("V")
    
        if debug:
            print("Time constant is %g ms" % (TC*1000))
            print("Fall time is %g ms" % (fall_time*1000))
            
        self.event_graph.stop()
            
        return waveform, TC
    
    
    
    def get_synapse_time_constant(self, channel_id, syn_type : str, debug_plot=False, debug=False):
        #TODO figure out why fall_time and rise_time are returned, and not the TC
        # Perhaps TC is returned incorrectly for inhibitory pulses?
        
        if str(syn_type+'_w') not in BIAS_TAGS:
            raise ValueError("syn_type must be \'ampa\', \'nmda\', \'gaba_a\' or \'gaba_b\'")
        
        waveform, fall_time, rise_time, TC = self.get_waveform_time_constant(channel_id)
    
        if debug_plot:
            plt.plot(waveform[:,0], np.log(waveform[:,1]))
            plt.xlabel("time, seconds")
            plt.ylabel("V")
    
        if debug:
            print("Time constant is %g ms" % (fall_time*1000))
        
        
        #TODO add TC export instead of fall\rise time
        
        if syn_type == 'ampa' or syn_type == 'nmda':
            return waveform, fall_time
        if syn_type == 'gaba_a' or syn_type == 'gaba_b':
            return waveform, rise_time
        
    
    
    def find_neuron_tau_range(self, neuron_id, channel_id, bias_step_normalized=0.05):
        '''
            Attempt to find the usable range of tau by sweeping the whole tau1 range
        
        '''
        if not self._scope_connected:
            raise RuntimeError("Oscilloscope no connected")
        
        self.disable_synapses()
        self.monitor_neuron(neuron_id)
        self.scope.hide_all_channels()
        self.scope.show_channel(channel_id)
        
        PEAK_TO_PEAK_MARGIN = 0.04
        BIAS_STEP = bias_step_normalized
        
        scaled_bias = 1

        ## Step 1 getting to the interval of search given by the amplitude
        scaled_bias_max = 0
        scaled_bias_min = 0
        peak_to_peak_arr = [0]
        
        # TODO: check why we need to bump the neuron gain a bit
        ut.set_neuron_gain(self.model, self._target_chip_id, self._target_core_id, int(1811*0.3), bias_format='index')
        self.scope.set_timescale(0.04)
        self.scope.set_channel_scale(channel_id, 0.05)
        self.scope.find_offset(channel_id)
        
        while True:
            scaled_bias -= BIAS_STEP
            
            ut.set_neuron_tau1(self.model, self._target_chip_id, self._target_core_id, int(1811*scaled_bias), bias_format='index')
            waveform, current_tau = self.get_neuron_time_constant(neuron_id, channel_id)
            peak_to_peak = np.max(waveform[:,1]) - np.min(waveform[:,1])
            
            if peak_to_peak > np.min(peak_to_peak_arr) + PEAK_TO_PEAK_MARGIN and scaled_bias > scaled_bias_max:
                scaled_bias_max = scaled_bias
                tau_at_max = current_tau
            
            if peak_to_peak < np.max(peak_to_peak_arr) - PEAK_TO_PEAK_MARGIN:
                scaled_bias_min = scaled_bias
                tau_at_min = current_tau
                break
        
            peak_to_peak_arr.append(peak_to_peak)
            
        self.tau1_range = (scaled_bias_min, scaled_bias_max)
        
    
    def find_synapse_tau_range(self, core_id, channel_id, neuron_id, syn_type : str):
        
        raise NotImplementedError("This function does not work properly yet...")
        
        # input checks
        if str(syn_type+'_w') not in BIAS_TAGS:
            raise ValueError("syn_type must be \'ampa\', \'nmda\', \'gaba_a\' or \'gaba_b\'")
        if not self._scope_connected:
            raise RuntimeError("Oscilloscope no connected")
        
        self.disable_synapses()
        self.set_state_measure_synapses()
        
        BIAS_STEP = 0.05
        
        scaled_bias = 1

        scaled_bias_max = 0
        scaled_bias_min = 0
        
        peak_to_peak_max = 0
        max_set_flag = False
        base_dc_level = 5
        
        ut.set_neuron_tau1(self.model, self._target_chip_id, self._target_core_id,
                         1811, bias_format='index')
        ut.set_neuron_dc(self.model, self._target_chip_id, self._target_core_id,
                         int(1811*0.96), bias_format='index')
        self.monitor_neuron(neuron_id)
        self.scope.hide_all_channels()
        self.scope.show_channel(channel_id)
        self.scope.set_timescale(0.01)
        self.scope.set_channel_scale(channel_id, 0.02)
        self.scope.find_offset(channel_id)
        
        if syn_type == 'gaba_a' or syn_type == 'gaba_b':
            offset = self.scope.get_channel_offset(channel_id)
            self.scope.set_channel_offset(channel_id, offset-0.02*6)
        
        # Set max value of the corresponding weight
        ut._set_bias(self.model, self._target_chip_id, self._target_core_id,
                     bias_name=BIAS_NAMES[BIAS_TAGS_LOOKUP[syn_type+'_w']],
                     bias_value=1811, bias_format='index')
        
        waveform, current_tau = self.get_synapse_time_constant(channel_id, syn_type)
        #base_dc_level = np.mean(waveform[:,1])
        
        while True:
            scaled_bias -= BIAS_STEP
            # set synapse tau using the index bias
            ut._set_bias(self.model, self._target_chip_id, self._target_core_id,
                    bias_name=BIAS_NAMES[BIAS_TAGS_LOOKUP[syn_type+'_tau']],
                    bias_value=int(1811*scaled_bias), bias_format='index')
            # clamp the I_gain to 2x the I_tau using the linear bias
            gain_bias_linear = ut._bias_as_linear(int(1811*scaled_bias), bias_format='index')
            ut._set_bias(self.model, self._target_chip_id, self._target_core_id,
                    bias_name=BIAS_NAMES[BIAS_TAGS_LOOKUP[syn_type+'_gain']],
                    bias_value=np.clip(2*gain_bias_linear, 0, LINEAR_BIAS_MAX), bias_format='linear')
            
            waveform, current_tau = self.get_synapse_time_constant(channel_id, syn_type)
            min_voltage = np.min(waveform[:,1])
            max_voltage = np.max(waveform[:,1])
            peak_to_peak = max_voltage - min_voltage
            
            if peak_to_peak > peak_to_peak_max:
                peak_to_peak_max = peak_to_peak
                
            if peak_to_peak < peak_to_peak_max*0.9 and not max_set_flag:
                print(current_tau)
                self.syn_tau_bias_range[syn_type] = (0, scaled_bias)
                
        #### TODO: finish the condition for reurning the found bounds
        
        
        
    def tune_tau(self, time_constant, channel_id, neuron_id, mode='neuron', precision=None):
        
        raise NotImplementedError("This function does not work properly yet...")
        
        '''
            This function tries to bring the actual TC as close as possible to the given target time_constant
        '''
        
        if not self._scope_connected:
            raise RuntimeError("Connect the oscilloscope to run this function")

        if precision is None:
            precision = time_constant/10
            
        syn_type = -1
    
        self.bts.monitor(core_id, neuron_id)
        self.scope.hide_all_channels()
        self.scope.show_channel(channel_id)
        
        self.bts.groups[core_id].set_linear_bias('IF_THR_N', int(pow(MAX_BIAS,0)))
    
        
        if mode == 'neuron':
            print("Will tune neuron")
            scaled_bias_min, scaled_bias_max = self.tau1_range
            if core_id != self.last_tuned_core_id or self.last_tuned_bias != 'tau1':
                print("Setting up for tuning neuron")
                
                self.scope.set_channel_scale(channel_id, 0.01)
                self.scope.set_timescale(0.04)
                print("Offsetting before measurement")
                self.set_neuron_tau_bias(core_id,  (scaled_bias_max + scaled_bias_min)/2)   
                self.scope.find_offset(channel_id)
                self.last_tuned_core_id = core_id
                self.last_tuned_bias = 'tau1'
        else:
            if self._spikegen_connected == False:
                raise Exception("Error: spikegen not connected!")
            self.set_neuron_tau_bias(core_id, 1)    
            self.bts.groups[core_id].set_linear_bias('IF_DC_P', int(pow(MAX_BIAS,0.96)))
            if mode == 'fast_exc':
                syn_type = 3
            elif mode == 'slow_exc':
                syn_type = 2
            elif mode == 'fast_inh':
                syn_type = 1
            elif mode == 'slow_inh':
                syn_type = 0
            else:
                raise Exception("Error: unknown time constant tuning mode")
            scaled_bias_min, scaled_bias_max = self.syn_tau_bias_range[syn_type]
            
            if core_id != self.last_tuned_core_id or self.last_tuned_bias != mode:
                print("Setting up for tuning synapse")
                self.scope.set_channel_scale(channel_id, 0.02)
                self.scope.set_timescale(0.01)
                self.scope.find_offset(channel_id)
                self.last_tuned_core_id = core_id
                self.last_tuned_bias = mode
                if syn_type == 0 or syn_type == 1:
                    offset = self.scope.get_channel_offset(channel_id)
                    self.scope.set_channel_offset(channel_id, offset-0.005*6)
                
        scaled_bias = 1        

        ## getting to the exact time constant
        while True:
            scaled_bias = (scaled_bias_max + scaled_bias_min)/2
            
            if mode == 'neuron':
                self.set_neuron_tau_bias(core_id, scaled_bias)            
                waveform, current_tau = self.get_neuron_time_constant(core_id, channel_id, neuron_id)
            else:
                self.set_synapse_tau_bias(core_id, scaled_bias)
                waveform, current_tau = self.get_synapse_time_constant(core_id, channel_id, syn_type)
        
            err = abs(current_tau - time_constant)
            print("Error is %g ms. Current tau is %g" % (err*1000, current_tau*1000))
        
            if err<precision:
                break
            
            if current_tau < time_constant:
                scaled_bias_max = scaled_bias
            else:
                scaled_bias_min = scaled_bias
                
        return scaled_bias, current_tau
    
        
    def full_tau1_tune(self, value, core_id, channel_id, neuron_id):
        raise NotImplementedError("This function does not work properly yet...")
        self.set_state_tau(core_id)
        self.find_neuron_tau_range(core_id, channel_id, neuron_id)
        return self.tune_tau(value, core_id, channel_id, neuron_id, mode='neuron')
        
    def full_syn_tau_tune(self, value, core_id, channel_id, neuron_id, syn_type : int):
        raise NotImplementedError("This function does not work properly yet...")
        self.set_state_tau(core_id)
        self.find_synapse_tau_range(core_id, channel_id, neuron_id)
        return self.tune_tau(value, core_id, channel_id, neuron_id, mode=syn_types[syn_type])
    
    def prepare_sweep(self, core_id, channel_id, bias_id):  
        """
        The aim of this function to get data from a few random neurons across the core to have an idea
        about the mean behaviour of the whole core for the given parameter.
        """
        
        raise NotImplementedError("This function does not work properly yet...")
        ##TODO: Sample these?
        neurons_to_check = [0, 50, 100, 150, 200, 250]
        
        if bias_id == 'tau1':
            full_tune_function = lambda neuron_id: self.full_tau1_tune(0.02, core_id, channel_id, neuron_id)
            check_value_function = lambda neuron_id: self.get_neuron_time_constant(core_id, channel_id, neuron_id)
        if bias_id == 'fast_exc':
            pass
            ##TODO add other biases
        
        for reference_neuron in neurons_to_check:
            self.bts.monitor(core_id, reference_neuron)
            scaled_bias, value = full_tune_function(reference_neuron)
            check_value_arr = []
                
            for remaining_neuron in neurons_to_check:
                self.bts.monitor(core_id, remaining_neuron)
                waveform, current_value = check_value_function(remaining_neuron)
                if current_value > 0 and current_value < 1:     #TODO: Improve the check for validity of the value.
                    check_value_arr.append(current_value)
            
            check_value_arr.append(value)
            check_value_arr = np.array(check_value_arr)
            
            #Add comparison to other measured values
            
            mid_neuron_id = np.argsort(abs(check_value_arr - check_value_arr.mean()))[0]
            break
        
        return neurons_to_check[mid_neuron_id]
                
        
    
    def get_all_core_values(self, channel_id : int, bias_tag : str):
        
        if bias_tag not in BIAS_TAGS:
            raise ValueError("Invalid bias tag")
        
        from pyvisa.errors import VisaIOError
        
        if bias_tag == 'tau1':
            get_value_function = lambda neuron_id: self.get_neuron_time_constant(neuron_id, channel_id)
        elif bias_tag == 'rfr':
            get_value_function = lambda neuron_id: self.get_rfr_period(neuron_id, channel_id)
        elif bias_tag == 'ampa_tau':
            get_value_function = lambda neuron_id: self.get_synapse_time_constant(channel_id, 'ampa')
        elif bias_tag == 'gaba_a_tau':
            get_value_function = lambda neuron_id: self.get_synapse_time_constant(channel_id, 'gaba_a')
        elif bias_tag == 'nmda_tau':
            get_value_function = lambda neuron_id: self.get_synapse_time_constant(channel_id, 'nmda')
        elif bias_tag == 'gaba_b_tau':
            get_value_function = lambda neuron_id: self.get_synapse_time_constant(channel_id, 'gaba_b')
        elif bias_tag[-2:] == '_w':
            get_value_function = lambda neuron_id: self.get_waveform_amplitude(channel_id)
            
        else:
            raise Exception("Error: Invalid bias_id")
            
        values = []
            
        for neuron_id in range(256):
            self.monitor_neuron(neuron_id)
            try:
                waveform, current_value = get_value_function(neuron_id)
            except (IndexError, VisaIOError):
                print("Warning: measurement error, trying again")
                try:
                    waveform, current_value = get_value_function(neuron_id)
                except (IndexError, VisaIOError):
                    print("Warning: measurement error, final attempt")
                    waveform, current_value = get_value_function(neuron_id)
                    
            print(neuron_id, current_value)
            if current_value > 1:
                print("**Warning: value greater than 1, trying to adapt channel offset")
                self.scope.find_offset(channel_id)
                waveform, current_value = get_value_function(neuron_id)
            values.append(current_value)
            
        return np.array(values)

        
            
    def get_lut(self, core_id : int, channel_id : int, neuron_id : int, bias_id : str):
        
        raise NotImplementedError("This function does not work properly yet...")
        
        if bias_id == 'tau1':
            scaled_bias_min, scaled_bias_max = self.tau1_range
            check_value_function = lambda n_id: self.get_neuron_time_constant(core_id, channel_id, n_id)
            set_bias_function = lambda scaled_bias: self.set_neuron_tau_bias(core_id, scaled_bias)
            
        elif bias_id == 'fast_exc':
            scaled_bias_min, scaled_bias_max = self.syn_tau_bias_range[3]
            check_value_function = lambda n_id: self.get_synapse_time_constant(core_id, channel_id, 3)
            set_bias_function = lambda scaled_bias: self.set_synapse_tau_bias(core_id, scaled_bias, 3)
        elif bias_id == 'fast_inh':
            scaled_bias_min, scaled_bias_max = self.syn_tau_bias_range[1]
            check_value_function = lambda n_id: self.get_synapse_time_constant(core_id, channel_id, 1)
            set_bias_function = lambda scaled_bias: self.set_synapse_tau_bias(core_id, scaled_bias, 1)
        elif bias_id == 'slow_exc':
            scaled_bias_min, scaled_bias_max = self.syn_tau_bias_range[2]
            check_value_function = lambda n_id: self.get_synapse_time_constant(core_id, channel_id, 2)
            set_bias_function = lambda scaled_bias: self.set_synapse_tau_bias(core_id, scaled_bias, 2)
        elif bias_id == 'slow_inh':
            scaled_bias_min, scaled_bias_max = self.syn_tau_bias_range[0]
            check_value_function = lambda n_id: self.get_synapse_time_constant(core_id, channel_id, 0)
            set_bias_function = lambda scaled_bias: self.set_synapse_tau_bias(core_id, scaled_bias, 0)
            
        else:
            raise Exception("Error: Unsupported bias_id")
        
        self.bts.monitor(core_id, neuron_id)
        
        lut = []
        
        for scaled_bias in np.arange(scaled_bias_min, scaled_bias_max, 0.01):
            set_bias_function(scaled_bias)
            trial_values = np.zeros(3)
            for attempt in range(3):
                waveform, value = check_value_function(neuron_id)
                trial_values[attempt] = value
                
            lut.append([trial_values.mean(), scaled_bias])
            
        return np.array(lut)
        
    
    def store_lut(self, core_id, bias_id):
        raise NotImplementedError("This function does not work properly yet...")
        pass

    def set_bias_from_lut(self, core_id, bias_id, value):
        
        raise NotImplementedError("This function does not work properly yet...")
        
        if str(core_id) in self.Info.cores:
            if bias_id in self.Info.cores[str(core_id)]:
                if 'fit' in self.Info.cores[str(core_id)][bias_id]:
                    scaled_bias = poly3(value, *self.Info.cores[str(core_id)][bias_id]['fit'])
                    if bias_id == 'tau1':                        
                        self.set_neuron_tau_bias(core_id, scaled_bias)
                    if bias_id == 'rfr':                        
                        self.set_neuron_rfr_bias(core_id, scaled_bias)
                    if bias_id == 'slow_inh':                        
                        self.set_synapse_tau_bias(core_id,scaled_bias,0)
                    if bias_id == 'fast_inh':                        
                        self.set_synapse_tau_bias(core_id,scaled_bias,1)
                    if bias_id == 'slow_exc':                        
                        self.set_synapse_tau_bias(core_id,scaled_bias,2)
                    if bias_id == 'fast_exc':                        
                        self.set_synapse_tau_bias(core_id,scaled_bias,3)
                
                else:
                   raise Exception("Error: LUT fit not available for bias %s of Core %d" % (bias_id, core_id))    
            else:
                raise Exception("Error: No data available for bias %s of Core %d" % (bias_id, core_id))   
        else:
            raise Exception("Error: No data available for core %d" % (core_id))       
        
    
    def get_ff_curve(self, freq_min_hz, freq_max_hz, freq_step_hz, measurement_period_s):
        
        #TODO add the code from the student lab
        
        pass
    

def get_tau_from_waveform(waveform, debug=None):
    """
    Adapted from Julian Buechel

    Parameters
    ----------
    waveform : np.array
        DESCRIPTION.
    debug : int, optional

    Returns
    -------
    fitted time constant

    """
    if debug:
        DEBUG=debug
    else:
        DEBUG = 0

    dt = 0.0002
    
    N_filter = int(0.002 / dt)
    ma_filter = np.ones((N_filter,))/N_filter
    
    N_skip = int(0.02/dt)
    
    neuron_trace=waveform
    
    N = neuron_trace.shape[0]
    neuron_trace = neuron_trace[N_skip:N-N_skip,:]
    v = neuron_trace[:,1]
    t = neuron_trace[:,0]
    t_min = t[0]
    t_max = t[-1]
    
    rising_times, falling_times = get_falling_times(neuron_trace, dt=0.0002, ma_filter=ma_filter, debug=DEBUG)
    falling_voltage, falling_time = get_falling_voltage_and_time(v, t, falling_times, rising_times)
    
    tau = get_tau(falling_voltage, falling_time)
    duration = falling_time[-1]-falling_time[0]
    t_fit = np.linspace(0,duration,len(falling_time))
    B = np.min(falling_voltage)
    A = np.max(falling_voltage) - B
    f_fit = A*np.exp(-t_fit/tau) + B
    
    if DEBUG > 0:
        plt.figure()
        plt.plot(waveform[:,0],waveform[:,1])
        plt.plot(t_fit,f_fit)
        plt.xlabel("time, s")
        plt.ylabel("voltage, V")
        plt.title()
    
    return tau


def find_mid_distribution_index(values):
    return np.argsort(abs(values - values.mean()))[0]


def find_mid_distribution_indices(values, max_delta):
    """
    Find the indices of the elements within the given sigma from the mean

    Parameters
    ----------
    values : np.array
        Input array of distributed values
    max_sigma : TYPE
        Maximum distance of the element from the mean of the distribution allowed

    Returns
    -------
    Numpy array of neuron ids within max_sigma width
    """
    
    deltas = abs(values - values.mean())
    
    return np.array([index for index, delta in enumerate(deltas) if delta<max_delta])


def detect_peak_to_peak(waveform):
    y_values = waveform[:,1]
    return np.max(y_values) - np.min(y_values)

def fit_with_sigmoid(data):
    start = int(len(data)*0.1)
    end = int(len(data)*0.9)
    popt, pcov = curve_fit(sigmoid, 1-data[start:end,1], data[start:end,0], method='dogbox')
    return popt

def sigmoid(x, a, b):
    return 1.0 / (1.0 + np.exp(-a*(x-b)))

def inverse_sigmoid(x, a, b):
    return np.log(x/(1-x))/a + b

def inverse_sigmoid_ext(x, a, b, c):
    x = c*x
    y= 1 - np.log(x/(1-x))/a - b
    return y

def poly3(x, a, b, c, d):
    return a*x*x*x + b*x*x + c*x + d

    
if __name__ == '__main__':
    
    net_gen = netgen.NetworkGenerator()
    DTC = DynapseTuningCompanion(model, net_gen)

    