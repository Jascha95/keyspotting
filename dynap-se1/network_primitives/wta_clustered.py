#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 17:42:35 2019

@author: dzenn
"""

import sys
#sys.path.append("/home/dzenn/Downloads/samna/ctxctl_contrib/")

from details import population_connection_details as ict
import numpy as np
from random import sample, random
import warnings
import netgen
import samna
from netgen import Neuron
import dynapse1utils as ut


from time import sleep, time


class WTAGroup():
    """
        A class to stimulate and record activity
        of an EI population of silicon neurons.
        
        The class connects with CTXCTL with RPyC.
        
        Available modes are:
            'bump' : one-dimensional Winner-take-all connectivity (local exc, global inh),
                    input is presented as a gaussian bump
            'fi' : no connections created, neurons are only monitored,
                    input is presented by the DC current bias
            'ff' : only connections from the spikegenerator are created,
                    input is presented by the Poisson spikegen uniformly to all neurons
            'core_ff' : connects spikegenerator to the whole core
            'single_EI_cluster' :
            'clustered_WTA' :
            'global_exc_WTA' :
        
    """
    
    def __init__(self, model, start_neuron, exc_size, inh_size, chip_id, core_id, mode, core_id_inh=None, ext_netgen=None,
                 exc_cluster_size=1,
                 inh_cluster_size=1,
                 connectivity={'wta_structure': 'normal',
                               'ee_recurrent': 0.2,
                               'ee_lateral':1.0,
                               'ee_lateral_2nd':0.4,
                               'ee_lateral_3rd':0.1,
                               'ei': 0.2,
                               'ie': 0.2,
                               'ii': 0.0,
                               'ee_global':0.2,
                               'ei_global':0.2,
                               'exc_global_size': 0,
                               'start_neuron_inh' : None,
                               'edge_wrap_around' : True}, 
                 input_multiplier=10,
                 clustered_input = False, allow_self_exc = False, debug = False):
        """
        Args:
            start_neuron (int) : index of the first neuron of the population
            exc_size (int) : size of the exc population
            chip_id (int) : [0..3] chip to place the population on
            core_id (int) : [0..3] core to place the population on
            mode (str) : a mode to run the population in ('bump', 'fi', 'ff' or 'core_ff', 'single_ei_cluster')
            netgen : NetworkGenerator
            exc_cluster_size : int
            allow_self_exc (bool): 
            debug (bool, optional) : debug prints
            model (samna model): 
        """
        
        expected_cams = check_cams_per_neuron_clustered_wta(exc_cluster_size=exc_cluster_size,
                                                            inh_size=inh_size,
                                                            num_clusters=exc_size/exc_cluster_size,
                                                            connectivity=connectivity,
                                                            input_multiplier=input_multiplier,
                                                            print_detailed = True)
        assert exc_size%exc_cluster_size==0, "ERROR: Cannot divide the excitatory population into clusters of specified size. Adjust configuration"
        assert inh_size%inh_cluster_size==0, "ERROR: Cannot divide the inhibitory population into clusters of specified size. Adjust configuration"
        if mode=="global_exc":
            assert exc_size/exc_cluster_size==inh_size/inh_cluster_size, "ERROR: Number of excitatory and inhibitory clusters has to match. Adjust configuration"
            assert connectivity['exc_global_size'] + inh_size < 256, "ERROR: Global EXC population and INH clusters do not fit on the core of 256 neurons. Adjust configuration"
        assert start_neuron + exc_size <= 256, "ERROR: Excitatory population exceeds core size."
        assert expected_cams['exc_neuron'] <= 64, "ERROR: Excitatory neurons running out of CAMs (%d > 64). Adjust configuration" % expected_cams['exc_neuron']
        assert expected_cams['inh_neuron'] <= 64, "ERROR: Inhibitory neurons running out of CAMs (%d > 64). Adjust configuration" % expected_cams['inh_neuron']
        
        self.start_neuron = start_neuron
        self.start_neuron_inh = connectivity['start_neuron_inh']
        if self.start_neuron_inh is None:
            self.start_neuron_inh = self.start_neuron
        self.dummy_neuron_id = 1
        self.exc_size = exc_size
        self.chip_id = chip_id
        self.core_id = core_id
        self.core_id_inh = core_id_inh
        if core_id_inh is None:
            self.core_id_inh = core_id+1
        self.allow_self_exc = allow_self_exc
        self.debug = debug
        self.inh_size = inh_size
        self.exc_cluster_size = exc_cluster_size
        self.num_clusters = exc_size//exc_cluster_size
        self.inh_cluster_size = inh_cluster_size
        self.mode = mode
        self.wrap_around = connectivity['edge_wrap_around']
        
        ADDR_NUM_BITS = 15
        ISI_NUM_BITS = 16
        self.max_fpga_len = 2**ADDR_NUM_BITS-1
        self.max_isi = 2**ISI_NUM_BITS-1
        self.base_isi = 900
        
        self.connectivity_matrix = np.zeros((1024+4096,4096))
        self.clustered_input = clustered_input
        if self.clustered_input:
            self.cluster_rates = []
        
        
        self.model = model
        print("model ok")
        if ext_netgen is not None:
            self.netgen = ext_netgen
        else:
            self.netgen = netgen.NetworkGenerator()
        self.empty_configuration = self.netgen.make_dynapse1_configuration()

        print("netgen ok")
        self.SynTypes = samna.dynapse1.Dynapse1SynType
        print("SynTypes ok")

        
        # self.neurons = np.array(Neuron(get_shadow_state_neurons()
        self.v_neurons = np.array([Neuron(0, neuron_id//256, neuron_id%256, is_spike_gen=True) for neuron_id in np.arange(1024)])
        # self.bias_group = self.model.get_bias_groups()[chip_id*4 + core_id]
        
        if debug:
            print("Neurons ok")
            
        self.exc_population_ids = np.array([[self.chip_id, self.core_id, n_id] for n_id in np.arange(self.start_neuron, self.start_neuron + self.exc_size)])
        self.exc_population = np.array([Neuron(*neuron_id) for neuron_id in self.exc_population_ids])
        
        self.inh_population_ids = np.array([[self.chip_id, self.core_id_inh, n_id] for n_id in np.arange(self.start_neuron_inh, self.start_neuron_inh + self.inh_size)])
        self.inh_population = np.array([Neuron(*neuron_id) for neuron_id in self.inh_population_ids])
        
        self.global_exc_population_ids = []
        
        if self.mode == 'global_exc':
            self.global_exc_population_ids = np.array([[self.chip_id, self.core_id_inh, n_id] for n_id in np.arange(256 - int(inh_size) - int(connectivity['exc_global_size']), 256 - int(inh_size))])
            self.global_exc_population = np.array([Neuron(*neuron_id) for neuron_id in self.global_exc_population_ids])
        
        if debug:
            print("Populations ok")
        
        self.pop_id_lookup = {}
        for idx in range(len(self.exc_population_ids)):
            self.pop_id_lookup[self.exc_population_ids[idx][2]] = idx
        
        if self.mode != 'fi':
            if self.mode == 'bump':
                connect_populations(self.netgen, self.exc_population, self.exc_population,
                                    ict.neighbors1d_inner_i(self.exc_size) + ict.neighbors1d_inner_i(self.exc_size) + ict.neighbors1d_inner_i(self.exc_size) + ict.neighbors1d_outer_i(self.exc_size),
                                    ict.neighbors1d_inner_j(self.exc_size) + ict.neighbors1d_inner_j(self.exc_size) + ict.neighbors1d_inner_j(self.exc_size) + ict.neighbors1d_outer_j(self.exc_size), self.SynTypes.AMPA)
                
    #            print(16*0.7/self.exc_size)
    #            arr_i, arr_j = ict.set_all_to_all(len(self.population), len(self.inh_population), p = np.clip(16*0.7/self.exc_size, 0, 1))
    #            connect_populations(self.netgen, self.population, self.inh_population, arr_i, arr_j, syn_type=self.SynTypes.AMPA)
    #            
    #            print(0.7*8/(self.inh_fraction*self.exc_size))
    #            arr_i, arr_j = ict.set_all_to_all(len(self.inh_population), len(self.population), p = np.clip(0.7*8/(self.inh_fraction*self.exc_size), 0, 1))
    #            connect_populations(self.netgen, self.inh_population, self.population, arr_i, arr_j, syn_type=self.SynTypes.GABA_A)
                
    
                arr_i, arr_j = ict.set_all_to_all(len(self.exc_population), len(self.inh_population), p = connectivity['ei'])
                connect_populations(self.netgen, self.exc_population, self.inh_population, arr_i, arr_j, syn_type=self.SynTypes.AMPA)
                
    
                arr_i, arr_j = ict.set_all_to_all(len(self.inh_population), len(self.exc_population), p = connectivity['ie'])
                connect_populations(self.netgen, self.inh_population, self.exc_population, arr_i, arr_j, syn_type=self.SynTypes.GABA_A)
                
            if self.mode == 'single_ei_cluster':
                arr_i, arr_j = ict.set_all_to_all(len(self.exc_population), len(self.inh_population), p = connectivity['ei'])
                connect_populations(self.netgen, self.exc_population, self.inh_population, arr_i, arr_j, syn_type=self.SynTypes.AMPA)
                
    
                arr_i, arr_j = ict.set_all_to_all(len(self.inh_population), len(self.exc_population), p = connectivity['ie'])
                connect_populations(self.netgen, self.inh_population, self.exc_population, arr_i, arr_j, syn_type=self.SynTypes.GABA_A)
                
                arr_i, arr_j = ict.set_all_to_all(len(self.exc_population), len(self.exc_population), p = connectivity['ee_recurrent'])
                connect_populations(self.netgen, self.exc_population, self.exc_population, arr_i, arr_j, self.SynTypes.AMPA)
                
            if self.mode == 'clustered_WTA' or self.mode == 'global_exc':
                
                if self.debug:
                    print("Connecting clusters...")
                
                self.exc_cluster_neurons = []
                self.exc_cluster_neuron_ids = []
                self.inh_cluster_neurons = []
                self.inh_cluster_neuron_ids = []
                
                
                for cluster_id in range(self.num_clusters):
                    self.exc_cluster_neuron_ids.append(self.exc_population_ids[cluster_id*self.exc_cluster_size:(cluster_id+1)*self.exc_cluster_size])
                    self.exc_cluster_neurons.append(self.exc_population[cluster_id*self.exc_cluster_size:(cluster_id+1)*self.exc_cluster_size])
                    
                    if self.mode == 'global_exc':
                        self.inh_cluster_neuron_ids.append(self.inh_population_ids[cluster_id*self.inh_cluster_size:(cluster_id+1)*self.inh_cluster_size])
                        self.inh_cluster_neurons.append(self.inh_population[cluster_id*self.inh_cluster_size:(cluster_id+1)*self.inh_cluster_size])
                
                ## Connecting clusters
                
                for cluster_id in range(self.num_clusters):
                    
                    # Recurrent conncetions within clusters
                    if self.debug:
                        print("Connecting cluster %d to cluster %d with p=%.2f" % (cluster_id, cluster_id, connectivity['ee_recurrent']))
                    
                    arr_i, arr_j = ict.set_all_to_all(exc_cluster_size, exc_cluster_size, p = connectivity['ee_recurrent'])
                    connect_populations(self.netgen,
                                        self.exc_cluster_neurons[cluster_id],
                                        self.exc_cluster_neurons[cluster_id],
                                        arr_i, arr_j, self.SynTypes.AMPA)
                    
                    # Inhibitory clusters:
                    if self.mode == 'global_exc':
                        # EXC to INH connections
                        if self.debug:
                            print("Connecting cluster %d to INH cluster %d with p=%.2f" % (cluster_id, cluster_id, connectivity['ei']))
                        arr_i, arr_j = ict.set_all_to_all(exc_cluster_size, inh_cluster_size, p = connectivity['ei'])
                        connect_populations(self.netgen,
                                        self.exc_cluster_neurons[cluster_id],
                                        self.inh_cluster_neurons[cluster_id],
                                        arr_i, arr_j, self.SynTypes.AMPA)
                        
                        # INH to EXC connections
                        if self.debug:
                            print("Connecting INH cluster %d to cluster %d with p=%.2f" % (cluster_id, cluster_id, connectivity['ie']))
                        arr_i, arr_j = ict.set_all_to_all(inh_cluster_size, exc_cluster_size, p = connectivity['ie'])
                        connect_populations(self.netgen,
                                        self.inh_cluster_neurons[cluster_id],
                                        self.exc_cluster_neurons[cluster_id],
                                        arr_i, arr_j, self.SynTypes.GABA_B)
                        
                        # INH to INH connections
                        if self.debug:
                            print("Connecting INH cluster %d to INH cluster %d with p=%.2f" % (cluster_id, cluster_id, connectivity['ii']))
                        arr_i, arr_j = ict.set_all_to_all(inh_cluster_size, inh_cluster_size, p = connectivity['ii'])
                        connect_populations(self.netgen,
                                        self.inh_cluster_neurons[cluster_id],
                                        self.inh_cluster_neurons[cluster_id],
                                        arr_i, arr_j, self.SynTypes.GABA_B)
                        
                        # EXC cluster to global EXC population connections
                        if self.debug:
                            print("Connecting EXC cluster %d to global EXC population with p=%.2f" % (cluster_id, connectivity['ee_global']))
                        arr_i, arr_j = ict.set_all_to_all(exc_cluster_size, connectivity['exc_global_size'], p = connectivity['ee_global'])
                        connect_populations(self.netgen,
                                        self.exc_cluster_neurons[cluster_id],
                                        self.global_exc_population,
                                        arr_i, arr_j, self.SynTypes.NMDA)
                        
                        # Global EXC population to INH cluster connections
                        if self.debug:
                            print("Connecting global EXC population to INH cluster %d with p=%.2f" % (cluster_id, connectivity['ei_global']))
                        arr_i, arr_j = ict.set_all_to_all(connectivity['exc_global_size'], inh_cluster_size, p = connectivity['ei_global'])
                        connect_populations(self.netgen,
                                        self.global_exc_population,
                                        self.inh_cluster_neurons[cluster_id],
                                        arr_i, arr_j, self.SynTypes.AMPA)
                        
                        
                    
                    # Lateral connections to the first neighbour to the right
                    next_cluster_id = cluster_id+1
                    if self.wrap_around:
                        next_cluster_id = next_cluster_id%self.num_clusters
                    else:
                        next_cluster_id = np.clip(next_cluster_id, 0, self.num_clusters-1)
                    if self.debug:
                        print("Connecting cluster %d to cluster %d with p=%.2f" % (cluster_id, next_cluster_id, connectivity['ee_lateral']))
                    arr_i, arr_j = ict.set_all_to_all(exc_cluster_size, exc_cluster_size, p = connectivity['ee_lateral'])
                    connect_populations(self.netgen,
                                        self.exc_cluster_neurons[cluster_id],
                                        self.exc_cluster_neurons[next_cluster_id],
                                        arr_i, arr_j, self.SynTypes.AMPA)
                    
                    # Lateral connections to the second neighbour to the right
                    next_cluster_id = (cluster_id+2)%self.num_clusters
                    if self.debug:
                        print("Connecting cluster %d to cluster %d with p=%.2f" % (cluster_id, next_cluster_id, connectivity['ee_lateral_2nd']))
                    arr_i, arr_j = ict.set_all_to_all(exc_cluster_size, exc_cluster_size, p = connectivity['ee_lateral_2nd'])
                    connect_populations(self.netgen,
                                        self.exc_cluster_neurons[cluster_id],
                                        self.exc_cluster_neurons[next_cluster_id],
                                        arr_i, arr_j, self.SynTypes.AMPA)
                    
                    # Lateral connections to the third neighbour to the right
                    next_cluster_id = (cluster_id+3)%self.num_clusters
                    if self.debug:
                        print("Connecting cluster %d to cluster %d with p=%.2f" % (cluster_id, next_cluster_id, connectivity['ee_lateral_3rd']))
                    arr_i, arr_j = ict.set_all_to_all(exc_cluster_size, exc_cluster_size, p = connectivity['ee_lateral_3rd'])
                    connect_populations(self.netgen,
                                        self.exc_cluster_neurons[cluster_id],
                                        self.exc_cluster_neurons[next_cluster_id],
                                        arr_i, arr_j, self.SynTypes.AMPA)
                    
                    # Lateral connections to the first neighbour to the left
                    prev_cluster_id = cluster_id-1
                    if self.wrap_around:
                        prev_cluster_id = (self.num_clusters+next_cluster_id)%self.num_clusters
                    else:
                        prev_cluster_id = np.clip(prev_cluster_id, 0, self.num_clusters-1)
                    if self.debug:
                        print("Connecting cluster %d to cluster %d with p=%.2f" % (cluster_id, prev_cluster_id, connectivity['ee_lateral']))
                    arr_i, arr_j = ict.set_all_to_all(exc_cluster_size, exc_cluster_size, p = connectivity['ee_lateral'])
                    connect_populations(self.netgen,
                                        self.exc_cluster_neurons[cluster_id],
                                        self.exc_cluster_neurons[prev_cluster_id],
                                        arr_i, arr_j, self.SynTypes.AMPA)
                    
                    # Lateral connections to the second neighbour to the left
                    prev_cluster_id = (self.num_clusters+cluster_id-2)%self.num_clusters
                    if self.debug:
                        print("Connecting cluster %d to cluster %d with p=%.2f" % (cluster_id, prev_cluster_id, connectivity['ee_lateral_2nd']))
                    arr_i, arr_j = ict.set_all_to_all(exc_cluster_size, exc_cluster_size, p = connectivity['ee_lateral_2nd'])
                    connect_populations(self.netgen,
                                        self.exc_cluster_neurons[cluster_id],
                                        self.exc_cluster_neurons[prev_cluster_id],
                                        arr_i, arr_j, self.SynTypes.AMPA)
                    
                    # Lateral connections to the third neighbour to the left
                    prev_cluster_id = (self.num_clusters+cluster_id-3)%self.num_clusters
                    if self.debug:
                        print("Connecting cluster %d to cluster %d with p=%.2f" % (cluster_id, prev_cluster_id, connectivity['ee_lateral_3rd']))
                    arr_i, arr_j = ict.set_all_to_all(exc_cluster_size, exc_cluster_size, p = connectivity['ee_lateral_3rd'])
                    connect_populations(self.netgen,
                                        self.exc_cluster_neurons[cluster_id],
                                        self.exc_cluster_neurons[prev_cluster_id],
                                        arr_i, arr_j, self.SynTypes.AMPA)
                    
                        
                
                ## Connecting inhibitory popultaion
                if self.mode != 'global_exc':
                    if self.debug:
                        print("Creating EI connections with p=%.2f" % connectivity['ei'])
                    
                    arr_i, arr_j = ict.set_all_to_all(len(self.exc_population), len(self.inh_population), p = connectivity['ei'])
                    connect_populations(self.netgen, self.exc_population, self.inh_population, arr_i, arr_j, syn_type=self.SynTypes.AMPA)
                    
                    if self.debug:
                        print("Creating IE connections with p=%.2f" % connectivity['ie'])
        
                    arr_i, arr_j = ict.set_all_to_all(len(self.inh_population), len(self.exc_population), p = connectivity['ie'])
                    connect_populations(self.netgen, self.inh_population, self.exc_population, arr_i, arr_j, syn_type=self.SynTypes.GABA_B)
                    
                    if self.debug:
                        print("Creating II connections with p=%.2f" % connectivity['ii'])
        
                    arr_i, arr_j = ict.set_all_to_all(len(self.inh_population), len(self.inh_population), p = connectivity['ii'])
                    connect_populations(self.netgen, self.inh_population, self.inh_population, arr_i, arr_j, syn_type=self.SynTypes.GABA_B)
            
                
                
                
            if debug is True:
                print('Connecting input...')
                
            self.input_ids = []
        
            if self.mode == 'core_ff' or self.mode == 'single_ei_cluster':
                self.input_ids.append(1)
                for id_triplet, neuron in zip(self.exc_population_ids, self.exc_population):
                    chip_id, core_id, n_id = id_triplet
                    idx = chip_id*1024 + core_id*256 + n_id                                  
                    for i in range(input_multiplier):
                        if debug:
                            print("Creating connection input %d to %d" %(self.input_ids[-1], idx))
                        self.netgen.add_connection(self.v_neurons[self.input_ids[-1]], neuron, self.SynTypes.NMDA)
            else:         
                for id_triplet, neuron in zip(self.exc_population_ids, self.exc_population):
                    chip_id, core_id, n_id = id_triplet
                    idx = chip_id*1024 + core_id*256 + n_id
                    if clustered_input:
                        id_to_set = (idx//self.exc_cluster_size)%1024
                    else:
                        id_to_set = (idx)%1024
                    if self.allow_self_exc:
                        self.input_ids.append(id_to_set)
                    else:
                        self.input_ids.append(1023-id_to_set)
                    for i in range(input_multiplier):
                        if debug:
                            print("Creating connection input %d to %d" %(self.input_ids[-1], idx))
                        self.netgen.add_connection(self.v_neurons[self.input_ids[-1]], neuron, self.SynTypes.NMDA)
                
                if clustered_input:
                    self.input_ids = np.array(self.input_ids)[::exc_cluster_size]
                self.num_input_neurons = len(self.input_ids)
                
        self.new_config = self.netgen.make_dynapse1_configuration()
        ut.apply_biases_from_configuration(self.model.get_configuration(), self.new_config)
        self.model.apply_configuration(self.new_config)
        if debug is True:
            print('Connections created successfully')
        
        
        ## Constructing the filtering graph
        
        # print(self.exc_population_ids[:,2], self.inh_population_ids[:,2], self.global_exc_population_ids[:,2])
        if len(self.global_exc_population_ids):
            print(np.concatenate((self.exc_population_ids[:,2], self.inh_population_ids[:,2], self.global_exc_population_ids[:,2])))
            neurons_to_set = np.concatenate((self.exc_population_ids, self.inh_population_ids, self.global_exc_population_ids))
        elif len(self.inh_population_ids):
            print(np.concatenate((self.exc_population_ids[:,2], self.inh_population_ids[:,2])))
            neurons_to_set = np.concatenate((self.exc_population_ids, self.inh_population_ids))
        else:
            print(self.exc_population_ids[:,2])
            neurons_to_set = self.exc_population_ids
        
        self.graph, self.filter_node, self.sink_node = ut.create_neuron_select_graph(self.model, neurons_to_set)
        self.graph_raster, self.filter_node_raster, self.running_raster_sink_node = ut.create_neuron_select_graph(self.model, neurons_to_set)
        
        self.graph.start()
        self.graph_raster.start()

              
        self.poisson_spike_gen = self.model.get_poisson_gen()
        self.poisson_spike_gen.set_chip_id(self.chip_id)
        self.regular_spike_gen = self.model.get_fpga_spike_gen()     
        for i in range(1024):
            self.poisson_spike_gen.write_poisson_rate_hz(i,0)
            
        self.rates = [0 for i in range(self.exc_size)]
            
        if debug is True:
            print('Init complete')
        
        
    def get_rates(self, measurement_period = 1, spike_limit = None, keep_buffer=False, return_spikes=False, debug=False):
        """
            Get firing rates integrated over the measurement period
        """
        
        start_time = time()
        if not keep_buffer:
            evts = self.sink_node.get_events()
            sleep(measurement_period)
        end_time = time()
        evts = self.sink_node.get_events()
        recorded_evts=[]
        self.rates = np.array([0 for i in range(self.exc_size)])
        self.inh_rates = np.array([0 for i in range(self.inh_size)])
        
        if spike_limit is None:
            spike_limit=len(evts)
        else:
            start_time = evts[0].timestamp/1e+06
            end_time = evts[min(spike_limit, len(evts)-1)].timestamp/1e+06
            
        if debug:
            print("Evts length: %d, spike_limit: %d, spikes to filter: %d" % (len(evts), spike_limit, len(evts[0:spike_limit])))
        
        for evt in evts[0:spike_limit]:
            if evt.neuron_id in self.exc_population_ids[:,2]:
                neuron_id = evt.neuron_id - self.start_neuron
                self.rates[neuron_id] += 1
                if return_spikes:
                    recorded_evts.append((evt.timestamp, neuron_id))
            if len(self.inh_population_ids):
                if evt.neuron_id in self.inh_population_ids[:,2]:
                    neuron_id = evt.neuron_id - self.inh_population_ids[0,2]
                    self.inh_rates[neuron_id] += 1
                
        self.rates = self.rates/(end_time - start_time)
        
        if self.exc_cluster_size>1:
            self.cluster_rates=[]
            self.cluster_sd=[]
            for cluster_id in range(self.exc_size//self.exc_cluster_size):
                self.cluster_rates.append(np.mean(self.rates[cluster_id*self.exc_cluster_size:(cluster_id+1)*self.exc_cluster_size]))
                self.cluster_sd.append(np.std(self.rates[cluster_id*self.exc_cluster_size:(cluster_id+1)*self.exc_cluster_size]))
        
        self.expected_value = pop_code2double(self.true_rates)
        if self.exc_cluster_size>1:
            self.estimated_value = pop_code2double(self.cluster_rates)
        else:
            self.estimated_value = pop_code2double(self.rates)
            
        if debug:
            print("Enc: %g, Dec: %g" % (self.expected_value, self.estimated_value))
            
        if return_spikes:
            return np.array(recorded_evts)
        
        
    def set_bump_input(self, value, sigma = None, distortion = 0., amplitude=100, noise=0.0):
        """
            Set bump input
        """
        if sigma is None:
            sigma = 1/(len(self.input_ids))
            
        self.input_amplitude = amplitude
        
        if self.clustered_input:
            self.true_rates = double2pop_code(value, self.num_input_neurons, sigma, amplitude)
            self.input_rates = self.true_rates
        else:
            self.true_rates = double2pop_code(value, self.num_input_neurons//self.exc_cluster_size, sigma, amplitude)
            self.input_rates = np.repeat(self.true_rates, self.exc_cluster_size)
        self.value = value
        
        if distortion != 0:
            ids = sample(range(len(self.input_rates)), int(len(self.input_rates)*distortion))
            for idx in ids:
                self.input_rates[idx] = 0
                
        if noise != 0.0:
            for idx in range(len(self.input_rates)):
                self.input_rates[idx] += self.input_rates[idx]*noise*np.random.normal()
        
        for idx, rate in zip(self.input_ids, self.input_rates):
            self.poisson_spike_gen.write_poisson_rate_hz(idx,rate)
        self.value = value
        if self.poisson_spike_gen.is_running() is False:
            self.poisson_spike_gen.start()
            
            
    def set_regular_bump_input(self, value, sigma = None, distortion = 0., amplitude=100):
        """
            Set bump input with regular rates
        """
        
        if self.exc_size > int((1/amplitude)*1e+06):
            warnings.warn('Spike frequency might be irregular because maximum rate is too high for the given number of neurons')
        
        if self.poisson_spike_gen.is_running():
            self.poisson_spike_gen.stop()
        
        # self.regular_spike_gen.stop()
        
        if sigma is None:
            sigma = 1/(self.num_input_neurons//self.exc_cluster_size)
            
        self.input_amplitude = amplitude
        
        if self.clustered_input:
            self.true_rates = double2pop_code(value, self.num_input_neurons, sigma, amplitude)
            self.input_rates = self.true_rates
        else:
            self.true_rates = double2pop_code(value, self.num_input_neurons//self.exc_cluster_size, sigma, amplitude)
            self.input_rates = np.repeat(self.true_rates, self.exc_cluster_size)
        self.value = value
        
        if distortion != 0:
            ids = sample(range(self.exc_size), int(self.exc_size*distortion))
            for idx in ids:
                self.input_rates[idx] = 0
                
        ### Prepare spiketimes of regular rates per neuron
        spiketimes = {}
        
        for neuron_id in range(len(self.true_rates)):
            if self.true_rates[neuron_id] > 1.0:
                spiketimes[neuron_id] = np.arange(neuron_id, 1e+06, int((1/self.true_rates[neuron_id])*1e+06))
            
        ### Compile spikes to a single array
        output_events = []
        
        for neuron_id in spiketimes.keys():
            for timestamp in spiketimes[neuron_id]:
                output_events.append([timestamp, self.input_ids[neuron_id]])
                
        output_events = np.array(output_events)
        output_events = output_events[output_events[:,0].argsort()]
        
        if output_events[0,0] != 0:
            output_events = np.insert(output_events, 0, [0, self.dummy_neuron_id], axis = 0)  ## Add first reference spike if first event is not at time 0
            
            
        ### Insert dummy spikes if some ISIs are greater that max_isi
        tmp_id = 1
        while tmp_id < len(output_events):
            if output_events[tmp_id,0] - output_events[tmp_id-1,0] > self.max_isi:
                output_events = np.insert(output_events, tmp_id, [output_events[tmp_id-1,1]+self.max_isi-1, self.dummy_neuron_id], axis = 0)
            tmp_id += 1
        
        self.output_events = output_events
        
        self.set_preloaded_stimulus(output_events)
        
        ### Start the spikegen
        self.value = value
        self.regular_spike_gen.start()
        
        
            
    def set_preloaded_stimulus(self, spike_train, verbose=False):
        
        if self.poisson_spike_gen.is_running():
            self.poisson_spike_gen.stop()
        
        self.regular_spike_gen.stop()
                
        ut.set_fpga_spike_gen(self.regular_spike_gen,
                              spike_times=spike_train[:,0]*1e-06,
                              indices=spike_train[:,1].astype(int),
                              target_chips=[self.chip_id]*len(spike_train),
                              isi_base=self.base_isi,
                              repeat_mode=True)
        
        if verbose:
            print("Stimulus loaded")
                
            
    def set_double_bump_input(self, value1, value2, sigma = None, distortion = 0., amplitude1=100, amplitude2=80):
        """
            Set double bump input
        """
        if sigma is None:
            sigma = 1/(self.num_input_neurons//self.exc_cluster_size)
            
        self.input1_amplitude = amplitude1
        self.input2_amplitude = amplitude2
        
        if self.clustered_input:
            self.true_rates = np.maximum(double2pop_code(value1, self.num_input_neurons, sigma, amplitude1),
                            double2pop_code(value2, self.num_input_neurons, sigma, amplitude2))
            self.input_rates = self.true_rates
        else:
            self.true_rates = np.maximum(double2pop_code(value1, self.num_input_neurons//self.exc_cluster_size, sigma, amplitude1),
                            double2pop_code(value2, self.num_input_neurons//self.exc_cluster_size, sigma, amplitude2))
            self.input_rates = np.repeat(self.true_rates, self.exc_cluster_size)
        
        if distortion != 0:
            ids = sample(range(self.exc_size), int(self.exc_size*distortion))
            for idx in ids:
                self.input_rates[idx] = 0
        
        for idx, rate in zip(self.input_ids, self.input_rates):
            self.poisson_spike_gen.write_poisson_rate_hz(idx,rate)
        self.value1 = value1
        self.value2 = value2
        if self.poisson_spike_gen.is_running() is False:
            self.poisson_spike_gen.start()
            
            
    def set_square_input(self, value, width = None, distortion = 0., amplitude = 100):
        """
            Set square input
            
            Args:
                value (float) : from 0 to 1, square input center position
                width (float) : from 0 to 1, width of the input shape in value range
                distortion (float) : form 0 to 1, fraction of neurons to be randomly
                                     shut off to test signal restoration
        """
        if width is None:
            width = 1/self.exc_size
        
        self.true_rates = np.zeros(self.exc_size)
        
        for index in range(self.exc_size):
            if abs(index/self.exc_size - value)%1 <= width/2 or \
                abs(index/self.exc_size - value - 1)%1 <= width/2:
                self.true_rates[index] = amplitude
        
        if distortion != 0:
            ids = sample(range(self.exc_size), int(self.exc_size*distortion))
            for idx in ids:
                self.true_rates[idx] = 0
        
        for idx, rate in zip(self.input_ids, self.true_rates):
            self.poisson_spike_gen.write_poisson_rate_hz(idx,rate)
        self.value = value
        if self.poisson_spike_gen.is_running() is False:
            self.poisson_spike_gen.start()
            
            
    def set_freq_input(self, freq):
        """
            Set uniform spike input
        """
        self.true_rates = [freq for idx in self.input_ids]
        for idx, rate in zip(self.input_ids, self.true_rates):
            self.poisson_spike_gen.write_poisson_rate_hz(idx, rate)
        self.freq = freq
        if self.poisson_spike_gen.is_running() is False:
            self.poisson_spike_gen.start()
            
    def set_single_neuron_input(self, neuron_id, freq):
        """
            Set uniform spike input
        """
        self.true_rates = [0 for idx in self.input_ids]
        if self.clustered_input==False:
            for idx in range(self.exc_cluster_size):
                self.true_rates[neuron_id*self.exc_cluster_size+idx] = freq
        else:
            self.true_rates[neuron_id] = freq
        for idx, rate in zip(self.input_ids, self.true_rates):
            self.poisson_spike_gen.write_poisson_rate_hz(idx,rate)
        self.freq = freq
        if self.poisson_spike_gen.is_running() is False:
            self.poisson_spike_gen.start()
            
    def set_dc_input(self, fine_value):
        """
            Set core dc input
        """
        self.bias_group.set_bias("IF_DC_P", fine_value, 7)
            
            

def check_cams_per_neuron_clustered_wta(exc_cluster_size : int, inh_size : int, num_clusters : int,
                                        connectivity={'wta_structure': 'normal',
                                                       'ee_recurrent': 0.2,
                                                       'ee_lateral':1.0,
                                                       'ee_lateral_2nd':0.4,
                                                       'ee_lateral_3rd':0.1,
                                                       'ei': 0.2,
                                                       'ie': 0.2,
                                                       'ii': 0.0,
                                                       'ee_global':0.2,
                                                       'ei_global':0.2,
                                                       'exc_global_size': 0},
                                        input_multiplier=10,
                                        print_detailed=False):
    """
    Checks the number of expected inputs (CAMs) per neuron in every population.

    Parameters
    ----------
    exc_cluster_size : int
        DESCRIPTION.
    inh_size : int
        DESCRIPTION.
    num_clusters : int
        DESCRIPTION.
    connectivity : dict, optional
        DESCRIPTION. 
    input_multiplier : int, optional
        DESCRIPTION. The default is 10.
    print_detailed : bool, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    dict
        DESCRIPTION.

    """
        
    if connectivity['wta_structure'] == 'global_exc':
        
        exc_neuron_cams_dict = {'input' : input_multiplier,
                                'ee_recurrent' : int(exc_cluster_size*connectivity['ee_recurrent']),
                                'ee_lateral' : int(2*exc_cluster_size*connectivity['ee_lateral']) + int(2*exc_cluster_size*connectivity['ee_lateral_2nd']) + int(2*exc_cluster_size*connectivity['ee_lateral_3rd']),
                                'ie' : int(inh_size/num_clusters*connectivity['ie'])
                                }
        exc_neuron_cams = exc_neuron_cams_dict['input'] + exc_neuron_cams_dict['ee_recurrent'] + exc_neuron_cams_dict['ee_lateral'] + exc_neuron_cams_dict['ie']
        
        inh_neuron_cams_dict = {'exc_cluster' : int(exc_cluster_size*connectivity['ei']),
                                'ii_recurrent' : int(inh_size/num_clusters*connectivity['ii']),
                                'ei_global' : int(connectivity['exc_global_size']*connectivity['ei_global'])
                                }
        inh_neuron_cams = inh_neuron_cams_dict['exc_cluster'] + inh_neuron_cams_dict['ii_recurrent'] + inh_neuron_cams_dict['ei_global']
        
        global_exc_cams_dict = {'ee_global' : int(num_clusters*exc_cluster_size*connectivity['ee_global'])}
        global_exc_cams = global_exc_cams_dict['ee_global']
        
        
        if print_detailed:
            print("Exc cluster neuron CAMs:")
            print("-> Input: ", exc_neuron_cams_dict['input'])
            print("-> EE recurrent: ", exc_neuron_cams_dict['ee_recurrent'])
            print("-> EE lateral: ", exc_neuron_cams_dict['ee_lateral'])
            print("-> IE: ", exc_neuron_cams_dict['ie'])
            print("Total: ", exc_neuron_cams)
            
            print("\nInh cluster neuron CAMs:")
            print("-> EI from exc cluster: ", inh_neuron_cams_dict['exc_cluster'])
            print("-> II recurrent: ", inh_neuron_cams_dict['ii_recurrent'])
            print("-> EI from global exc: ", inh_neuron_cams_dict['ei_global'])
            print("Total: ", inh_neuron_cams)
            
            print("\nGlobal exc neuron CAMs:")
            print("Total: ", global_exc_cams)
            
        
        return {'exc_neuron':exc_neuron_cams, 'inh_neuron':inh_neuron_cams, 'global_exc_cams':global_exc_cams}
        
    else:
        exc_neuron_cams_dict = {'input' : input_multiplier,
                                'ee_recurrent' : int(exc_cluster_size*connectivity['ee_recurrent']),
                                'ee_lateral' : int(2*exc_cluster_size*connectivity['ee_lateral']) + int(2*exc_cluster_size*connectivity['ee_lateral_2nd']) + int(2*exc_cluster_size*connectivity['ee_lateral_3rd']),
                                'ie' : int(inh_size*connectivity['ie'])
                                }
        exc_neuron_cams = exc_neuron_cams_dict['input'] + exc_neuron_cams_dict['ee_recurrent'] + exc_neuron_cams_dict['ee_lateral'] + exc_neuron_cams_dict['ie']
        
        
        inh_neuron_cams = int(exc_cluster_size*num_clusters*connectivity['ei']) + int(inh_size*connectivity['ii'])
        inh_neuron_cams_dict = {'ei' : int(exc_cluster_size*num_clusters*connectivity['ei']),
                            'ii_recurrent' : int(inh_size*connectivity['ii'])
                            }
        
        if print_detailed:
            print("Exc cluster neuron CAMs:")
            print("-> Input: ", exc_neuron_cams_dict['input'])
            print("-> EE recurrent: ", exc_neuron_cams_dict['ee_recurrent'])
            print("-> EE lateral: ", exc_neuron_cams_dict['ee_lateral'])
            print("-> IE: ", exc_neuron_cams_dict['ie'])
            print("Total: ", exc_neuron_cams)
            
            print("\nInh cluster neuron CAMs:")
            print("-> EI: ", inh_neuron_cams_dict['ei'])
            print("-> II recurrent: ", inh_neuron_cams_dict['ii_recurrent'])
            print("Total: ", inh_neuron_cams)
    
        return {'exc_neuron':exc_neuron_cams, 'inh_neuron':inh_neuron_cams}
       
        
def connect_populations(connector, pre_pop, post_pop, i, j, syn_type):
    """
    Wrapper function for connecting populations given lists of indices
    """
    assert len(i) == len(j), "index lists i and j must be of equal length"
    for index in range(len(i)):
        connector.add_connection(pre_pop[i[index]], post_pop[j[index]], syn_type)
    

def gaussian(mu, sigma, amplitude, inputSize):
    """
        Generates rates based on the gaussian profile
    """
    i = np.arange(inputSize)
    shift = mu % 1
    coarse = int(mu - shift)
#    dist = amplitude*np.exp(-np.power((i - shift - int(inputSize/2))/inputSize, 2.) / (2 * np.power(sigma, 2.)))
    dist = amplitude*np.max([np.exp(-np.power((i - shift - int(inputSize/2))/inputSize, 2.) /
                                    (2 * np.power(sigma, 2.))),
                np.exp(-np.power((i - shift - int(inputSize/2))/inputSize+1, 2.) /
                       (2 * np.power(sigma, 2.)))], axis = 0)
    return dist[(int(inputSize/2) - coarse + i)%inputSize]


def double2pop_code(value, inputSize, sigma=None, amplitude=100):
    """
        Generates rates based on the gaussian profile
    """
    if sigma is None:
        sigma = 1/inputSize
    
    mu = value*inputSize % inputSize
    activity = gaussian(mu, sigma, amplitude, inputSize)
    return activity


def pop_code2double(popArray):
    """Calculate circular mean of an array

    @author: Peter Diehl
    """
    size = len(popArray)
    complex_unit_roots = np.array(
        [np.exp(1j * (2 * np.pi / size) * cur_pos) for cur_pos in range(size)])
    cur_pos = (np.angle(np.sum(popArray * complex_unit_roots)) %
               (2 * np.pi)) / (2 * np.pi)
    return cur_pos
