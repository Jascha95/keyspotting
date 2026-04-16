import sys
sys.path.append("..")

import samna
import dynapse1utils as ut
import netgen
from netgen import Neuron
import numpy as np
import time


cam_actions = {(-1, 1) : "remove_gaba",
               (1, 1) : "add_ampa",
               (1, -1) : "remove_ampa",
               (-1, -1) : "add_gaba",
               (0, 1) : "add_ampa",
               (0, -1) : "add_gaba"}

class InteractiveNeuronPool:
    
    def __init__(self, model, chip_id, core_id, start_neuron, size, num_inp_neurons=None, net_gen=None, inp_matrix=None, matrix=None):
        
        assert start_neuron+size < 256, "ERROR: population allocation exceeds the size of the core!"
        
        self.model = model
        self.net_gen = net_gen
        if net_gen is None:
            self.net_gen=netgen.NetworkGenerator()
        
        self.chip_id=chip_id
        self.core_id=core_id
        self.start_neuron=start_neuron
        self.size=size
        self.matrix=np.zeros((size,size))
        self.input_matrix_exc=np.zeros((size,size))
        self.input_matrix_inh=np.zeros((size,size))

        if num_inp_neurons:
            self.num_inp_neurons=num_inp_neurons
            self.input_matrix=np.zeros((num_inp_neurons,size))
            self.input_neurons=[Neuron(0, n_id//256, n_id%256, is_spike_gen=True) for n_id in range(self.start_neuron, self.start_neuron+num_inp_neurons)]
        else:
            self.num_inp_neurons=None
            self.input_neurons_exc=[Neuron(0, (core_id+1)%4, n_id, is_spike_gen=True) for n_id in range(self.start_neuron, self.start_neuron+self.size)]
            self.input_neurons_inh=[Neuron(0, (core_id+2)%4, n_id, is_spike_gen=True) for n_id in range(self.start_neuron, self.start_neuron+self.size)]
        
        self.neurons=[Neuron(chip_id, core_id, n_id) for n_id in range(self.start_neuron, self.start_neuron+self.size)]
        
        #if inp_matrix is not None:
            #self.set_input_matrix(inp_matrix)
        if matrix is not None:
            self.set_matrix(matrix)
            
        if inp_matrix is None:
            for inp_neuron_exc, inp_neuron_inh, neuron in zip(self.input_neurons_exc, self.input_neurons_inh, self.neurons):
                net_gen.add_connection(inp_neuron_exc,
                                        neuron,
                                        samna.dynapse1.Dynapse1SynType.AMPA)
                net_gen.add_connection(inp_neuron_inh,
                                        neuron,
                                        samna.dynapse1.Dynapse1SynType.GABA_A)
        else:
            self.set_input_matrix(inp_matrix)
            
        new_config = self.net_gen.make_dynapse1_configuration()
        ut.apply_biases_from_configuration(self.model.get_configuration(), new_config)
        self.model.apply_configuration(new_config)
        
        
        self.poisson_spike_generator=model.get_poisson_gen()
        self.fpga_spike_gen=model.get_fpga_spike_gen()
        self.poisson_spike_generator.set_chip_id(self.chip_id)

        event_graph, event_filter_node, event_sink_node = ut.create_neuron_select_graph(self.model, [(chip_id, core_id, n_id) for n_id in range(self.start_neuron, self.start_neuron+self.size)])
        self.event_graph=event_graph
        self.event_filter_node=event_filter_node
        self.event_sink_node=event_sink_node
        
        self.event_graph.start()
        
        
        
    def set_matrix(self, matrix, debug=True):
        if debug:
            update_start_time=time.time()
        apply_matrix_diff(self.net_gen, self.neurons, self.neurons, self.matrix, matrix)
        self.matrix=matrix.copy()
        new_config = self.net_gen.make_dynapse1_configuration()
        ut.apply_biases_from_configuration(self.model.get_configuration(), new_config)
        self.model.apply_configuration(new_config)
        
        if debug:
            print("Matrix update time: ", time.time()-update_start_time)
    
    def set_input_matrix(self, matrix, exc_type="ampa", inh_type="gaba_b", debug=True):
        
        if debug:
            print("Setting input matrix: ", matrix)
        
        if debug:
            update_start_time=time.time()
        
        if self.num_inp_neurons:
            apply_matrix_diff(self.net_gen, self.input_neurons, self.neurons, self.input_matrix, matrix, exc_type=exc_type)
            self.input_matrix=matrix.copy()
        else:
            exc_matrix = np.clip(matrix, a_min=0, a_max=None)
            inh_matrix = np.clip(matrix, a_min=None, a_max=0)
                
            apply_matrix_diff(self.net_gen, self.input_neurons_exc, self.neurons, self.input_matrix_exc, exc_matrix, exc_type=exc_type)
            apply_matrix_diff(self.net_gen, self.input_neurons_inh, self.neurons, self.input_matrix_inh, inh_matrix, inh_type=inh_type)
            self.input_matrix_exc=exc_matrix.copy()
            self.input_matrix_inh=inh_matrix.copy()

        new_config = self.net_gen.make_dynapse1_configuration()
        ut.apply_biases_from_configuration(self.model.get_configuration(), new_config)
        self.model.apply_configuration(new_config)
        
        if debug:
            print("Input matrix update time: ", time.time()-update_start_time)
    
    def set_dc_level(self, dc_level_linear):
        ut.set_neuron_dc(self.model, self.chip_id, self.core_id, dc_level_linear)
        
    def set_neuron_tau1(self, tau_1_level_linear):
        ut.set_neuron_tau1(self.model, self.chip_id, self.core_id, tau_1_level_linear)
        
    def set_input_exc_weight(self, weight_linear):
        ut.set_ampa_weight(self.model, self.chip_id, self.core_id, weight_linear)
        
    def set_exc_weight(self, weight_linear):
        ut.set_nmda_weight(self.model, self.chip_id, self.core_id, weight_linear)
        
    def set_inh_weight(self, weight_linear):
        ut.set_gaba_b_weight(self.model, self.chip_id, self.core_id, weight_linear)
        
    def send_spike_pulse(self, rates_vector, duration_s):
        

        
        self.poisson_spike_generator.stop()
        if self.num_inp_neurons:
            assert len(rates_vector)==len(self.input_neurons), "ERROR: Input rates vector does not match the number of input neurons"
            for inp_neuron, rate in zip(self.input_neurons, rates_vector):
                self.poisson_spike_generator.write_poisson_rate_hz(inp_neuron.core_id*256+inp_neuron.neuron_id, rate)

        else:
            assert len(rates_vector)==len(self.input_neurons_exc), "ERROR: Input rates vector does not match the number of input neurons"
            assert len(rates_vector)==len(self.input_neurons_inh), "ERROR: Input rates vector does not match the number of input neurons"

            for inp_neuron, rate in zip(self.input_neurons_exc, rates_vector):
                if rate >=0:
                    self.poisson_spike_generator.write_poisson_rate_hz(inp_neuron.core_id*256+inp_neuron.neuron_id, rate)
                else:
                    self.poisson_spike_generator.write_poisson_rate_hz(inp_neuron.core_id*256+inp_neuron.neuron_id, 0)
                    
            for inp_neuron, rate in zip(self.input_neurons_inh, rates_vector):
                if rate <=0:
                    self.poisson_spike_generator.write_poisson_rate_hz(inp_neuron.core_id*256+inp_neuron.neuron_id, abs(rate))
                else:
                    self.poisson_spike_generator.write_poisson_rate_hz(inp_neuron.core_id*256+inp_neuron.neuron_id, 0)
            
        self.poisson_spike_generator.start()
        time.sleep(duration_s)
        self.poisson_spike_generator.stop()

    def set_preloaded_input(self, spiketrain, repeat_mode=False):

        spike_times, indices = spiketrain[:, 0], spiketrain[:, 1].astype(int)
        inp_neuron_indices = np.array([self.input_neurons[n_id].core_id*256 + self.input_neurons[n_id].neuron_id for n_id in indices])

        if self.poisson_spike_generator.is_running():
            self.poisson_spike_generator.stop()
        
        self.fpga_spike_gen.stop()
                
        ut.set_fpga_spike_gen(self.fpga_spike_gen,
                              spike_times=spike_times*1e-06,
                              indices=inp_neuron_indices.astype(int),
                              target_chips=[self.chip_id]*len(spike_times),
                              isi_base=900,
                              repeat_mode=repeat_mode)

        self.fpga_spike_gen.start()

    
    def start_recording(self):
        self.event_sink_node.get_events()
        
    def get_events(self):
        self.recorded_events=self.event_sink_node.get_events()
        self.raster_data=np.array([(evt.timestamp, evt.neuron_id) for evt in self.recorded_events])
        
        
    def get_rates(self, duration_s):
        self.start_recording()
        time.sleep(duration_s)
        self.get_events()
        
        self.rates=np.zeros(len(self.neurons))
        if len(self.recorded_events)>0:
            for evt in self.recorded_events:
                self.rates[evt.neuron_id - self.start_neuron]+=1

            self.rates=self.rates*(1/(duration_s))
            
    def plot_raster_data(self, filename=None):
        import matplotlib as plt
        raster_data = self.raster_data
        
        plt.figure(figsize=(10,6))
        plt.scatter(raster_data[:,0]*1e-06, raster_data[:,1]-self.start_neuron, marker='|', s=4)
        plt.title("Raster plot")
        plt.xlabel("time, s")
        plt.ylabel("neuron ID")
            
        

def apply_matrix_diff(net_gen, pre_neurons, post_neurons, old_matrix, new_matrix, exc_type="nmda", inh_type="gaba_b"):
    connections_added = 0
    connections_removed = 0
    
    if exc_type == "nmda":
        exc_syn_type=samna.dynapse1.Dynapse1SynType.NMDA
    else:
        exc_syn_type=samna.dynapse1.Dynapse1SynType.AMPA
        
    if inh_type == "gaba_b":
        inh_syn_type=samna.dynapse1.Dynapse1SynType.GABA_B
    else:
        inh_syn_type=samna.dynapse1.Dynapse1SynType.GABA_A
    
    for i_idx in range(len(pre_neurons)):
        for j_idx in range(len(post_neurons)):
            old_value = old_matrix[i_idx, j_idx]
            new_value = new_matrix[i_idx, j_idx]
            cam_delta = new_value - old_value
            
            if cam_delta != 0:
                cam_steps = np.sign(np.arange(old_value, new_value, np.sign(cam_delta)))
                for i in cam_steps:
                    action = cam_actions[i, np.sign(cam_delta)]
                    
                    if action == "add_ampa":
                        connections_added += 1
                        net_gen.add_connection(pre_neurons[i_idx],
                                               post_neurons[j_idx],
                                               exc_syn_type)
                    if action == "add_gaba":
                        connections_added += 1
                        net_gen.add_connection(pre_neurons[i_idx],
                                               post_neurons[j_idx],
                                               inh_syn_type)
                    if action == "remove_ampa":
                        connections_removed -= 1
                        net_gen.remove_connection(pre_neurons[i_idx],
                                                  post_neurons[j_idx],
                                                  exc_syn_type)
                    if action == "remove_gaba":
                        connections_removed -= 1
                        net_gen.remove_connection(pre_neurons[i_idx],
                                                  post_neurons[j_idx],
                                                  inh_syn_type)
                        
    print(f"Netgen updated. Connections added: {connections_added}, connections removed: {connections_removed}")
    #print("Update time: %g" % (time.time() - update_start_time))
        

def plot_matrix(matrix):
    pass


def plot_input_matrix(matrix):
    pass

def plot_activity_projection(data):
    pass