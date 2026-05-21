
import numpy as np
import time
import matplotlib.pyplot as plt

def generate_regular_rate_spiketrain(rates_vector, neuron_ids = None, duration_s = 1):

    if neuron_ids is not None:
        assert len(rates_vector) == len(neuron_ids), "ERROR: The size of the list of provided neruon ids does not match the rates vector!"
    else:
        neuron_ids = np.arange(len(rates_vector))

    # create empty spiketrain
    spiketrain=np.array([]).reshape(0,2)

    for firing_rate, neuron_id in zip(rates_vector, neuron_ids):
        # generate timestamps for each spike generator, shifted by a few microseconds
        # using the neuron index to avoid overlaps because timestamps cannot repeat
        timestamps = np.linspace(0, duration_s, firing_rate)+neuron_id*1e-06
        neuron_ids_extended = np.ones(len(timestamps)).astype(int)*neuron_id
        spiketrain = np.concatenate((spiketrain, np.column_stack((timestamps,neuron_ids_extended))))

    #fpga_spiketrain=np.array(fpga_spiketrain, np.column_stack((timestamps1,indices1)))
    spiketrain = spiketrain[spiketrain[:,0].argsort()]
    return spiketrain

def insert_dummy_spikes(spiketrain, dummy_neuron_id=1023):
    raise NotImplementedError("This function will be added soon.")


def get_rates_from_raster(timestamps_s, neuron_ids, expected_neuron_ids=None, measurement_interval_s=None):
    """ Compute individual firing rates given an input raster.

    Args:
        timestamps_s

        neuron_ids

        expected_neuron_ids

        measurement_interval_s
    
    """
    
    assert len(timestamps_s)==len(neuron_ids), "Invalid spiketrain: lengths of timestamp and neuron_ids arrays is mismatched"
    
    if measurement_interval_s is None:
        measurement_interval_s = timestamps_s[-1] - timestamps_s[0]

    if expected_neuron_ids is not None:
        rates = dict.fromkeys(expected_neuron_ids, 0)
    else:
        rates = {}

    for timestamp, neuron_id in zip(timestamps_s, neuron_ids):
        rates.setdefault(neuron_id, 0) #in case we encounter neuron_id that is not expected
        rates[neuron_id] += 1

    for n_id, evt_count in rates.items(): # normzalize by the time interval
            rates[n_id]=evt_count/measurement_interval_s

    return np.array(list(rates.values()))


def get_instantaneous_activity(timestamps_s, activity_dt_s : float, exp_tc_s : float):
    """ Compute instantaneous activity based on spike train provided using the exponential convolution
    kernel with the specified time constant.

    Args:
        timestamps_s (list): _description_
        activity_dt_s (float): _description_
        exp_tc_s (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    activity_data=[]
    activity_t=[]
    t_curr=timestamps_s[0]
    instantaneous_activity=0
    
    for timestamp in timestamps_s:
        while timestamp>= t_curr:
            activity_t.append(t_curr)
            activity_data.append(instantaneous_activity)
            t_curr += activity_dt_s
            instantaneous_activity=instantaneous_activity*np.exp(-activity_dt_s/exp_tc_s)
        
        instantaneous_activity += 1/exp_tc_s
    
    return np.array(activity_data), activity_t


def get_instantaneous_activity_array(raster_data, activity_dt, exp_tc_s, num_neurons):
    
    activity_data=[]
    activity_t=[]
    t_curr=raster_data[0,0]
    instantaneous_activity=np.zeros(num_neurons)
    
    for event in raster_data:
        while event[0]>= t_curr:
            activity_t.append(t_curr)
            activity_data.append(instantaneous_activity)
            t_curr += activity_dt
            instantaneous_activity=instantaneous_activity*np.exp(-activity_dt/exp_tc_s)
        
        instantaneous_activity[int(event[1])] += 1/exp_tc_s
    
    return np.array(activity_data), activity_t


def compute_vector_alignment(v_array, w_array, subtract_mean_v=False, subtract_mean_w=False):
    if v_array.ndim == 1 and w_array.ndim==2:
        v_array=np.array([v_array,]*len(w_array))
    
    rho=[]
    for v, w in zip(v_array, w_array):
        if subtract_mean_v:
            v = v.copy()
            v_no_mean = v - v.mean()
        if subtract_mean_w:
            w = w.copy()
            w_no_mean = w - w.mean()
        
        rho.append(np.dot(v/np.linalg.norm(v), w_no_mean/np.linalg.norm(w)))
        if np.isnan(rho[-1]):
            rho[-1]=0
    
    return np.array(rho)


def record_population_response(sink_node, fpga_spike_gen, num_neurons, plot=True):
    pre_stim_time=0.0 #s
    stim_time=1 #s
    post_stim_time=0 #s

    #reset the recording
    sink_node.get_events()

    #wait for any dynamics to stabilize
    time.sleep(pre_stim_time)

    #start the pulse
    fpga_spike_gen.start()
    time.sleep(stim_time)

    #stop the pulse
    fpga_spike_gen.stop()

    # wait a bit more after the stimulus is off
    time.sleep(post_stim_time)

    #get_the events
    events = sink_node.get_events()

    # sort and plot the events
    event_ids = np.array([evt.neuron_id for evt in events])
    timestamps_us = np.array([evt.timestamp for evt in events])
    timestamps_us[:]-=timestamps_us[0]
    timestamps_s = timestamps_us*1e-06


    #plt.figure(figsize=(18,2), dpi=150)
    #plt.plot(fpga_spiketrain[:,0], fpga_spiketrain[:,1], '.')
    #plt.xlabel("time (s)")
    #plt.ylabel("spikegen")
    
    if plot:

        plt.figure(figsize=(18,2), dpi=150)
        plt.scatter(timestamps_s, event_ids, marker='|')
        plt.xlabel("time (s)")
        plt.ylabel("neuron ID ")
        
        activity_dt = 0.01 #s
        exp_time_constant = 0.03 #s

        activity, activity_t = get_instantaneous_activity_array(timestamps_s,
                                                                activity_dt,
                                                                exp_time_constant)

        activity = activity/num_neurons

        plt.figure(figsize=(18,2), dpi=150)
        plt.plot(activity_t, activity)
        plt.xlabel("time (s)")
        plt.ylabel("Activity (Hz)")
        
        plt.show()
        
    return timestamps_s, event_ids
